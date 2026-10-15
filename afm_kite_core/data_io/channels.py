"""AFM text decoding and multi-channel parsing helpers."""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from .paths import resolve_readable_file

NUMERIC_SPLIT_RE = re.compile(r"[\t,\s]+")


def read_text(path: str | Path) -> str:
    """Read text with AFM-export friendly encoding fallbacks."""

    data = Path(path).read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def parse_numeric_row(line: str) -> list[float] | None:
    """Parse one delimited numeric row, returning ``None`` for headers."""

    parts = [part for part in NUMERIC_SPLIT_RE.split(line.strip()) if part]
    if len(parts) < 2:
        return None
    try:
        return [float(part) for part in parts]
    except ValueError:
        return None


def _extract_float(line: str, key: str) -> float | None:
    match = re.search(
        rf"{re.escape(key)}\s*:\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?)",
        line,
    )
    return float(match.group(1)) if match else None


def _parse_base_metadata(lines: list[str]) -> dict[str, Any]:
    metadata: dict[str, Any] = {"Resolution": {}, "ScanTime": {}}
    for line in lines:
        stripped = line.strip().rstrip(";")
        if stripped == "Data:" or stripped.startswith("{"):
            break
        if stripped.startswith("Resolution ratio:"):
            resolution = metadata["Resolution"]
            values = {
                "XResolution": _extract_float(stripped, "Xresolution"),
                "YResolution": _extract_float(stripped, "Yresolution"),
                "XSize": _extract_float(stripped, "Xsize"),
                "YSize": _extract_float(stripped, "Ysize"),
                "Angle": _extract_float(stripped, "Scan angle"),
            }
            if values["XResolution"] is not None:
                resolution["XResolution"] = int(round(values["XResolution"]))
            if values["YResolution"] is not None:
                resolution["YResolution"] = int(round(values["YResolution"]))
            if values["XSize"] is not None:
                resolution.update({"XSize": values["XSize"], "XUnit": "um"})
            if values["YSize"] is not None:
                resolution.update({"YSize": values["YSize"], "YUnit": "um"})
            if values["Angle"] is not None:
                resolution["Angle"] = values["Angle"]
            continue
        if stripped.startswith("StartTime:"):
            metadata["ScanTime"]["StartTime"] = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("EndTime:"):
            metadata["ScanTime"]["EndTime"] = stripped.split(":", 1)[1].strip()
    return {key: value for key, value in metadata.items() if value}


def _parse_plain_channel_header(line: str) -> dict[str, Any] | None:
    if "ScanMode:" not in line or "ImageShowMode:" not in line:
        return None
    mode_info = {
        key.strip(): value.strip()
        for part in line.strip().rstrip(";").split(",")
        if ":" in part
        for key, value in [part.split(":", 1)]
    }
    return {"ModeInfo": mode_info} if "ImageShowMode" in mode_info else None


def _flush_channel(
    channels: list[dict[str, Any]],
    metadata: dict[str, Any] | None,
    rows: list[list[float]],
    *,
    metadata_source: str,
    data_start_line: int | None,
    require_square: bool,
) -> None:
    if metadata is None:
        return
    if not rows:
        raise ValueError(f"Channel {len(channels) + 1} has metadata but no numeric rows")
    widths = {len(row) for row in rows}
    if len(widths) != 1:
        raise ValueError(
            f"Channel {len(channels) + 1} has inconsistent row widths: {sorted(widths)}"
        )
    matrix = np.asarray(rows, dtype=float)
    if matrix.ndim != 2 or matrix.size == 0:
        raise ValueError(f"Channel {len(channels) + 1} is empty")
    resolution = metadata.get("Resolution", {})
    expected = (resolution.get("YResolution"), resolution.get("XResolution"))
    if all(expected) and matrix.shape != tuple(int(item) for item in expected):
        raise ValueError(
            f"Channel {len(channels) + 1} shape {matrix.shape} does not match metadata "
            f"{tuple(int(item) for item in expected)}"
        )
    if require_square and matrix.shape[0] != matrix.shape[1]:
        raise ValueError(
            f"Channel {len(channels) + 1} is not square: {matrix.shape}. "
            "This usually means multiple channels were accidentally merged."
        )
    channels.append(
        {
            "metadata": metadata,
            "matrix": matrix,
            "metadata_source": metadata_source,
            "data_start_line": 0 if data_start_line is None else int(data_start_line),
        }
    )


def parse_afm_channels(
    path: str | Path,
    require_square: bool = False,
) -> list[dict[str, Any]]:
    """Parse one AFM text export into one or more channel records."""

    txt_path = resolve_readable_file(path)
    lines = read_text(txt_path).splitlines()
    base_metadata = _parse_base_metadata(lines)
    channels: list[dict[str, Any]] = []
    metadata: dict[str, Any] | None = None
    metadata_source = ""
    data_start_line: int | None = None
    rows: list[list[float]] = []

    for line_no, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped or stripped == "Data:":
            continue
        if stripped.startswith("{"):
            _flush_channel(
                channels,
                metadata,
                rows,
                metadata_source=metadata_source,
                data_start_line=data_start_line,
                require_square=require_square,
            )
            try:
                metadata = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON metadata at line {line_no}: {exc}") from exc
            metadata_source, data_start_line, rows = "json", None, []
            continue
        header = _parse_plain_channel_header(stripped)
        if header is not None:
            _flush_channel(
                channels,
                metadata,
                rows,
                metadata_source=metadata_source,
                data_start_line=data_start_line,
                require_square=require_square,
            )
            metadata = copy.deepcopy(base_metadata)
            metadata.update(header)
            metadata_source, data_start_line, rows = "plain_header", None, []
            continue
        values = parse_numeric_row(stripped)
        if values is None:
            continue
        if metadata is None:
            metadata = copy.deepcopy(base_metadata)
            metadata["ModeInfo"] = {"ImageShowMode": "matrix"}
            metadata_source = "implicit"
        if data_start_line is None:
            data_start_line = line_no - 1
        rows.append(values)

    _flush_channel(
        channels,
        metadata,
        rows,
        metadata_source=metadata_source,
        data_start_line=data_start_line,
        require_square=require_square,
    )
    if not channels:
        raise ValueError(f"No numeric AFM matrix found in {txt_path}")
    return channels


__all__ = ["parse_afm_channels", "parse_numeric_row", "read_text"]
