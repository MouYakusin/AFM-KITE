"""Atomic tabular and Markdown writers for periodic measurements."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from afm_kite_core.artifacts.writers import write_text_file


def write_segments_table(path: Path, segments: list[dict[str, Any]]) -> None:
    """Write periodic high/low segment records as one atomic TSV artifact."""

    lines = [
        "segment_id\ttype\tstart_px\tend_px_exclusive\tcenter_px\t"
        "width_px\tmedian_height\tedge_touched"
    ]
    lines.extend(
        (
            "{segment_id}\t{type}\t{start_px}\t{end_px_exclusive}\t"
            "{center_px}\t{width_px}\t{median_height}\t{edge_touched}"
        ).format(**segment)
        for segment in segments
    )
    write_text_file(path, "\n".join(lines) + "\n")


def write_periodic_profile_report(
    report_path: Path,
    payload: dict[str, Any],
) -> None:
    """Write the human-readable periodic profile measurement report."""

    summary = payload["measurement_summary"]
    lines = [
        "# Periodic Profile Measurement Report",
        "",
        f"- Input profile: `{payload['profile_data_path']}`",
        f"- Profile length: {payload['profile_length']}",
        f"- Axis unit: {payload['axis_unit']}",
        f"- Height unit: {payload['height_unit']}",
        "",
        "## Summary",
        "",
        "| metric | value | unit |",
        "|---|---:|---|",
        f"| period_pitch | {summary['period']} | {payload['axis_unit']} |",
        f"| period_pitch_px | {summary['period_px']} | px |",
        f"| high_width | {summary['high_width']} | {payload['axis_unit']} |",
        f"| low_width | {summary['low_width']} | {payload['axis_unit']} |",
        (
            f"| depth_median_high_low | {summary['depth_median_high_low']} | "
            f"{payload['height_unit']} |"
        ),
        (
            f"| depth_percentile_high_low | {summary['depth_percentile_high_low']} | "
            f"{payload['height_unit']} |"
        ),
        f"| duty_cycle_high | {summary['duty_cycle_high']} | fraction |",
        f"| duty_cycle_low | {summary['duty_cycle_low']} | fraction |",
        "",
        "## Segments",
        "",
        "| segment | type | start_px | end_px | center_px | width_px | median_height | edge |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for segment in payload["segments"]:
        lines.append(
            (
                "| {segment_id} | {type} | {start_px} | {end_px_exclusive} | "
                "{center_px} | {width_px} | {median_height} | {edge_touched} |"
            ).format(**segment)
        )
    write_text_file(report_path, "\n".join(lines) + "\n")


__all__ = ["write_periodic_profile_report", "write_segments_table"]
