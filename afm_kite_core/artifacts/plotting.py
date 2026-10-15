"""Headless Matplotlib and writable cache configuration for rendering stages."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from uuid import uuid4

from afm_kite_core.identity import PRODUCT_SLUG

DEFAULT_CACHE_ROOT = Path.home() / ".cache" / PRODUCT_SLUG
PLOT_DPI = 600
PLOT_FONT_FAMILY = "Arial"
PLOT_TITLE_PAD = 6
PLOT_SAVEFIG_BBOX = "tight"
SCALE_BAR_POSITION = "outside_lower_right"
SCALE_BAR_RIGHT_AXES_FRACTION = 1.0
SCALE_BAR_LINE_Y_AXES_FRACTION = -0.08
SCALE_BAR_LABEL_GAP_AXES_FRACTION = 0.02


def _writable_directory(path: Path) -> bool:
    """Create ``path`` and verify that the current process can write to it."""

    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / f".write-test-{uuid4().hex}"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError:
        return False
    return True


def _select_cache_root() -> Path:
    requested = Path(
        os.environ.get("AFMDATA_CACHE_ROOT", DEFAULT_CACHE_ROOT)
    ).expanduser().resolve()
    if _writable_directory(requested):
        return requested
    fallback = Path(tempfile.gettempdir()) / f"{PRODUCT_SLUG}-cache"
    if not _writable_directory(fallback):
        raise OSError(
            f"Neither the requested cache root nor the fallback is writable: {requested}"
        )
    return fallback.resolve()


CACHE_ROOT = _select_cache_root()
MPL_CACHE = CACHE_ROOT / "matplotlib"
XDG_CACHE = CACHE_ROOT / "xdg"
MPL_CACHE.mkdir(parents=True, exist_ok=True)
XDG_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))
os.environ.setdefault("XDG_CACHE_HOME", str(XDG_CACHE))

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update(
    {
        "figure.dpi": PLOT_DPI,
        "savefig.dpi": PLOT_DPI,
        "savefig.bbox": PLOT_SAVEFIG_BBOX,
        "axes.titlepad": PLOT_TITLE_PAD,
        "axes.titleweight": "bold",
        "font.family": PLOT_FONT_FAMILY,
        "font.sans-serif": [PLOT_FONT_FAMILY],
    }
)

from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.ticker import AutoLocator  # noqa: E402
from mpl_toolkits.axes_grid1 import make_axes_locatable  # noqa: E402


AFM_GOLD_SAMPLED_STOPS = [
    (0.00, "#000004"),
    (0.05, "#000004"),
    (0.10, "#080004"),
    (0.15, "#160004"),
    (0.20, "#240004"),
    (0.25, "#310004"),
    (0.30, "#3F0004"),
    (0.40, "#5B0004"),
    (0.50, "#742704"),
    (0.60, "#8F5304"),
    (0.70, "#A97D27"),
    (0.80, "#C6A76D"),
    (0.90, "#E1D2B4"),
    (0.95, "#EFE7D8"),
    (1.00, "#FDFCFB"),
]
AFM_GOLD_COLORMAP_NAME = "afm_gold_sampled"
COLORMAP = LinearSegmentedColormap.from_list(
    AFM_GOLD_COLORMAP_NAME,
    AFM_GOLD_SAMPLED_STOPS,
    N=256,
)
PSEUDOCOLOR_FIGURE_SIZE = (6, 5)
PSEUDOCOLOR_ORIGIN = "lower"
PSEUDOCOLOR_PERCENTILE_CLIP = 1.0
COLORBAR_FRACTION = 0.038
COLORBAR_PAD = 0.12
COLORBAR_EXTREME_FONT_SIZE = 8


def resolve_afm_colormap(value: str):
    """Resolve the shared AFM map while preserving explicit caller overrides."""

    return COLORMAP if not value or value == AFM_GOLD_COLORMAP_NAME else value


def render_height_image(ax, matrix, **kwargs):
    """Render one height-derived image with the shared AFM orientation."""

    kwargs["origin"] = PSEUDOCOLOR_ORIGIN
    return ax.imshow(matrix, **kwargs)


def canonical_plot_title(title: object, *, channel_hint: object = "") -> str:
    """Return the suite-wide canonical title for channel-derived plots."""

    resolved_title = str(title or "").strip()
    source_text = f"{resolved_title} {channel_hint}".strip()
    normalized_source = "".join(
        character
        for character in source_text.casefold()
        if character.isalnum()
    )
    if "dcbias" not in normalized_source:
        return resolved_title
    difference_markers = ("difference", "residual", "delta", "minus", "Δ")
    is_difference = any(
        marker.casefold() in source_text.casefold()
        for marker in difference_markers
    )
    return "DCbias Difference" if is_difference else "DCbias"


def set_plot_title(ax, title: str, **kwargs):
    """Set a plot title with the suite-wide bold style and spacing."""

    kwargs["pad"] = PLOT_TITLE_PAD
    kwargs["fontweight"] = "bold"
    return ax.set_title(canonical_plot_title(title), **kwargs)


def save_figure(
    fig,
    path,
    *,
    dpi: float | None = None,
    pad_inches: float | None = None,
) -> None:
    """Save a figure with the suite-wide tight bounding-box contract."""

    options: dict[str, object] = {"bbox_inches": PLOT_SAVEFIG_BBOX}
    if dpi is not None:
        options["dpi"] = dpi
    if pad_inches is not None:
        options["pad_inches"] = pad_inches
    fig.savefig(path, **options)


def _positive_number(value: object) -> float | None:
    """Return a finite positive float, or ``None`` for unusable metadata."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not number > 0 or not number < float("inf"):
        return None
    return number


def _nice_scale_length(span: float, target_fraction: float = 0.2) -> float:
    """Choose a compact 1/2/5 scale-bar length near ``target_fraction``."""

    import math

    positive_span = _positive_number(span)
    if positive_span is None:
        return 1.0
    target = positive_span * float(target_fraction)
    exponent = math.floor(math.log10(target))
    base = 10.0**exponent
    candidates = [base, 2.0 * base, 5.0 * base, 10.0 * base]
    valid = [candidate for candidate in candidates if candidate <= target]
    return max(valid) if valid else base


def _format_scale_value(value: float) -> str:
    """Format a scale length without distracting trailing zeroes."""

    return f"{float(value):.4g}"


def _normalized_unit(value: object, default: str = "μm") -> str:
    """Return a compact canonical unit label."""

    text = str(value or default).strip().replace("µ", "μ")
    aliases = {
        "micrometer": "μm",
        "micrometers": "μm",
        "micrometre": "μm",
        "micrometres": "μm",
        "micron": "μm",
        "microns": "μm",
        "nanometer": "nm",
        "nanometers": "nm",
        "nanometre": "nm",
        "nanometres": "nm",
        "picometer": "pm",
        "picometers": "pm",
        "picometre": "pm",
        "picometres": "pm",
    }
    return aliases.get(text.casefold(), text)


def _unit_scale_to_micrometres(unit: object) -> float | None:
    """Return the multiplier that converts ``unit`` values to micrometres."""

    normalized = _normalized_unit(unit)
    return {
        "pm": 1.0e-6,
        "nm": 1.0e-3,
        "μm": 1.0,
        "mm": 1.0e3,
        "m": 1.0e6,
    }.get(normalized)


def _height_like(value_label: object) -> bool:
    """Return whether a semantic value label represents an AFM height channel."""

    normalized = "".join(
        character
        for character in str(value_label or "").casefold().replace("µ", "μ")
        if character.isalnum()
    )
    for base_label in ("height", "heightsensor"):
        if normalized == base_label:
            return True
        if normalized.startswith(base_label):
            unit_suffix = normalized[len(base_label) :]
            if unit_suffix in {"pm", "nm", "μm", "um", "mm", "m"}:
                return True
    return False


def canonical_plot_value_label(
    value_label: object,
    *,
    default: str = "value",
) -> str:
    """Return the shared plot label, removing units from height-channel titles."""

    resolved_label = str(value_label or default).strip()
    return "Height" if _height_like(resolved_label) else resolved_label


def _robust_span(matrix, percentile_clip: float = PSEUDOCOLOR_PERCENTILE_CLIP) -> float:
    """Return the finite percentile span used for automatic height units."""

    import numpy as np

    values = np.asarray(matrix, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0
    clip = min(max(float(percentile_clip), 0.0), 49.0)
    lower, upper = np.percentile(finite, [clip, 100.0 - clip])
    return float(upper - lower)


def _adaptive_height_display(
    matrix,
    *,
    value_label: str,
    value_unit: str,
) -> tuple[object, str, float, str]:
    """Scale a height matrix to the notebook's automatic μm/nm display unit."""

    import numpy as np

    resolved_unit = _normalized_unit(value_unit)
    if not _height_like(value_label):
        return matrix, resolved_unit, 1.0, "source"
    to_micrometres = _unit_scale_to_micrometres(resolved_unit)
    if to_micrometres is None:
        return matrix, resolved_unit, 1.0, "source"
    span_micrometres = _robust_span(matrix) * to_micrometres
    if span_micrometres >= 1.0:
        display_unit = "μm"
        scale_factor = to_micrometres
    else:
        display_unit = "nm"
        scale_factor = to_micrometres * 1000.0
    return (
        np.asanyarray(matrix, dtype=float) * scale_factor,
        display_unit,
        float(scale_factor),
        "auto",
    )


def _adaptive_axis_context(
    matrix_shape: tuple[int, int],
    physical_context: dict[str, object] | None,
) -> dict[str, object]:
    """Resolve notebook-style physical spans and a common μm/nm axis unit."""

    rows, cols = (int(matrix_shape[0]), int(matrix_shape[1]))
    context = physical_context or {}
    raw_pixel_width = _positive_number(context.get("pixel_width"))
    raw_pixel_height = _positive_number(context.get("pixel_height"))
    pixel_width = raw_pixel_width or 1.0
    pixel_height = raw_pixel_height or 1.0
    raw_x_unit = _normalized_unit(context.get("x_unit") or "μm")
    raw_y_unit = _normalized_unit(context.get("y_unit") or "μm")
    x_to_micrometres = _unit_scale_to_micrometres(raw_x_unit)
    y_to_micrometres = _unit_scale_to_micrometres(raw_y_unit)
    units_defaulted = bool(
        context.get("units_defaulted")
        or raw_pixel_width is None
        or raw_pixel_height is None
        or not context.get("x_unit")
        or not context.get("y_unit")
    )

    raw_x_span = float(cols) * pixel_width
    raw_y_span = float(rows) * pixel_height
    if x_to_micrometres is not None and y_to_micrometres is not None:
        x_span_micrometres = raw_x_span * x_to_micrometres
        y_span_micrometres = raw_y_span * y_to_micrometres
        display_unit = (
            "nm"
            if max(x_span_micrometres, y_span_micrometres) < 1.0
            else "μm"
        )
        display_scale = 1000.0 if display_unit == "nm" else 1.0
        x_span = x_span_micrometres * display_scale
        y_span = y_span_micrometres * display_scale
        x_unit = display_unit
        y_unit = display_unit
        display_mode = "auto"
    else:
        x_span = raw_x_span
        y_span = raw_y_span
        x_unit = raw_x_unit
        y_unit = raw_y_unit
        display_mode = "source"

    return {
        "rows": rows,
        "cols": cols,
        "x_span": float(x_span),
        "y_span": float(y_span),
        "x_unit": x_unit,
        "y_unit": y_unit,
        "display_mode": display_mode,
        "defaulted": units_defaulted,
    }


def _physical_ticks(span: float, pixel_count: int) -> tuple[list[float], list[str]]:
    """Map Matplotlib automatic physical ticks back onto pixel data coordinates."""

    import numpy as np

    locator = AutoLocator()
    tick_values = np.asarray(locator.tick_values(0.0, float(span)), dtype=float)
    epsilon = max(abs(float(span)), 1.0) * 1.0e-12
    tick_values = tick_values[
        (tick_values >= -epsilon) & (tick_values <= float(span) + epsilon)
    ]
    positions = -0.5 + tick_values * float(pixel_count) / float(span)
    return positions.tolist(), [_format_scale_value(value) for value in tick_values]


def _colorbar_extreme_label(value: float, unit: str) -> str:
    """Format the notebook-style top or bottom colorbar annotation."""

    suffix = f" {unit}" if unit else ""
    return f"{float(value):.3f}{suffix}"


def style_colorbar_extreme_labels(
    colorbar,
    *,
    minimum_label: str,
    maximum_label: str,
    font_size: float = COLORBAR_EXTREME_FONT_SIZE,
) -> None:
    """Apply the shared AFM colorbar style with bold endpoint annotations."""

    colorbar.set_ticks([])
    colorbar.ax.tick_params(
        which="both",
        left=False,
        right=False,
        labelleft=False,
        labelright=False,
    )
    colorbar.outline.set_linewidth(0.8)
    colorbar.ax.text(
        0.5,
        1.02,
        maximum_label,
        transform=colorbar.ax.transAxes,
        fontsize=font_size,
        fontweight="bold",
        ha="center",
        va="bottom",
        clip_on=False,
    )
    colorbar.ax.text(
        0.5,
        -0.02,
        minimum_label,
        transform=colorbar.ax.transAxes,
        fontsize=font_size,
        fontweight="bold",
        ha="center",
        va="top",
        clip_on=False,
    )


def decorate_height_map(
    fig,
    ax,
    image,
    *,
    matrix_shape: tuple[int, int],
    physical_context: dict[str, object] | None = None,
    value_label: str = "Height",
    value_unit: str = "μm",
    colorbar_fraction: float = COLORBAR_FRACTION,
    colorbar_pad: float = COLORBAR_PAD,
) -> dict[str, object]:
    """Add notebook-style AFM colorbar, physical axes, and visible scale bar.

    Height-derived images use physical lateral units from scan metadata. Missing
    dimensions or units use the suite-wide 1 μm/pixel and μm defaults.
    """

    context = physical_context or {}
    axes_context = _adaptive_axis_context(matrix_shape, context)
    rows = int(axes_context["rows"])
    cols = int(axes_context["cols"])
    x_span = float(axes_context["x_span"])
    y_span = float(axes_context["y_span"])
    x_unit = str(axes_context["x_unit"])
    y_unit = str(axes_context["y_unit"])
    units_defaulted = bool(axes_context["defaulted"])

    x_ticks, x_tick_labels = _physical_ticks(x_span, cols)
    y_ticks, y_tick_labels = _physical_ticks(y_span, rows)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_tick_labels)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_tick_labels)
    ax.set_xlabel(x_unit)
    ax.set_ylabel(y_unit)
    pixel_aspect = (
        (y_span / max(float(rows), 1.0))
        / (x_span / max(float(cols), 1.0))
    )
    ax.set_aspect(pixel_aspect)

    resolved_value_label = canonical_plot_value_label(
        value_label,
        default="Height",
    )
    source_value_unit = _normalized_unit(
        value_unit or context.get("height_unit") or "μm"
    )
    display_matrix, resolved_value_unit, value_scale_factor, value_display_mode = (
        _adaptive_height_display(
            image.get_array(),
            value_label=resolved_value_label,
            value_unit=source_value_unit,
        )
    )
    if value_scale_factor != 1.0:
        image.set_data(display_matrix)
        current_vmin, current_vmax = image.get_clim()
        image.set_clim(
            None if current_vmin is None else current_vmin * value_scale_factor,
            None if current_vmax is None else current_vmax * value_scale_factor,
        )
    colorbar_label = (
        f"{resolved_value_label} ({resolved_value_unit})"
        if resolved_value_unit
        else resolved_value_label
    )
    divider = make_axes_locatable(ax)
    colorbar_axis = divider.append_axes(
        "right",
        size=f"{100.0 * float(colorbar_fraction):g}%",
        pad=f"{100.0 * float(colorbar_pad):g}%",
    )
    colorbar = fig.colorbar(image, cax=colorbar_axis)
    display_vmin, display_vmax = image.get_clim()
    minimum_label = _colorbar_extreme_label(display_vmin, resolved_value_unit)
    maximum_label = _colorbar_extreme_label(display_vmax, resolved_value_unit)
    style_colorbar_extreme_labels(
        colorbar,
        minimum_label=minimum_label,
        maximum_label=maximum_label,
    )

    scale_value = _nice_scale_length(x_span)
    scale_unit = x_unit
    scale_fraction = min(max(scale_value / x_span, 0.04), 0.4)
    scale_right = SCALE_BAR_RIGHT_AXES_FRACTION
    scale_left = scale_right - scale_fraction
    scale_y = SCALE_BAR_LINE_Y_AXES_FRACTION
    scale_label_y = scale_y - SCALE_BAR_LABEL_GAP_AXES_FRACTION
    ax.plot(
        [scale_left, scale_right],
        [scale_y, scale_y],
        transform=ax.transAxes,
        color="black",
        linewidth=1.5,
        solid_capstyle="butt",
        clip_on=False,
        zorder=20,
    )
    ax.text(
        (scale_left + scale_right) / 2.0,
        scale_label_y,
        f"{_format_scale_value(scale_value)} {scale_unit}",
        transform=ax.transAxes,
        color="black",
        fontsize=8,
        fontweight="bold",
        ha="center",
        va="top",
        clip_on=False,
        zorder=20,
    )

    return {
        "colorbar": {
            "label": colorbar_label,
            "value_unit": resolved_value_unit,
            "source_value_unit": source_value_unit,
            "display_mode": value_display_mode,
            "minimum": float(display_vmin),
            "maximum": float(display_vmax),
            "minimum_label": minimum_label,
            "maximum_label": maximum_label,
            "visible_side_label": False,
            "ticks": [],
            "fraction": float(colorbar_fraction),
            "pad": float(colorbar_pad),
            "extreme_font_size": COLORBAR_EXTREME_FONT_SIZE,
        },
        "scale_bar": {
            "length": float(scale_value),
            "unit": scale_unit,
            "physical": True,
            "position": SCALE_BAR_POSITION,
            "right_axes_fraction": float(scale_right),
            "line_y_axes_fraction": float(scale_y),
            "label_y_axes_fraction": float(scale_label_y),
            "label_gap_axes_fraction": SCALE_BAR_LABEL_GAP_AXES_FRACTION,
            "defaulted": units_defaulted,
        },
        "axes": {
            "x_unit": x_unit,
            "y_unit": y_unit,
            "x_label": x_unit,
            "y_label": y_unit,
            "x_span": x_span,
            "y_span": y_span,
            "pixel_aspect": float(pixel_aspect),
            "display_mode": axes_context["display_mode"],
            "origin": PSEUDOCOLOR_ORIGIN,
            "defaulted": units_defaulted,
        },
    }


__all__ = [
    "AFM_GOLD_COLORMAP_NAME",
    "AFM_GOLD_SAMPLED_STOPS",
    "CACHE_ROOT",
    "COLORBAR_EXTREME_FONT_SIZE",
    "COLORBAR_FRACTION",
    "COLORBAR_PAD",
    "COLORMAP",
    "MPL_CACHE",
    "PLOT_DPI",
    "PLOT_FONT_FAMILY",
    "PLOT_SAVEFIG_BBOX",
    "PLOT_TITLE_PAD",
    "PSEUDOCOLOR_FIGURE_SIZE",
    "PSEUDOCOLOR_ORIGIN",
    "PSEUDOCOLOR_PERCENTILE_CLIP",
    "SCALE_BAR_LABEL_GAP_AXES_FRACTION",
    "SCALE_BAR_LINE_Y_AXES_FRACTION",
    "SCALE_BAR_POSITION",
    "SCALE_BAR_RIGHT_AXES_FRACTION",
    "XDG_CACHE",
    "canonical_plot_title",
    "canonical_plot_value_label",
    "decorate_height_map",
    "plt",
    "render_height_image",
    "resolve_afm_colormap",
    "save_figure",
    "set_plot_title",
    "style_colorbar_extreme_labels",
]
