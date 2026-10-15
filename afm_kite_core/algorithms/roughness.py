"""Pure roughness scope validation and masked-statistics algorithms."""

from __future__ import annotations

from typing import Any, Optional, Union

import numpy as np

from .statistics import surface_stats

_ROUGHNESS_SCOPE_ORDER = (
    "surface",
    "regions",
    "region_union",
    "background",
)


def normalise_roughness_scopes(
    scopes: Optional[Union[list[str], tuple[str, ...]]],
    regions_json_path: str,
) -> list[str]:
    """Validate requested scopes and return them in canonical order."""

    if scopes is None:
        return ["regions"] if regions_json_path else ["surface"]
    if isinstance(scopes, str):
        raise ValueError(
            "scopes must be a list containing surface, regions, "
            "region_union, or background"
        )

    requested = {str(scope).strip().lower() for scope in scopes if str(scope).strip()}
    invalid = requested.difference(_ROUGHNESS_SCOPE_ORDER)
    if invalid:
        raise ValueError(
            "Unknown roughness scopes: {}. Expected surface, regions, "
            "region_union, or background.".format(
                ", ".join(sorted(invalid))
            )
        )
    if not requested:
        raise ValueError("scopes must contain at least one requested roughness scope")
    if requested.intersection(
        {"regions", "region_union", "background"}
    ) and not regions_json_path:
        raise ValueError(
            "regions, region_union, and background scopes require a "
            "regions_json_path"
        )
    return [scope for scope in _ROUGHNESS_SCOPE_ORDER if scope in requested]


def masked_roughness_stats(
    matrix: np.ndarray,
    mask: np.ndarray,
    contaminant_label_image: np.ndarray,
    contaminants: list[dict[str, Any]],
    pixel_area: float | None,
) -> dict[str, Any]:
    """Measure roughness within a mask before and after outlier exclusion."""

    stats = surface_stats(matrix[mask])
    area_px = int(np.sum(mask))
    contaminant_mask = mask & (contaminant_label_image > 0)
    contaminant_area_px = int(np.sum(contaminant_mask))
    clean_stats = surface_stats(matrix[mask & ~contaminant_mask])
    return {
        **stats,
        "area_px": area_px,
        "area_physical": float(area_px * pixel_area) if pixel_area else None,
        "contaminant_count": len(contaminants),
        "contaminant_area_px": contaminant_area_px,
        "contaminant_area_physical": (
            float(contaminant_area_px * pixel_area) if pixel_area else None
        ),
        "contaminant_fraction": (
            float(contaminant_area_px / area_px) if area_px else 0.0
        ),
        "clean_mean_height": clean_stats.get("mean_height"),
        "clean_median_height": clean_stats.get("median_height"),
        "clean_Sa": clean_stats.get("Sa"),
        "clean_Sq": clean_stats.get("Sq"),
        "clean_Sz": clean_stats.get("Sz"),
    }


__all__ = [
    "masked_roughness_stats",
    "normalise_roughness_scopes",
]
