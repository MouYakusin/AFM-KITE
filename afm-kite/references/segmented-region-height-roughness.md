# `segmented_region_height_roughness`

Use for value-defined height regions and requested per-region mean height, Sa, Sq, combined segmented-region-union Sa/Sq, local-outlier assessment, or explicitly requested whole-matrix or label-zero-background statistics.

## Supported Sample Types

- HS100-MG and other patterned or calibration height maps with value-separable low-height or high-height regions.
- Surfaces containing separable steps, islands, particles, pillars, pits, pores, or comparable regions defined by height values.

Use the workflow only when the requested measurement scope is compatible with matrix, segmented-region, combined region-union, ranked-region, background, or outlier statistics. Sample type alone is not sufficient.

## Contents

- Workflow Card
- Optional Review Stages
- Domain and Scope Rules
- Ranked Region Selection
- Quality and Measurement Rules
- Calculation and Reporting Scope

## Workflow Card

```text
data_ingest
├── pseudocolor_render(source) [optional]
└── matrix_level(trend_mode="plane")
    ├── pseudocolor_render(leveled) [optional]
    ├── matrix_roughness_statistics(["matrix"]) [when requested alone]
    └── value_regions_segment(polarity=<target semantics>, write_plots=<requested>) [when needed]
        ├── local_outliers_detect [when scientifically requested]
        └── matrix_roughness_statistics(
                statistics_scopes=<requested>,
                regions_data_path=regions,
                outliers_data_path=<when required>
            )
            └── rank returned region records [when requested]
```

| Stage | Required call and inputs | Output or rule |
|---|---|---|
| ingest | `data_ingest` once for the physical input | selected height-like matrix, matching metadata, manifest |
| source preview | `pseudocolor_render`; ingested height matrix | optional `source` render only |
| level | `matrix_level`; selected matrix; `trend_mode="plane"`; matching metadata | one detrended matrix shared by every requested statistics scope |
| leveled preview | `pseudocolor_render`; leveled matrix | optional `leveled` render only |
| regions | `value_regions_segment`; leveled matrix; matching metadata; target-semantic polarity; `write_plots=<requested>` | stable positive target labels whose numeric sign does not encode height polarity; label zero is background; structured segmentation quality; optional categorical mask and height overlay |
| outliers | `local_outliers_detect`; leveled matrix; matching metadata; regions and explicit scope when required | optional scientific result; its mask and overlay are inherent outputs of this callable |
| statistics | `matrix_roughness_statistics`; leveled matrix; exactly requested scopes; regions/outliers only when required | mean height, Sa, Sq, area, identity, and supporting statistics for every positive-label region when `regions` is requested, or one pooled result over all positive-label pixels when `region_union` is requested |

## Optional Review Stages

Treat the first three requested review stages as independently selectable:

- `source`
- `leveled`
- `segmented_regions_overlay`
- `local_outliers_overlay` [only when the outlier-analysis branch is required]

When no review stages are requested, do not call `pseudocolor_render` and set `value_regions_segment(write_plots=false)`. Generate `source` and `leveled` only after the corresponding numeric matrix exists, using:

| Stage | `value_label` | `plot_title` | `color_scale_mode` |
|---|---|---|---|
| `source` | `Height` | `Height` | `auto` |
| `leveled` | `Height` | `Height` | `auto` |

Generate `segmented_regions_overlay` by setting `value_regions_segment(write_plots=true)`. Do not pass its categorical label matrix to `pseudocolor_render`; it must not receive a continuous height colorbar.

`local_outliers_detect` always writes its mask and height overlay when that scientific branch runs. Do not call it solely to make `local_outliers_overlay`, and do not claim that its render can be disabled independently. When process visualization is requested without exact stages, use `source`, `leveled`, and `segmented_regions_overlay`; add `local_outliers_overlay` only when the outlier analysis is otherwise required.

Every review image must be a leaf attached to an already-required numeric artifact or scientific branch. Do not add segmentation, outlier detection, statistics, filtering, or another preprocessing stage solely to create a figure.

## Domain and Scope Rules

- Use plane detrending unless the user or an objective diagnostic requires another mode.
- Skip segmentation when the request needs only the `matrix` scope and no region or outlier validation. Otherwise create one region result and reuse it.
- Use `polarity="dark"` for low-height, lower, depressed, recessed, pit, pore, hole, or trench targets.
- Use `polarity="bright"` for high-height, raised, protruding, pillar, dot, island, particle, or ridge targets.
- Use `polarity="auto"` only when the task and physical context do not identify whether the target is high or low.
- Treat `label_value > 0` only as target-region membership. Never infer bright polarity from the phrase positive label.
- Default to the requested segmented-region scope. Include `matrix` or label-zero `background` only when explicitly requested.
- Map `matrix`, `regions`, `region_union`, and `background` to whole-surface, per-positive-label, all-positive-label-union, and label-zero requests, respectively.
- For `region_union`, use every finite leveled-height pixel whose segmentation label is greater than zero as one combined mask. Calculate its mean height, Sa, and Sq directly from those pooled pixels. Never obtain union Sa or Sq by taking an arithmetic, area-weighted, or other average of per-region Sa or Sq values.
- Interpret arithmetic mean deviation as Sa and RMS deviation as Sq only for a validated height-like channel.
- When region mean height is requested, report the arithmetic mean of the finite leveled heights inside each positive region label. Do not substitute the segmentation threshold, region median, or background mean.
- Run `local_outliers_detect` only when requested or required by an explicit validation rule. Do not silently exclude detected outliers from roughness; pass `outliers_data_path` only when the requested measurement definition requires that exclusion.
- Keep detected components generic local outliers unless the request and objective evidence support a domain label.

## Ranked Region Selection

- When the request asks for the largest `N` segmented regions, calculate the `regions` scope once, sort the returned region records by `area_px` in descending order and then by `label_value` in ascending order, and select the first `N` records.
- Use `area_px` as the canonical ranking value. Treat `area_physical` as supporting selection evidence when calibration is available; uniform per-pixel calibration preserves the same ordering.
- Do not invoke `matrix_roughness_statistics` again and do not call `feature_select` solely to perform this ranking.
- If fewer than `N` valid positive-label region records are available, report the available ranks and an objective limitation for each missing rank. Never substitute label-zero background for a missing positive-label region.

## Quality and Measurement Rules

A valid regional measurement requires a suitable height-like matrix, matching metadata and units, finite support, and a region result with `quality.passed=true` for every region-dependent scope. Do not invoke `local_outliers_detect` or `matrix_roughness_statistics` for region-dependent scopes when segmentation quality fails. Every requested scope must use the same leveled matrix. Per-region mean height, Sa, and Sq must also share the same label mask and retain the stable region ID. A region-union result must use that same label mask with `label > 0`, retain the contributing label values and region count as diagnostics, and calculate deviations around the pooled union mean. Preserve thresholds, pixel counts, union area, outlier summaries, and other returned statistics as supporting diagnostics.

## Calculation and Reporting Scope

Calculate every value required by segmentation, scope selection, ranking, outlier validation, and roughness measurement. Preserve labels, region counts, areas, thresholds, outlier diagnostics, and other supporting values in structured artifacts. In the final answer, report only the requested statistics for the selected scopes, ranked regions, or combined region union, with units and stable input identity; include a supporting value only when requested or needed to explain a material limitation.
