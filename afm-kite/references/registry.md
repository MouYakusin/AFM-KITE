# Workflow Registry

Use this compact index only to select workflow cards. Do not load every card.

## Routing Contract

Translate the request into:

- sample type when explicitly stated or objectively supported by the request;
- required results;
- measurement scope;
- selection semantics;
- relationship and ordering semantics;
- requested statistic.

Use sample type and requested measurement semantics together. Treat supported
sample types as routing evidence, not as an exclusive whitelist. Never route
from a sample or filename alone, and never let a sample-type match override
incompatible result, scope, selection, relationship, ordering, or statistic
semantics. When sample type is absent or ambiguous, route by the remaining
measurement semantics.

Apply this order:

1. Select one workflow when its supported sample types are compatible, its declared results cover every required result, and its key semantics satisfy the request.
2. When multiple single workflows qualify, prefer the closest joint match of sample type, scope, selection, relationship, ordering, and statistic semantics; then prefer fewer unrequested branches.
3. Only when no single workflow qualifies, select the smallest set whose union covers every required result and semantic constraint.
4. Merge equivalent upstream nodes across selected cards. Do not select another card for a result already covered with the required semantics.

This gives the routing priority:

```text
single complete semantic match
> single broader match
> smallest multi-card covering set
```

## Workflow Capabilities

| Supported sample types (examples) | Required results | Key scope and selection semantics | Workflow ID | Read this card |
|---|---|---|---|---|
| wafers and other nominally continuous planar surfaces with sparse local depressions | whole-map Sa/Sq together with largest-area defect depth | whole height map; defect selected by maximum area before depth measurement; depth relative to local background | `whole_map_roughness_largest_area_defect` | `references/whole-map-roughness-largest-area-defect.md` |
| HS100-MG and other height maps containing value-separable low/high regions, steps, islands, particles, pillars, pits, or pores | mean height, Sa, Sq, or outlier assessment for requested matrix, segmented-region, combined segmented-region union, ranked-region subset, or background scopes | value-defined labels with bright, dark, or auto polarity resolved from target-height semantics; optional deterministic area ranking; explicit requested statistics scope | `segmented_region_height_roughness` | `references/segmented-region-height-roughness.md` |
| grating standards and other approximately one-dimensional periodic ridge-groove, line-space, stripe, or trench structures | pitch, ridge-groove contrast, width, or duty cycle | periodic topography; frequency-derived profile-normal direction; requested periodic statistic | `periodic_topography_pitch_contrast` | `references/periodic-topography-pitch-contrast.md` |
| paired before/after or condition-A/B AFM/KPFM samples, including Ag-TiO2 grains or islands with comparable height and target-signal channels | condition-A/B target means over the same region and second-minus-first change | registered shared two-dimensional mask; ordered conditions; requested ID, coordinate, or ranked selection | `registered_region_mean_condition_change` | `references/registered-region-mean-condition-change.md` |
| multi-sample or condition-series AFM/KPFM datasets, including perovskites, with one topography and one target-signal channel per input | per-input whole-map topography Sa/Sq and target-channel mean, followed by cross-input correlations | one complete row per physical input; matched input set; Pearson correlation and requested classification | `cross_input_topography_signal_correlation` | `references/cross-input-topography-signal-correlation.md` |

## Registered Callable Tools

- `data_ingest`
- `data_group_build`
- `matrix_inspect`
- `matrix_level`
- `matrix_filter`
- `pseudocolor_render`
- `matrices_subtract`
- `value_regions_segment`
- `gradient_regions_segment`
- `local_outliers_detect`
- `residual_features_prepare`
- `residual_features_segment`
- `residual_features_measure`
- `matrix_roughness_statistics`
- `frequency_direction_candidates`
- `directional_series_extract`
- `periodic_series_statistics`
- `feature_select`
- `paired_region_catalog`
- `paired_region_select`
- `paired_region_measure`

## Data-Flow Rules

1. Start every dedicated-tool workflow at `data_ingest`; call it exactly once per physical raw input and reuse its manifest, selected matrices, and matching metadata.
2. Merge equivalent upstream nodes when multiple cards are required. Never re-ingest a physical file while composing workflows.
3. Preserve earlier artifacts when a later stage requires multiple upstream inputs.
4. Never pass preview images as numeric matrices.
5. Let explicit user requirements override card defaults when the data and active tools support them.
6. Treat every workflow card as a calculation contract, not as authority to broaden the final answer. Preserve card-required intermediate values, selection quantities, diagnostics, and quality checks in structured artifacts; report only measurements requested by the user, plus identity information and material validity limitations.

## Tool Contract Fallback

Read only the relevant section of `references/tool-reference.md` when a callable, parameter, result field, or error payload remains unclear. When a required dedicated callable is unavailable but `generic_compute` is exposed, preserve the selected card's scientific objectives and dependencies through the capability-aware fallback in `SKILL.md`.
