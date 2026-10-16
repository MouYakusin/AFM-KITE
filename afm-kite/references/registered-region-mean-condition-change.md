# `registered_region_mean_condition_change`

Use for the mean target-channel signal of the same registered grain, particle, island, or segmented region under two ordered conditions, together with the second-minus-first mean change.

## Supported Sample Types

- Paired AFM/KPFM measurements of the same sample before and after treatment, illumination, or another ordered condition change.
- Granular samples such as the repository's Ag-TiO2 group, and comparable samples with a height-like registration channel and a target-signal channel such as potential.

Use the workflow only when the target is a same-region condition-A/B mean comparison and second-minus-first change. Sample type alone is not sufficient.

## Ag-TiO2 Condition-Role Resolution

Apply this resolution only when the requested workflow is an Ag-TiO2 before/after illumination comparison with exactly two candidate physical members.

1. Resolve roles from explicit user semantics first. Accept a mapping stated in the request by stable input ID or filename. If that mapping conflicts with a filename token, follow the explicit user mapping and disclose the conflict.
2. For any unresolved role, match each filename stem case-insensitively against `^Ag-TiO2-(?P<pair_id>.+)-(?P<condition>before|after)$`. Accept the filename mapping only when it yields one `before` member and one `after` member with the same non-empty `pair_id`.
3. If exactly two members remain but neither user semantics nor the validated filename convention establishes both roles, randomly assign one member to `before` and the other to `after` exactly once. Do not rerandomize after inspecting data or results.
4. Always map condition A to `before` and condition B to `after`. Pass `data_manifest_paths=[before_manifest, after_manifest]` to `data_group_build` with `relationship="ordered_conditions"`, `labels=["before", "after"]`, and `reference_data_index=0`. Set `relationship_source="user_explicit"` when the user established the roles; otherwise use `relationship_source="workflow_required"`.
5. Define the ordered change as `after - before`.
6. Report the exact input-ID/filename-to-role mapping. For a randomized assignment, state that the roles lacked supporting semantic evidence, that the assignment was randomized, and that every condition-labeled result and the change sign are conditional on that assignment.
7. If there are not exactly two candidate physical members, do not randomly discard or pair inputs. Report that a unique comparison pair could not be formed.
8. Do not apply this resolution or its randomized fallback to unrelated condition-pair workflows.

The workflow establishes the ordered scientific relationship. User semantics, the validated filename convention, or the disclosed randomized fallback maps the two physical members to its condition roles.

When the task asks for the largest physical grain, call `paired_region_select(selection_metric="area_px", sort_order="descending", rank=1)`. Do not accept its generic `absolute_change` default for that task. Within one registered map, pixel area and calibrated physical area have the same ordering; preserve the calibrated area as supporting selection context. Report it only when the user requests it or when it is necessary to explain the selection or a material limitation.

## Workflow Card

```text
ingest A + ingest B
├── pseudocolor_render(source height or target matrices) [optional]
└── data_group_build(ordered conditions; register when needed)
    ├── pseudocolor_render(active registered height or target matrices) [optional]
    ├── active reference topography -> matrix_level
    │   ├── pseudocolor_render(leveled reference height) [optional]
    │   └── value_regions_segment(write_plots=<requested>)
    │       └── Height-defined labels F001 ... Fnn
    └── original target A + original target B + Height transform + labels
        └── paired_region_catalog
            ├── registered condition-B full field
            ├── full-field target difference B - A
            ├── per-region change catalog (JSON + TSV)
            ├── paired change atlas [optional]
            └── paired_region_select(ID, coordinate, or ranked metric)
                ├── selected change overlay [optional]
                └── shared selected mask
                    └── paired_region_measure
                        ├── same-pixel region means + B-minus-A change
                        ├── optional local-background-normalized change
                        └── paired region difference render [optional]
```

## Exact Roles and Handoffs

- Call `data_ingest` once for each physical member.
- Call `data_group_build` with grounded ordered-condition semantics, stable labels, a reference member, and `registration="translation"` when coordinate frames are not already comparable.
- Use a height-like registration and reference channel when the target signal is on another channel.
- Estimate one transform from the condition-A and condition-B height maps. Let `data_group_build` create active registered matrices for qualitative review, but retain its original channel index and returned transform for quantitative paired-region sampling. Do not independently estimate a transform from target-channel contrast.
- Level the active reference topography with `matrix_level(trend_mode="plane")`, then segment every grain with `value_regions_segment`. Never segment the target-channel difference map.
- Pass the data-group JSON and Height-derived regions JSON to `paired_region_catalog`. It inverse-samples the original condition-B target over the complete condition-A frame, writes the full-field B-minus-A matrix, and calculates a stable `F###` record for every labeled region before any region is selected.
- Use `paired_region_select` to choose a cataloged region by `feature_id`, an `(x_px, y_px)` coordinate, or a ranked metric such as `absolute_change`, `change_mean`, `area_px`, or paired coverage. Its output deliberately follows the selected-feature contract so downstream validation stays reusable.
- For a request for the largest physical grain, use `selection_metric="area_px"`, `sort_order="descending"`, and `rank=1`; never leave the generic `absolute_change` default active.
- Pass the data-group JSON and the selected-feature JSON returned by `paired_region_select` to `paired_region_measure`. Use `mask_mode="eroded"` with `erosion_px=2` for the primary result unless the feature is too small; retain the full-mask result as a sensitivity check when requested.
- Set `calculate_background_normalized=true` only when the user requests the local-background drift check; otherwise set it to `false`.
- Let `paired_region_measure` read condition A directly and inverse-map the same reference coordinates into the original condition-B target matrix. Do not use the already registered condition-B target matrix as its numeric input.
- Validate shape, finite coverage, units, registration quality, mask support, stable member order, and `result.quality.passed`.
- Report the raw condition means and paired second-minus-first mean as the primary result, with the ordered change calculated as second minus first. Report local-background normalization only as a supplementary drift check.

The final answer contains only the condition measurements and ordered change requested by the user, together with the stable input-to-condition mapping and validated units. Keep registration shifts, coverage values, mask sensitivities, catalog rankings, calibrated area, and other selection or quality diagnostics in structured artifacts unless requested or needed to explain a material validity limitation.

## Optional Pseudocolor Stages

Treat requested renders as an independently selectable set:

- `source_height_A`
- `source_height_B`
- `source_target_A`
- `source_target_B`
- `active_height_A`
- `active_height_B`
- `active_target_A`
- `active_target_B`
- `leveled_reference_height`
- `segmented_regions_overlay`
- `feature_candidates_overlay`
- `selected_feature_overlay`
- `full_field_difference_B_minus_A`
- `paired_change_atlas`
- `selected_change_overlay`
- `paired_region_difference_B_minus_A`

When no stages are requested, do not render any pseudocolor or segmentation review image. Call `pseudocolor_render` only after the corresponding numeric artifact exists. If a source path and its active path are identical because that member is the registration reference or registration is disabled, render that matrix once and let the one artifact satisfy both requested roles.

Use these rendering rules:

| Stage family | `value_label` | `color_scale_mode` | `colormap` |
|---|---|---|---|
| source, active, or leveled height | `Height` | `auto` | `afm_gold_sampled` |
| source or active target A | metadata-backed target label | `auto` | `afm_gold_sampled` |
| source or active target B | metadata-backed target label | `auto` | `afm_gold_sampled` |
| full-field difference B minus A | metadata-backed target-difference label | `symmetric_zero` | `coolwarm` |
| paired change atlas | metadata-backed target-difference label | `symmetric_zero` | `coolwarm` |
| selected change overlay | metadata-backed target-difference label | `symmetric_zero` | `coolwarm` |
| paired region difference B minus A | metadata-backed target-difference label | `symmetric_zero` | `coolwarm` |

`paired_region_catalog` always writes the registered condition-B field, the full-field difference matrix, the finite paired mask, and JSON/TSV per-region catalog because they are analysis artifacts required for whole-field discovery. Set `render_change_atlas=true` only when the corresponding qualitative atlas is requested. The atlas labels the existing Height-derived regions as `F###` and annotates their already-calculated mean change; it does not perform a second segmentation or measurement.

Generate the plain `full_field_difference_B_minus_A` pseudocolor only when requested by passing the catalog's `full_field_difference_npy_path` and matching metadata path to `pseudocolor_render` with `color_scale_mode="symmetric_zero"` and `colormap="coolwarm"`. The renderer consumes the already-required difference matrix and performs no new analysis.

Generate `selected_change_overlay` by setting `paired_region_select(render_selected_overlay=true)`. It renders the existing full-field difference, keeps all Height-derived boundaries as context, and highlights only the selected region. It does not rerun catalog calculation.

Generate `paired_region_difference_B_minus_A` only when requested by setting `paired_region_measure(write_difference_matrix=true)` and passing its returned `paired_difference_npy_path` plus `paired_difference_metadata_path` to `pseudocolor_render` with `color_scale_mode="symmetric_zero"` and `colormap="coolwarm"`. Otherwise set `write_difference_matrix=false`. This matrix contains only valid paired selected-region pixels; pixels outside the analysis mask are `NaN`. The required measurement already calculates these paired pixel differences; writing and rendering them adds no new numeric analysis stage.

Generate `segmented_regions_overlay` by setting `value_regions_segment(write_plots=true)`; otherwise set `write_plots=false`. Do not pass its categorical label matrix to `pseudocolor_render`.

Generate `feature_candidates_overlay` with `feature_select(render_candidates_overlay=true)`. Generate `selected_feature_overlay` with `feature_select(render_selected_overlay=true)`. Set both requested flags on the single required feature-selection call. The candidate image outlines and labels every segmented feature as `F###`; the selection image retains gray candidate outlines and emphasizes only the chosen feature in cyan. Neither render reruns segmentation or ranking.

For condition-A/B source heights and the active registered condition-B height, use one shared height scale taken from the existing ingest summaries. Use a second shared scale for the paired target renders. Pass those limits as `color_min` and `color_max`; do not call another statistics or matrix operation only to choose display limits. The leveled reference height retains its own scale because leveling changes its value range. Separately auto-scaled A and B renders are not quantitatively color-comparable.

When the user requests process visualization without naming exact stages, use this review bundle, deduplicating identical source and active paths:

1. `source_height_A` and `source_height_B` for the raw geometric context.
2. `active_height_B` for the registered condition-B frame.
3. `leveled_reference_height` for preprocessing review.
4. `feature_candidates_overlay` for Height segmentation review.
5. `active_target_A` and `active_target_B` with shared display limits.
6. `paired_change_atlas` for full-field discovery across every `F###` region.
7. `selected_change_overlay` for the user-selected region.
8. `paired_region_difference_B_minus_A` for the final measured in-region change.

Every render must be a leaf attached to a numeric artifact already required by the analysis. Do not add `matrix_filter`, `matrices_subtract`, another segmentation, another statistics call, or any other calculation solely to create a figure. If an optional render's source artifact is not otherwise required, omit that render.

## Registration and Coordinate Reuse

Use the condition-A coordinate frame as the canonical frame:

1. Estimate translation from the two height maps.
2. Apply the condition-B transform to its height and same-shaped channels only for active registered review artifacts. Registered edges are `NaN`, not nearest-edge duplicates.
3. Build all region labels once from the condition-A reference height.
4. Before selecting a region, let `paired_region_catalog` sample the original condition-B target over the complete condition-A frame and calculate the full-field difference plus every region summary.
5. Select one of those Height-defined labels without changing the transform or recomputing segmentation.
6. For every selected condition-A mask pixel `(y, x)`, sample the original condition-B target at `(y - shift_y, x - shift_x)` using the transform returned by `data_group_build`.
7. Keep only coordinates where condition A, condition B, and the mapped source coordinate are all valid. Calculate both condition means and the difference from exactly this paired support.

For a requested whole-grain mean, this shared two-dimensional mask is the primary workflow. A line samples only a subset of the grain and must not replace the region mean.

Use the eroded condition-A mask as the default primary estimate to reduce boundary mixing from subpixel registration. The full mask and an intersection with a separately segmented registered condition-B feature are sensitivity checks, not replacements for the primary Height-defined reference mask.

## Registration Quality Gate

Treat the estimated shift as a transform candidate, not as proof of successful registration. Before reporting paired values:

- verify compatible matrix shapes, lateral calibration, and height-channel identity;
- verify adequate finite overlap after applying the transform;
- inspect the returned registration diagnostics and reject an out-of-range or boundary-dominated shift;
- verify that the selected condition-A feature remains topographically supported at the same coordinates in registered condition B;
- inspect `paired_region_catalog` full-field finite coverage and each candidate region's quality gate before selection;
- require `paired_region_measure` finite paired coverage of at least 0.95 by default and inspect its region-size and shift-range gates;
- verify that `mean(B-A)` and `mean(B)-mean(A)` agree within floating precision.

Use requested `active_height_A`, `active_height_B`, or segmentation renders only for qualitative review of these checks. If the height alignment is ambiguous, the selected feature is not supported in registered condition B, or mask coverage is inadequate, do not report a paired region difference as reliable. Report the failed quality condition instead.

## Capability and Interpretation Rules

Use `paired_region_catalog` for whole-field discovery and comparable per-region screening. Use `paired_region_measure` for the final quantitative shared-mask result. Do not replace it with separately calculated condition means that can silently use different finite pixels.

Do not relabel a height-specific field from an unrelated statistics tool as potential or another target signal. Describe a value as potential, current, height, or another physical quantity only when matched metadata supports that interpretation and unit.

A paired region difference is valid only when the same registered mask has adequate finite support in both members. Never substitute a feature-to-substrate line contrast for the requested registered-region mean.
