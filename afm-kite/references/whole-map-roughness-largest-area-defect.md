# `whole_map_roughness_largest_area_defect`

Use when one wafer height-map request asks for both whole-map surface roughness and the maximum depth of the largest-area dark defect.

## Supported Sample Types

- Semiconductor or material wafers.
- Other nominally continuous, approximately planar height maps containing sparse local depressions or dark defects.

Use the workflow only when the requested target is whole-map Sa/Sq together with largest-area defect depth relative to a local background. Sample type alone is not sufficient.

## Workflow Card

```text
data_ingest
├── pseudocolor_render(source) [optional]
└── matrix_level(trend_mode="plane")
    ├── pseudocolor_render(leveled) [optional]
    ├── matrix_roughness_statistics(statistics_scopes=["matrix"])
    └── matrix_filter(filter_name="gaussian", sigma=<background_sigma_px>)
        ├── pseudocolor_render(smooth_background) [optional]
        └── matrices_subtract(leveled - smooth_background)
            ├── pseudocolor_render(residual) [optional]
            └── residual_features_prepare(background_sigma_px=<background_sigma_px>)
                └── residual_features_segment(
                    polarity="dark",
                    robust_sigma_threshold=5.0,
                    min_area_px=3,
                    min_global_range_fraction=0.03
                )
                    └── residual_features_measure(
                        selection_mode="largest_area",
                        render_candidates_overlay=<requested>,
                        render_selected_overlay=<requested>
                    )
```

| Stage | Required call and inputs | Output or rule |
|---|---|---|
| ingest | `data_ingest` once for the physical input | selected height-like source matrix, matching metadata, manifest |
| source preview | `pseudocolor_render`; ingested height matrix | optional `source` render only |
| level | `matrix_level`; source matrix; `trend_mode="plane"`; matching metadata | one leveled matrix shared by roughness and defect branches |
| leveled preview | `pseudocolor_render`; leveled matrix | optional `leveled` render only |
| roughness | `matrix_roughness_statistics`; leveled matrix; `statistics_scopes=["matrix"]` | whole-map Sa and Sq; do not segment value regions |
| smooth | `matrix_filter`; leveled matrix; Gaussian `sigma=<background_sigma_px>` resolved by the AFM background-scale rule below | smooth local-background model |
| smooth preview | `pseudocolor_render`; filtered matrix | optional `smooth_background` render only |
| subtract | `matrices_subtract`; leveled minus filtered matrix | residual matrix |
| residual preview | `pseudocolor_render`; residual matrix | optional `residual` render only |
| prepare | `residual_features_prepare`; source, leveled, residual paths; reuse the exact resolved `background_sigma_px` | prepared residual-feature contract |
| segment | `residual_features_segment`; prepared contract; `polarity="dark"`; `robust_sigma_threshold=5.0`; `min_area_px=3`; `min_global_range_fraction=0.03` | all qualifying dark candidates |
| measure | `residual_features_measure`; prepared and candidate contracts; `selection_mode="largest_area"`; set the two overlay flags from the requested stages | measure only the area-ranked candidate, report its maximum depth, and optionally write either or both defect-review overlays in the same call |

## AFM Local-Background Scale

Resolve `background_sigma_px` once before filtering, then pass the same value to `matrix_filter` and `residual_features_prepare`.

- Use `background_sigma_px=12.0` by default for a 256×256 pixel, 20×20 μm wafer height map with localized submicrometre defects. Its pixel pitch is about 0.078 μm/px, so 12 px represents a Gaussian sigma of about 0.94 μm and a Gaussian FWHM of about 2.21 μm.
- Treat 12 px as a workflow default for this AFM sampling regime, not as a universal AFM constant. AFM wafer maps often combine a slowly varying substrate, localized defects, and sparse extreme spikes. A background scale larger than the target defect width keeps the defect in the residual; a scale that follows a sharp spike can create an opposite-polarity residual halo.
- When scan size or resolution differs, preserve the approximately 0.94 μm physical background scale with `background_sigma_px = max(1.0, 0.94 / mean_pixel_pitch_um)`, where `mean_pixel_pitch_um` is the mean of the metadata-derived X and Y pixel pitches. Record both the pixel and physical sigma.
- If the expected defect width is comparable to or larger than the resolved physical sigma, increase the background scale so the background varies more slowly than the target feature. If metadata is unavailable, use 12 px and report that the scale is pixel-domain only.
- Do not change sigma only to force a preferred candidate. Inspect the residual and candidate diagnostics whenever the selected feature changes materially across reasonable nearby scales.

## Optional Pseudocolor Stages

Treat the requested pseudocolor stages as an independently selectable set:

- `source`
- `leveled`
- `smooth_background`
- `residual`
- `defect_candidates_overlay`
- `largest_defect_overlay`

When no stages are requested, do not render any pseudocolor or defect overlay. For `source`, `leveled`, `smooth_background`, and `residual`, call `pseudocolor_render` only after the corresponding numeric artifact exists. Use these rendering arguments:

| Stage | `value_label` | `plot_title` | `color_scale_mode` |
|---|---|---|---|
| `source` | `Height` | `Height` | `auto` |
| `leveled` | `Height` | `Height` | `auto` |
| `smooth_background` | `Height` | `Height` | `auto` |
| `residual` | `Residual Height` | `Residual Height (Leveled - Background)` | `symmetric_zero` |

Generate `defect_candidates_overlay` with `render_candidates_overlay=true`. It shows every segmented candidate with a gray outline and stable gray `C###` label. Generate `largest_defect_overlay` with `render_selected_overlay=true`. It keeps all candidate outlines as gray context, hides every unselected label, and emphasizes only the maximum-area candidate with a cyan outline and its `C###` label. Do not pass a label matrix to `pseudocolor_render`; label matrices are categorical artifacts and must not receive a height colorbar. Set both requested flags on the single measurement call instead of repeating `residual_features_measure`. Rendering either image does not broaden the quantitative scope: calculate and report depth only for the selected candidate.

## Selection and Measurement Rules

- Define the selected defect as the qualifying residual candidate with maximum `area_px`. Break equal-area ties by the smaller `label_value`.
- Select by area before calculating or sorting depths. Do not substitute the deepest candidate for the largest-area candidate.
- Keep the full candidate union excluded from the local-reference annulus even though only one candidate is measured.
- For the selected dark defect, calculate maximum depth as local-reference median minus the minimum leveled height inside the selected component.
- Apply a mandatory dark-polarity quality gate before reporting. Require positive `max_depth`, `mean_depth`, and `median_depth`. Reject a nominally dark component when it contains a strong positive extremum inconsistent with its local reference or when maximum depth is positive but mean or median depth is non-positive; this pattern indicates an opposite-polarity halo or morphology-contaminated component.
- If the selected component fails the quality gate, do not report its maximum depth and do not silently substitute the deepest candidate. Rebuild the local background with the resolved AFM scale, rerun segmentation, and report that no valid largest-area dark defect is available if the failure persists.
- Review the candidate or selected overlay when rendering was requested or when numerical polarity diagnostics are ambiguous. Treat the rendered height image as qualitative validation only.
- Preserve the total candidate count, selected candidate ID and area, local-reference median, and selected-component minimum height as supporting selection and measurement diagnostics. Report depth only for the selected candidate unless the user explicitly requests those supporting values.
- The full-map Sa and Sq include every finite pixel in the leveled matrix, including defect pixels. Do not describe them as defect-excluded roughness.

## Stop and Reporting Rules

Stop the defect branch when no dark candidate passes segmentation and report that no largest-area defect is available. The independent whole-map roughness branch may still complete. In the final answer, report only the requested whole-map roughness and selected-defect measurements with units. Keep candidate count, selected candidate ID and area, local-reference median, and selected-component minimum height in structured artifacts unless the user requests them or they are necessary to explain a material limitation.
