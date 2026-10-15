# `periodic_topography_pitch_contrast`

Use for gratings, grooves, trenches, ridges, stripes, pitch, period, ridge/groove width, duty cycle, and high-low contrast.

## Supported Sample Types

- AFM grating or pitch-calibration standards.
- Approximately one-dimensional periodic ridge-groove, line-space, stripe, or trench structures.

Use the workflow only when the requested target is a periodic statistic such as pitch, contrast, width, or duty cycle. Sample type alone is not sufficient.

## Minimal Workflow Card

```text
data_ingest
└── matrix_level(trend_mode="plane")
    └── frequency_direction_candidates(minimum=4, maximum=0)
        └── directional_series_extract(fft_centroid angle, bin=1, smoothing=0)
            └── periodic_series_statistics(percentiles=85/15, minimum=4, maximum=0)
```

Execute these five calls directly. Do not add `matrix_inspect`, preview rendering, file readback, or an axis-aligned fallback unless a returned diagnostic or explicit user request makes that branch necessary.

## Exact Calls and Field Handoff

| Stage | Required call | Required inputs and parameters | Consume |
|---|---|---|---|
| ingest | `data_ingest` | physical AFM export; one call only | selected height-like matrix, matching metadata, manifest |
| level | `matrix_level` | ingested matrix; `trend_mode="plane"`; matching metadata | detrended matrix and preserved physical metadata |
| orientation | `frequency_direction_candidates` | detrended matrix; `minimum_period_samples=4.0`; `maximum_period_samples=0.0`; matching metadata | `result.fft_centroid_angle_degrees` and quality diagnostics |
| profile | `directional_series_extract` | detrended matrix; `direction_mode="angle"`; `direction_degrees=result.fft_centroid_angle_degrees`; `bin_size_samples=1.0`; `smoothing_sigma_samples=0.0`; matching metadata | directional numeric series |
| measure | `periodic_series_statistics` | directional series; `upper_percentile=85.0`; `lower_percentile=15.0`; `minimum_period_samples=4.0`; `maximum_period_samples=0.0`; spacing and units from metadata | segment levels, widths, center spacings, frequency period, quality diagnostics |

## Critical Selection Rules

- Use `result.fft_centroid_angle_degrees` as the standard AFM profile-normal direction.
- Treat `result.angle_refinement.selected_angle_degrees` and other projection-search angles as diagnostics only. Never substitute them for the FFT-centroid angle in the standard workflow.
- Do not substitute `direction_mode="x"` or `"y"` when a valid frequency-derived direction exists.
- Define pitch as the median of available positive upper- and lower-segment center spacings. Fall back to the frequency-derived period only when neither center spacing is available.
- Define ridge-groove height contrast as the upper-segment median level minus the lower-segment median level.
- Use the physical sample spacing, coordinate unit, and height unit carried by the matching metadata.

## Quality Gates

A periodic measurement is not determined when ingestion provides no suitable height-like channel, orientation or periodicity diagnostics are inadequate, units cannot be validated, or fewer than two usable same-type centers support a segment-spacing pitch. Do not estimate quantitative values from color, visual spacing, or an arbitrary profile.

## Calculation and Reporting Scope

Calculate and preserve the selected profile-normal direction, segment levels, widths, center spacings, frequency period, and quality diagnostics needed by the workflow. In the final answer, report only the periodic measurements explicitly requested by the user, with units. Report a direction or diagnostic only when requested or when it is necessary to explain a material orientation, periodicity, edge-segment, or calibration limitation. Report widths or duty cycle only when requested.
