# `cross_input_topography_signal_correlation`

Use when each physical input contributes one height-like topography measurement and one target-channel mean, followed by a correlation across inputs.

## Supported Sample Types

- Perovskite sample series with topography and DC-bias or another target-signal channel in each input.
- Other multi-sample, concentration-series, treatment-series, or condition-series AFM/KPFM datasets with one comparable topography channel and one comparable target channel per input.

Use the workflow only when the target includes per-input topography/signal measurements followed by a cross-input correlation. Sample type alone is not sufficient.

## Contents

- Workflow Card
- Optional Pseudocolor Stages
- Domain Decisions
- Capability Mapping
- Correlation and Classification
- Eligibility and Result Rules
- Calculation and Reporting Scope

## Workflow Card

```text
for each physical input
data_ingest
├── pseudocolor_render(source_topography) [optional]
├── pseudocolor_render(source_target) [optional]
├── selected topography -> matrix_level(trend_mode="plane")
│   ├── pseudocolor_render(leveled_topography) [optional]
│   └── matrix_roughness_statistics(["matrix"])
└── original target matrix -> finite-pixel whole-map mean

paired per-input table
├── Pearson correlation: Sa versus target-channel mean
└── Pearson correlation: Sq versus target-channel mean
```

| Stage | Required call and inputs | Output or rule |
|---|---|---|
| ingest | `data_ingest` once per physical input | one reusable channel index, metadata set, and manifest |
| source topography preview | `pseudocolor_render`; selected original height-like matrix | optional per-input qualitative render |
| source target preview | `pseudocolor_render`; selected original target matrix | optional per-input qualitative render |
| level | `matrix_level`; selected topography; `trend_mode="plane"`; matching metadata | per-input leveled topography |
| leveled topography preview | `pseudocolor_render`; leveled topography | optional per-input qualitative render |
| roughness | `matrix_roughness_statistics`; leveled topography; `statistics_scopes=["matrix"]` | one whole-map Sa and Sq row contribution |
| target mean | `generic_compute`; original target matrix finite values | one whole-map target-channel mean row contribution |
| correlation | `generic_compute`; complete paired rows only | Pearson `r`, r-sign classification, and diagnostics |

The per-input table is the sole numeric input to both correlations. Every row must contain the stable input ID, Sa, Sq, target-channel mean, and validated units.

## Optional Pseudocolor Stages

Treat the requested stages as an independently selectable set for each `input-XX`:

- `source_topography_<input-XX>`
- `leveled_topography_<input-XX>`
- `source_target_<input-XX>`

When no stages are requested, do not call `pseudocolor_render`. Render only after the corresponding numeric artifact exists, using:

| Stage family | `value_label` | `plot_title` | `color_scale_mode` |
|---|---|---|---|
| source or leveled topography | `Height` | `Height` | `auto` |
| source target | metadata-backed target label | metadata-backed target title | `auto` |

When the user requests cross-input visual comparison, apply shared display limits separately to the source-topography, leveled-topography, and target families, but only across matrices with compatible physical units. Derive the limits from `value_min` and `value_max` already present in the corresponding ingest or leveling summaries, and pass them as `color_min` and `color_max`; do not run another statistics operation solely to choose display limits. When shared comparison is not requested, use independent automatic scales. When process visualization is requested without exact stages, render all three stages for every eligible input so the review set remains symmetric. This card defines no default correlation plot; generate one only when explicitly requested and an exposed deterministic capability supports it.

Every render must be a leaf attached to an already-required matrix. Do not add filtering, segmentation, another measurement, or file readback solely to create a figure.

## Domain Decisions

- Select channels from ingest metadata, not from filename text or rendered images.
- For the topography branch, use plane detrending unless an objective diagnostic requires another mode.
- Calculate Sa as the arithmetic mean absolute deviation and Sq as the root mean square deviation of the finite leveled topography values.
- Calculate the target-channel mean from all finite values in the original target matrix. Do not level, filter, segment, or background-normalize that channel unless the task explicitly requests it.
- Preserve one topography Sa, one topography Sq, and one target-channel mean per physical input. Do not pool pixels across files before calculating per-input metrics.
- Use the same ordered set of eligible input IDs for both variables in each correlation.
- Require compatible physical units across the rows used by one correlation. Apply a deterministic metadata-backed conversion when available; otherwise report the incompatibility rather than correlating incomparable values.

## Capability Mapping

When matching dedicated tools are exposed:

1. Use `data_ingest` once per physical input.
2. Use `matrix_level` on the selected topography matrix with `trend_mode="plane"`.
3. Use `matrix_roughness_statistics` with the whole-matrix scope for Sa and Sq.
4. Use `generic_compute` for the finite-pixel target-channel mean and the two Pearson correlations when no dedicated general mean or correlation callable is exposed.

When dedicated ingestion or measurement tools are unavailable, implement the same scientific workflow through `generic_compute`. Preserve one ingest result and one per-input metric row for each physical input.

## Correlation and Classification

- Require at least three complete per-input rows for a Pearson correlation.
- Treat a correlation as undefined when either variable has zero finite variance or when fewer than three complete rows remain.
- Calculate Pearson's `r` from the paired rows. Do not calculate or report a p-value.
- Classify the relationship solely from `r`: `positive` when `r > 0`, `negative` when `r < 0`, and `no clear correlation` when `r = 0`.
- Do not infer a relationship from separately ordered or incomplete lists.

## Eligibility and Result Rules

An input is eligible only when its required channels and units are validated. Exclude no input silently. Preserve a row-level eligibility record and the objective reason for every missing-channel, invalid-unit, or non-finite-row exclusion. The result contains Sa, Sq, and target-channel mean with units for every eligible input, plus both correlation coefficients, classification outcomes, paired-row counts, and missing-channel, constant-variable, incompatible-unit, or insufficient-sample diagnostics.

## Calculation and Reporting Scope

Treat paired-row counts, variance checks, and eligibility diagnostics as supporting calculations. Preserve them in the structured result, but include them in the final answer only when the user requests them or when they are necessary to explain an undefined result, a classification, or another material limitation. The final answer otherwise contains only the per-input measurements, Pearson r coefficients, and r-sign classifications requested by the task.
