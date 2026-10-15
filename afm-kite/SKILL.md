---
name: afm-kite
description: Analyze AFM and KPFM raw exports, matrices, and multichannel data. Use for periodic topography and pitch, segmented-region height and roughness, whole-map roughness with largest-area defect depth, registered condition-pair region means and changes, and cross-input topography-signal correlations. Route requests through generic scientific workflows while keeping scan-mode, channel physics, and sample-specific meaning out of public tool names and descriptions.
---

# AFM-KITE

## Scientific Contract

- Preserve stable input identity, validated units, metadata, and provenance.
- Represent cross-file relationships, ordering, and condition roles explicitly.
- Keep preprocessing, quality control, and intermediate artifacts separate from reported measurements.
- Allow the selected workflow to calculate and preserve supporting values needed for selection, validation, and quality control without treating them as user-requested results.
- Use rendered images only for qualitative review, never as quantitative evidence.

## Reference Loading

Do not discover Skill files with `glob`, `ls`, or recursive listing. Use the exact paths below.

1. Read `references/registry.md` once to map the request's sample type, required results, and key measurement semantics, then select one complete workflow or the smallest necessary set.
2. Read only the selected workflow card:
   - `references/whole-map-roughness-largest-area-defect.md`
   - `references/segmented-region-height-roughness.md`
   - `references/periodic-topography-pitch-contrast.md`
   - `references/registered-region-mean-condition-change.md`
   - `references/cross-input-topography-signal-correlation.md`
3. Prefer one workflow whose declared results and semantics completely cover the request. Merge cards only when no single card can do so, and reuse equivalent upstream nodes.
4. Do not read `references/tool-reference.md` when active tool schemas and structured results are sufficient. Read only its relevant section after a schema ambiguity, recoverable tool error, missing result-field contract, or absent runtime execution contract.

## Domain Execution

Select a single scientific workflow when its declared results and key semantics completely cover the requested measurements. When several single workflows qualify, prefer the closest semantic match with the fewest unrequested branches. Select the smallest covering set and merge cards only when no single workflow qualifies; reuse equivalent upstream operations.

Use each workflow card's supported sample types as supporting routing evidence together with the requested measurement target. Do not select a workflow from a sample or filename alone, and do not let sample-type compatibility override incompatible result, scope, selection, relationship, ordering, or statistic semantics. When sample type is absent or ambiguous, route by the requested measurement semantics.

For each required scientific operation, use a matching dedicated deterministic tool when it is exposed. Otherwise implement the same operation through `generic_compute`, preserving the card's inputs, dependencies, measurement definitions, and quality gates.

Establish one ingest representation per physical input. When `data_ingest` is exposed, call it once for that input and reuse its channel index, metadata, and manifest. Otherwise pass the physical input directly to `generic_compute` once for ingestion and reuse the resulting representation. Report a missing capability only when neither the exposed dedicated tools nor `generic_compute` can perform a required scientific operation.

The Skill owns domain decisions that map AFM/KPFM language to scientific operations and tool arguments, including channel role, trend mode, measurement scope, direction selection, thresholds, calibration, relationship semantics, and physical interpretation.

## Domain Reporting

Workflow cards define the scientific calculation scope. Compute and preserve any intermediate values, selection quantities, supporting diagnostics, and quality-control measurements that the selected workflow requires, even when they are not requested as final results.

The user task defines the final-answer reporting scope. Report only explicitly requested measurements, their validated units, and the identity information needed to associate them with the correct input, channel, region, or condition. Keep other calculated values in structured artifacts or audit records. Include them in the final answer only when the user explicitly requests them or when they are necessary to explain a material scientific validity limitation.

Interpret a numeric field as height, potential, current, roughness, pitch, depth, or another physical quantity only when matching channel metadata and units support that interpretation. Do not add physical interpretation beyond the requested quantities unless explicitly requested.
