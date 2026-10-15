# AFM Evaluation Runner

`run_experiments.py` evaluates a 2 x 2 capability-bundle factorial while preserving the established method names:

| Method | Shared NumPy/SciPy baseline | Skill bundle | Tool bundle |
|---|---:|---:|---:|
| `basic_agent` | 1 | 0 | 0 |
| `skill_agent` | 1 | 1 | 0 |
| `tool_agent` | 1 | 0 | 1 |
| `skill_tool_agent` | 1 | 1 | 1 |

`basic_agent` is therefore a generic scientific-compute baseline, not a no-capability control. Every method receives the same restricted `generic_compute` StructuredTool, which exposes NumPy and SciPy in an isolated worker without shell access or persistent analysis scripts. Conventional imports of `json`, `math`, `numpy` (including `fft`, `linalg`, and `polynomial`), and the selected SciPy modules `fft`, `interpolate`, `linalg`, `ndimage`, `optimize`, `signal`, `spatial`, and `stats` resolve to preloaded safe modules. Restricted `open` may read only declared inputs and may write only below the policy-routed output directory. The worker cannot import the repository's AFM implementations.

### Generic-compute safety limits

The public interface and host-side subprocess controller live in `runner/generic_compute`. The controller launches `runner/generic_compute/worker.py` as an isolated Python process. The worker provides restricted numerical execution.

`generic_compute` keeps its public call shape unchanged, but the worker enforces a bounded execution envelope in addition to the configured parent-process timeout:

- at most 32 declared input files, with a 512 MiB limit per input;
- best-effort POSIX CPU, address-space, open-file, core-dump, and output-file-size limits (the applied/skipped limit names are recorded in execution metadata);
- at most 64 output files, 256 MiB per file, and 512 MiB written in one invocation;
- 20,000 retained stdout characters, with truncation and omitted-character counts;
- bounded result depth, collection size, total item count, string length, and final JSON response size, including explicit recursive-reference rejection;
- `np.load` restricted to declared numeric `.npy` arrays, with pickle and memory mapping disabled; reflective, raw-file, ctypes, arbitrary-import, and shell-oriented access remains forbidden.

These controls reduce accidental resource exhaustion and obvious escape paths, but the Python AST/builtin restrictions are not a complete hostile-code security boundary. Production use with mutually untrusted submitters must additionally place the worker in an OS/container sandbox with a read-only filesystem, a dedicated writable output mount, no network, a non-root user, syscall filtering, and cgroup or equivalent CPU/memory/PID limits.

The Skill bundle contains the copied `afm-kite` skill, its references, its minimal assignment prompt, and the associated context-loading overhead. The Tool bundle contains only the 22 dedicated deterministic AFM tools and their necessary callable schemas. Bundle-level results must not be attributed to a single file, prompt, or callable.

Workspace scoping, artifact classification, standard error results, result compaction, provenance, audit traces, model settings, and resource limits are shared infrastructure. The two Tool-bundle methods add the dedicated AFM tool catalog to the shared `generic_compute` tool.

Every cell uses `deepseek-v4-pro` with Thinking Mode explicitly disabled and `temperature=0.0`. Repeated Runs remain necessary to measure residual service-side and agent-workflow variation.

The capability-ceiling study uses model-call, tool-call, and recursion guards rather than an equal-cost budget. Total tokens are recorded but never capped. Tool-call count and token use are descriptive diagnostics only and do not affect answer quality or Run acceptance.

## Prompt composition

Experiment variants, factor definitions, and prompt composition are defined in `runner/protocol.py`. Editable prompt content remains in `prompts.json`.

Experiment prompts live in the `system_prompts` namespace of `prompts.json`:

| Method | Prompt sections |
|---|---|
| `basic_agent` | `common` |
| `skill_agent` | `common + skill_bundle` |
| `tool_agent` | `common` |
| `skill_tool_agent` | `common + skill_bundle` |

The Common prompt exclusively defines every method's evidence boundary, workspace policy, physical-input filesystem restriction, shared `generic_compute` baseline, structured-result handling, retry policy, completion rule, and reporting contract. It does not name a treatment-specific AFM workflow or dedicated callable. The Skill-bundle prompt only assigns `/afm-kite/SKILL.md` and limits the Skill to AFM/KPFM workflow selection, measurement semantics and domain parameters, workflow-specific channel and condition-role resolution, metadata-backed physical interpretation, and scientific validity gates. It also makes explicit that the Skill may broaden the scientific calculation scope with necessary supporting values and quality checks, but may not broaden the final-answer scope. It does not restate the Common prompt's runtime policies. Task prompts own case inputs, required metrics, the high-level comparison target, dynamic artifact routing, and the authoritative final-answer reporting scope. For Skill methods, workflow-specific condition-role resolution follows the Skill's priority order: explicit user/task semantics, then a validated filename convention, then a one-time disclosed randomized fallback for an otherwise unresolved two-member pair. The runner validates the repository's known case membership and uses deterministic input ordering before model execution; non-Skill methods receive no additional role-resolution instruction. The Tool bundle adds only the dedicated deterministic AFM callables and their necessary schemas; it adds no Tool-specific system prompt. Consequently, `basic_agent` and `tool_agent` receive byte-identical system prompts, as do `skill_agent` and `skill_tool_agent`. Tool availability, rather than additional behavioral instructions, is the Tool treatment.

The Skill expresses scientific operations independently of runtime capability. For each operation it uses a matching dedicated callable when exposed and otherwise preserves the same scientific definition through `generic_compute`. Workflow cards may calculate and preserve intermediate values, selection quantities, supporting diagnostics, and quality-control measurements required to produce or validate the requested results. Those supporting values remain in structured artifacts or audit records unless the task requests them or they are needed to explain a material objective limitation. Tool schemas and `tool-reference.md` describe callable names, arguments, constraints, structured results, and error payload fields; they do not repeat the Common prompt's retry, dependency, artifact-routing, or final-answer policies. Workflow selection, parameter choice, and physical interpretation remain in the Skill.

Every method follows the same content-only final-answer contract: report only the requested results, the identity information and units needed to interpret them, the key processing steps needed to understand them, and material objective limitations. No fixed headings, table, JSON schema, or exact metric-label wording is required. Final answers need not reproduce invocation logs, runtime metadata, artifact paths, unrequested metrics, or supporting diagnostics; those remain available in structured artifacts and the Run audit.

## Data cases

Tasks and required reporting items are defined in the `task_prompts` namespace of `prompts.json`.

- `grating`: one physical file per case.
- `hs100_mg`: one physical file per case; segment all low-height regions, combine every positive-label region into one union mask, and report one Sa and one Sq calculated directly from all finite leveled-height pixels in that union.
- `perovskite`: files named `perovskite<group>-<item>.txt`; the group number defines the case, item numbers must start at 1 and be consecutive, and at least three files are required for Pearson correlation.
- `wafer`: one physical file per case.
- `ag_tio2`: one before/after pair per case. The runner accepts only `Ag-TiO2-<case>-<before|after>.txt`, requires one member of each role, and orders them as before then after for stable case construction. Skill methods still resolve and disclose the condition-role mapping according to the Skill rather than treating runner order alone as scientific evidence.

### Current live perovskite cases

The perovskite data were regrouped on 2026-08-19 so that the first numeric component denotes test position and the second denotes sample number. The current live roster is:

| Case ID | Ordered inputs | Meaning |
|---|---|---|
| `perovskite-group-1` | `perovskite1-1.txt`, `perovskite1-2.txt`, `perovskite1-3.txt` | test position 1 across samples 1–3 |
| `perovskite-group-2` | `perovskite2-1.txt`, `perovskite2-2.txt`, `perovskite2-3.txt` | test position 2 across samples 1–3 |

The current live corpus therefore contains 18 cases in total: grating 6, HS100-MG 2, perovskite 2, wafer 4, and Ag-TiO2 4. See [`data/perovskite/README.md`](../data/perovskite/README.md) for the old-to-new filename mapping and measured DCbias means.

Existing `experiments_outputs/perovskite-*` directories are immutable historical batches whose snapshots preserve the pre-regrouping inputs. Do not resume their experiment-series IDs with the live regrouped files. Start a new series ID; compatibility checks are expected to reject an old series because case membership and input digests changed.

Before model execution, the runner freezes every selected input and, when needed, the AFM-KITE Skill into one batch-level snapshot. The snapshot manifest records input sizes and BLAKE2b digests, plus an input-only combined digest. Skill resources record only their paths and sizes: the runner neither generates nor stores Skill content hashes. Every Run verifies input contents and the Skill file inventory and sizes before copying resources into its workspace. Skill methods receive the copied Skill snapshot under `/afm-kite/`, which is the assigned treatment.

## Running experiments

The runner separates the scientific target from the work done by one process:

- a **cell** is one `case_id x method` combination;
- a **random draw** selects one method for the current case from its eligible method pool;
- an **attempt** is the actual model Run produced by that draw;
- a **batch** is one command invocation and contains at most `RUNS_PER_BATCH` draws.

A method is eligible while its cell has fewer than the target number of successful Runs and has not reached its per-cell attempt limit. After every attempt, the pool is recalculated: a cell that reaches its success target leaves the pool, while a failed cell remains eligible until its attempt limit is reached.

The editable configuration section at the top of `run_experiments.py` provides defaults for direct IDE execution:

| Configuration | CLI override | Meaning |
|---|---|---|
| `EXPERIMENT_GROUP` | `--experiment` or a group shortcut | Registered data group |
| `EXPERIMENT_CASE` | `--case`, `--case-id`, `--all-cases` | Selected case scope |
| `EXPERIMENT_METHODS` | `--methods` | Method pool used by the random sampler |
| `TARGET_SUCCESSFUL_RUNS` | `--successful-runs` | Required successes per case-method cell; default `3` |
| `MAX_ATTEMPTS_PER_CELL` | `--max-attempts-per-cell` | Attempt safety limit per case-method cell |
| `RANDOM_SEED` | `--random-seed` | Reproducible randomization seed |
| `RUNS_PER_BATCH` | `--runs-per-batch` | Maximum draws in one invocation; default `1` |
| `EXPERIMENT_SERIES_ID` | `--experiment-series-id` | Cross-batch series identifier; empty derives a stable ID |
| `AUTO_RESUME` | `--no-resume` | Load compatible earlier batches by default |
| `DRY_RUN` | `--dry-run` | Print the plan without invoking the model |

CLI values take precedence over the configuration constants. `EXPERIMENT_METHODS` is an explicit tuple, so registered methods can be selected, removed, or reordered directly in the configuration section.

```bash
# Inspect the next random draw without invoking the model.
python -m experiments.run_experiments \
  --experiment grating --case-id grating-grating1 \
  --methods basic_agent skill_agent tool_agent skill_tool_agent \
  --successful-runs 3 --random-seed 20260724 --dry-run

# Group shortcuts and 1-based case positions remain available.
python -m experiments.run_experiments --grating --case 1 --dry-run

# Run exactly one random draw for one case, then stop.
python -m experiments.run_experiments --grating \
  --case-id grating-grating1 \
  --experiment-series-id grating-random-v1 \
  --successful-runs 3 --max-attempts-per-cell 12 \
  --runs-per-batch 1

# Run the next draw later. Resume is automatic for the same experiment series ID.
python -m experiments.run_experiments --grating \
  --case-id grating-grating1 \
  --experiment-series-id grating-random-v1 \
  --successful-runs 3 --max-attempts-per-cell 12 \
  --runs-per-batch 1

# Run one case x one method, one attempt per invocation, to five successes.
python -m experiments.run_experiments --grating \
  --case-id grating-grating1 \
  --experiment-series-id grating-tool-only-v1 \
  --methods tool_agent --successful-runs 5 \
  --max-attempts-per-cell 12 --runs-per-batch 1

# Allow up to eight adaptive draws in one process when a larger batch is useful.
python -m experiments.run_experiments --grating --all-cases \
  --experiment-series-id grating-random-v1 \
  --methods basic_agent skill_agent tool_agent skill_tool_agent \
  --successful-runs 3 --max-attempts-per-cell 12 \
  --runs-per-batch 8
```

`--experiment` accepts every group registered in `runner.protocol.GROUPS`. `--methods` accepts any one or more methods registered in `AGENT_VARIANTS`. Use `--case N`, `--case-id ID`, or `--all-cases` to select data. `--successful-runs N` sets the target for every selected cell. `--max-attempts-per-cell K` applies independently to every cell and must be at least `N`. The maximum number of Runs available to the complete study is:

```text
selected cases x selected methods x max attempts per cell
```

`--runs-per-batch B` limits only the current process. It does not change the study target or the attempt allowance. With its default value of `1`, every invocation executes at most one case-method attempt and can be stopped or resumed between Runs.

A quality Run is successful only when Agent execution finishes without an error, the artifact and dependency-isolation audits pass, and the final answer is non-empty. Scientific correctness is deliberately not used as a retry condition. Failed attempts remain in `runs.jsonl`. A separate method-explicit `results_for_table.jsonl` contains one compact, table-oriented row for every attempt, including failed attempts and the `quality_run_success` flag.

### Randomization rule

Cases are handled in their discovered order. For the first case that still has eligible methods, the runner samples one method with equal probability from that case's current pool. It then executes the selected attempt and updates the pool before the next draw. Thus method order is random, but every selected cell remains targeted until it either reaches the success target or exhausts its attempt allowance.

Selection is reproducible rather than dependent on process-global random state. The draw uses the experiment series ID, `RANDOM_SEED`, case ID, monotonically increasing draw index, and ordered eligible-method list. Every run record stores the complete pre-draw pool, selected method, seed, draw index, and selection probability. Keeping the same configuration and prior records therefore reconstructs the same next draw.

A single-method pool is a supported special case: its selection probability is `1`, and repeated batches continue that cell until the requested number of successful Runs is reached.

### Split-batch resume

Resume is enabled by default. The runner scans earlier `config.json` and `runs.jsonl` files with the same experiment series ID, then keys progress by:

```text
experiment_series_id + case_id + method
```

When `EXPERIMENT_SERIES_ID`/`--experiment-series-id` is empty, a stable ID is derived from the factor schema, group, selected cases, ordered method pool, success target, per-cell attempt limit, and random seed. Repeating the same command therefore continues the same experiment series without requiring a manually chosen ID. Set an explicit ID when a human-readable name is useful. Use a new ID for an independent repeat, and use `--no-resume` only when earlier records with that ID must intentionally be ignored. If `--experiment-series-id` is omitted and automatic resume remains enabled, the generated ID is stable. If `--no-resume` is used without an explicit ID, the runner adds the batch name to make the generated ID unique.

Successful cells are removed from later candidate pools. Failed cells continue with the next per-cell attempt number until `--max-attempts-per-cell` is reached. If all selected cells are already complete, the command exits successfully without creating another batch. If no eligible method remains but at least one cell is incomplete, it exits with status `2`.

Before prior Runs are reused, the runner requires matching group, factor schema, prompt contract, artifact policy, success target, per-cell attempt limit, random seed, ordered method pool, model configuration, execution limits, system prompts, and any overlapping case definitions. A mismatch stops the resume instead of silently combining incompatible Runs. Different cases may contribute separate compatible batches to the same explicit experiment series ID.

## Run artifacts

Each batch is written under `experiments_outputs/<group>-<sequence>/`:

- `config.json`: experiment series ID, randomization rule and next draw, resume discovery, factor assignments, prompt composition, cases, target successes, per-cell attempt limit, seed, batch size, and batch-snapshot identity.
- `batch_snapshot/`: copies of all selected inputs and the optional Skill, plus `manifest.json` with input content digests and a path/size-only Skill inventory.
- `runs.jsonl`: one complete run record per invocation.
- `summary.json`: current random draws, cell completion, audit compliance, calls, tokens, and per-method operational and timing summaries. Timing reports count, mean, median, population standard deviation, minimum, and maximum separately for all attempts, quality successes, non-timeout quality failures, and timeouts; legacy records without timing are counted as missing rather than treated as zero.
- `series_state.json`: prior compatible batches, next draw index, current batch draws, and every selected cell's completion state.
- `results_for_table.jsonl`: method, factor, case, input, random-selection audit, successful repetition, attempt, UTC start/end timestamps, elapsed seconds, timeout classification, answer, error, and Run-path fields for direct post-hoc LLM organization.

Each invocation has an isolated workspace:

```text
runs/001-skill_tool_agent-<case>-success-01-attempt-01/
├── inputs/
├── afm-kite/      # Skill-bundle methods only
├── outputs/
│   ├── by_input/input-XX/
│   └── cross_input/
├── scratch/
├── answer.md
├── audit_input_inventory.json
├── audit_index.md
├── tool_trace.json
├── model_trace.json
├── run.json
└── manifest.json
```

The run record uses the current schema only:

- `factor_assignment`: `skill_bundle` and `tool_bundle` booleans.
- `effective_capabilities`: shared generic compute plus enabled bundles.
- `loaded_skill_paths`: active Skill paths.
- `tool_call_counts`: `generic_compute`, `dedicated_afm`, and `filesystem_or_other`.
- `random_selection`: seed, draw index, eligible method pool before selection, selected method, selection probability, target successful repetition, and per-cell attempt.
- `limits`: model/tool-call guards, recursion guard, and an explicit null total-token limit.
- `started_at_utc`, `finished_at_utc`, and `elapsed_seconds`: the end-to-end runner-attempt timing, measured with a monotonic high-resolution clock and accompanied by auditable UTC timestamps.
- `timed_out`: whether the attempt ended through a `TimeoutError` subclass; timeout durations remain separate from other quality failures in method summaries.

## External result evaluation

This repository stops after generating and auditing the Run records. Use `results_for_table.jsonl` as the handoff artifact for any downstream evaluation by a separate model or review workflow. The repository does not define, calculate, load, or analyze an answer-quality score.

When Runs are split across batches, keep the same derived or explicit `experiment_series_id` and compatible series-level configuration. Use a new experiment series ID for an independent single-method or method-subset experiment.
