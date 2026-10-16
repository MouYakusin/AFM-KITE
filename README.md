# AFM-KITE

AFM-KITE (Knowledge Integration with Tool Execution for Atomic Force Microscopy) is a framework for reliable quantitative AFM/KPFM data analysis. It combines an LLM Agent, an AFM Analysis Skill that encodes reusable procedural knowledge, and Deterministic AFM Tools that execute established numerical operations.

## Architecture

```text
agent.py                         AFM-KITE LLM Agent entry point
afm_kite_core/                   Deterministic AFM Tools and execution core
  algorithms/                   in-memory NumPy/SciPy numerical algorithms
  data_io/                      AFM matrix, channel, and metadata readers
  operations/                   deterministic workflow orchestration
  artifacts/                    persistence, provenance, and workspace safeguards
  runtime/                      Agent scoping, Run lifecycle, and observability
  contracts.py                  shared types and artifact shapes
  serialization.py              shared JSON-safe conversion helpers
  tools/                        25 Deterministic AFM Tools split by capability
    boundary.py                 shared public tool-result contract
    registry.py                 single canonical public tool list
afm-kite/                         AFM Analysis Skill and workflow knowledge
experiments/                      Component-ablation Evaluation Runner and audits
data/                             local AFM/KPFM inputs (ignored)
outputs/                          generated analysis artifacts (ignored)
experiments_outputs/              isolated experiment runs (ignored)
```

The framework has three distinct roles:

- **LLM Agent:** interprets the request, identifies the measurement objective, selects and loads a workflow, and orchestrates execution.
- **AFM Analysis Skill:** stores reusable procedural knowledge, including workflow definitions, metric definitions, parameter rules, quality checks, and reporting requirements.
- **Deterministic AFM Tools:** execute established ingestion, preprocessing, feature-identification, and measurement operations through tested callable interfaces.

AFM-KITE refers to this integrated framework, not to a new AFM algorithm or to the Skill alone. The same deterministic execution profile is shared by the interactive entry point and the evaluation conditions.

Raw inputs, credentials, generated artifacts, and experiment batches remain local by default. The committed repository contains the executable framework, Skill, prompts, and configuration templates; it does not publish measurement data or secrets.

## LLM Agent entry point

`agent.py` owns the analysis model settings, prompt, limits, fixed Skill assignment, and interactive entry point:

```bash
python agent.py data/wafer/wafer1.txt
python agent.py data/sample-a.txt data/sample-b.txt
```

Run the entry point from the repository root and list every input file needed by the request. At least one project-local regular file is required; symlinks and paths outside the project are rejected. Before the first model-backed run, create the ignored local credential file from the committed template:

When starting `agent.py` directly from VS Code without command-line arguments, edit `DEFAULT_INPUT_PATHS` near the top of `agent.py`. Command-line paths take precedence when they are supplied.

```bash
cp api.env.example api.env
# Edit api.env and set DEEPSEEK_API_KEY.
```

The default LLM Agent uses `deepseek-v4-pro` with Thinking Mode explicitly enabled and `temperature=0`. Each Run has a recursion limit of 200, a 100-model-call limit, a 200-tool-call limit, a 1200-second wall-clock timeout, and a 180-second timeout for each deterministic tool invocation.

Each analysis invocation gets a private sandbox at `outputs/runs/<run-id>/`. Explicit inputs and the AFM Analysis Skill are copied into the same sandbox before model execution. The LLM Agent's filesystem backend and Deterministic AFM Tools are rooted at that Run—not at the project directory—so it sees paths such as `/inputs/input-01/sample.txt`, `/afm-kite/`, `/outputs`, and `/scratch`. Only the Run sandbox is placed under the permission guard; unrelated project files and directories keep their normal permissions throughout execution. Persistent analysis artifacts belong in the Agent-visible `/outputs`; `/scratch` is reserved for disposable runtime files.

The default entry point exposes the AFM Analysis Skill and Deterministic AFM Tools to the LLM Agent. Its runtime contract owns the dynamic output and scratch roots, host-managed Run records, physical-input filesystem restriction, structured-result consumption, retry and dependency policy, completion rules, the default four-heading answer format, and the fixed Skill assignment. Tool capabilities are supplied through registered callable schemas rather than a separate behavioral prompt.

The Agent cannot create subagents or use the generic `write_file`, `edit_file`, or `write_todos` tools, so applicable analysis must go through the exposed deterministic capabilities. The runtime denies built-in filesystem reads anywhere under `/inputs/**`; each physical raw AFM/KPFM file enters the dedicated workflow through exactly one `data_ingest` call. Each ingestion saves every detected channel, while downstream stages reuse only the required channel artifacts and matching metadata. When several files require a shared structure, the Agent passes their manifests to `data_group_build` with a relationship grounded by the user or loaded Skill; the safe unspecified default is `collection`. Filenames are not relationship evidence unless the Skill defines an explicit workflow-specific convention, such as the validated Ag-TiO2 before/after role mapping. For that workflow, explicit user semantics take priority over filenames. If exactly two members remain unresolved, the Skill allows one disclosed randomized role assignment; the Agent must report the mapping and state that all condition labels and the change sign are conditional on it.

Every finalized Run writes aggregate model usage to `run.json` and one record per model call to `model_trace.json`. The trace includes normalized input, output, total, reasoning, and prompt-cache token counts when the provider returns them. Usage accumulated before a failed or interrupted Run is retained.

## Deterministic AFM Tools

The public Deterministic AFM Tools are exported from `afm_kite_core.tools`. Their names remain generic: they operate on matrices, regions, profiles, manifests, and data groups. AFM/KPFM interpretation, workflow selection, and parameter reasoning live in the AFM Analysis Skill.

The active implementation uses a tool-layer plus algorithm-layer design:

```text
Agent or Python caller
  -> public tool in afm_kite_core.tools
  -> workflow in afm_kite_core.operations
  -> matrix/channel I/O in afm_kite_core.data_io
  -> in-memory numerical functions in afm_kite_core.algorithms
  -> stable status/artifacts/summary/result dictionary
```

`afm_kite_core.tools.registry` is the only maintained public tool list. The former dynamic engine loader, duplicate engine registry, and alternate import surfaces have been removed. Import public callables directly from `afm_kite_core.tools`.

For registered ordered-condition region analysis, `paired_region_catalog` creates the condition-A-frame full-field B-minus-A matrix and an `F###` per-region JSON/TSV catalog before selection. `paired_region_select` then chooses a region by ID, coordinate, or ranked change metric, and `paired_region_measure` returns to the original condition matrices for the final shared-mask measurement.

Matplotlib and XDG runtime caches default to `~/.cache/afm-kite/`, outside the guarded project workspace. Set `AFMDATA_CACHE_ROOT` to select a different cache directory.

Height-derived pseudocolor and annotation overlays share one plotting contract: every image has a value-unit colorbar and a lateral scale bar, and every plot whose base image is a height map uses the canonical bold title `Height`. Processing-stage meaning belongs in artifact roles, annotations, or captions rather than the height-map title. All scale bars use the periodic measurement-direction reference layout: the line is right-aligned to the main axes at `x=1.00`, placed below it at `y=-0.08`, and its centered label is placed at `y=-0.10` in main-axes coordinates. Ingested, leveled, filtered, and subtracted matrices write a same-stem `_metadata.json` sidecar that downstream tools discover automatically; an explicit metadata path still takes precedence. If physical units are genuinely absent, the suite defaults value and lateral units to `μm` and records the assumed lateral calibration as `1 μm/pixel`.

All Matplotlib figures use Arial, use a 6 pt title pad when a title is present, and are rendered/saved at 600 DPI with `bbox_inches="tight"`. Periodic-orientation height overlays use a 10 pt `profile-normal direction` annotation.

Core numerical dependencies are NumPy, SciPy, scikit-image, Matplotlib, and Pillow. Agent execution additionally requires Deep Agents, LangChain, the DeepSeek integration, and python-dotenv.

Python support is split by runtime boundary:

| Component | Supported Python |
|---|---|
| Deterministic `afm_kite_core` | Python 3.9+ |
| LLM Agent runtime and Evaluation Runner | Python 3.11+ |

`requirements.txt` pins the project's direct Agent and analysis runtime dependencies; pip resolves their transitive dependencies. Both Agent model builders enforce the Python 3.11+ boundary before loading their runtime dependencies.

```bash
python -m pip install -r requirements.txt
```

## Provenance and artifact traceability

Every public deterministic-tool call receives an independent 16-character `invocation_id` and `afm.provenance/v3` record. When a tool accepts an `output_dir`, its files are routed below `<output_dir>/<invocation_id>/` so that repeated calls and parameter variants cannot silently overwrite one another.

The provenance record currently contains:

- tool name and implementation package/version;
- all non-path parameters except `output_dir`, stored in deterministic key order;
- each declared input's parameter name and list index where applicable;
- requested and resolved input paths, existence and file kind;
- input size in bytes and BLAKE2b content digest.

The provenance block is returned with the public tool result and is automatically embedded in JSON artifacts written through the shared runtime. Every successful file-producing invocation also writes `invocation.json` using `afm.invocation-manifest/v3`. This manifest stores the invocation status and output directory, repeats the provenance block, and inventories every generated file with its relative/absolute path, size, and BLAKE2b digest. Binary artifacts are covered by this inventory even though they cannot embed a JSON provenance block themselves.

Non-path arguments are recorded as supplied. In particular, `generic_compute` provenance contains the submitted computation code, so callers must not place secrets in code or ordinary parameters. Strict reproducibility or evidentiary workflows can additionally record source-control revision, dependency/runtime versions, and random seeds.

## Component-ablation evaluation

`experiments/runner/agent.py` is the Evaluation Runner's LLM Agent factory. The runner evaluates the same four conditions described in the manuscript:

| Code method | Paper condition | Added capability |
|---|---|---|
| `basic_agent` | Baseline | shared generic computation only |
| `skill_agent` | Skill | AFM Analysis Skill |
| `tool_agent` | Tools | Deterministic AFM Tools |
| `skill_tool_agent` | AFM-KITE | AFM Analysis Skill + Deterministic AFM Tools |

All conditions use the same task prompt, common runtime contract, `deepseek-v4-pro` configuration, and general NumPy/SciPy computation. The repository records operational completion, execution time, tool calls, prompts, artifacts, and traces. Numerical agreement with expert references is a separate evaluation endpoint and is not inferred from execution success alone.

```bash
python -m experiments.run_experiments \
  --experiment grating --case-id grating-grating1 \
  --methods tool_agent --successful-runs 4 --dry-run
python -m experiments.run_experiments --grating --all-cases \
  --methods basic_agent skill_agent tool_agent skill_tool_agent
```

See [experiments/README.md](experiments/README.md) for case discovery, ablation conditions, output layout, and audit contracts.

### Manuscript evaluation corpus

The paper-reported corpus contains four AFM sample categories and 16 datasets: six grating samples, four wafer samples, two array samples (`hs100_mg`), and four granular samples (`ag_tio2`). The corresponding targets are periodic pitch and ridge-groove contrast, whole-map roughness and largest-defect depth, segmented-region roughness and mean height, and before/after grain potential change.

The runner also retains a perovskite cross-input correlation group for development and extended evaluation. It is outside the four-category, 16-dataset corpus reported in the manuscript and must not be included in manuscript totals without a corresponding revision of the paper.

## Skill

`afm-kite/` is a Skill resource directory, not a Python package. Its workflow registry maps domain requests to the generic public tools while keeping quantitative computation inside the deterministic core.

`SKILL.md` contains only activation, capability-aware execution, direct reference-loading rules, and shared interpretation boundaries. `references/registry.md` is a compact routing index; each workflow has one self-contained card under `references/`, and `references/tool-reference.md` is loaded only as the fallback detailed tool and error contract.

## Custom Deep Agents integration

Prefer the project-level `run_agent()` entry point: it already owns input validation and copying, the private Run sandbox, scoped tools, Skill snapshot, execution limits, invocation, and finalized records.

Custom integrations must provide equivalent boundaries:

- create a dedicated Run root and copy only declared inputs and the selected Skill into it;
- root both `FilesystemBackend` and `build_scoped_tools()` at that Run directory;
- apply `ArtifactPolicy` and `agent_workspace_guard`, restricting writes to `outputs/` and `scratch/`;
- expose `afm_kite_core.tools` as the only public AFM tool surface and preserve the structured result contract in `afm-kite/references/tool-reference.md`;
- never expose raw `get_tools()` results to a model-facing Agent without equivalent path validation.

Prompt instructions are not a security boundary. Integrations that bypass the provided entry points must enforce workspace and write restrictions through the backend or operating system.
