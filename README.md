# AFM-KITE

AFM-KITE (Knowledge Integration with Tool Execution for Atomic Force Microscopy) is a framework for reliable quantitative AFM/KPFM data analysis. It combines an LLM Agent, an AFM Analysis Skill, and Deterministic AFM Tools so that natural-language measurement requests can be translated into reproducible analysis workflows.

The three components have separate roles:

- **LLM Agent:** interprets the request, identifies the measurement objective, selects a workflow, and coordinates execution.
- **AFM Analysis Skill:** provides reusable procedural knowledge, including workflow definitions, metric definitions, parameter rules, quality checks, and reporting boundaries.
- **Deterministic AFM Tools:** execute established ingestion, preprocessing, feature-identification, and measurement operations through callable interfaces.

AFM-KITE is the integrated framework formed by these components. It is not a new AFM algorithm, and the Skill is not treated as physical ground truth. The paper evaluates the framework with a component-ablation study that compares Baseline, Skill, Tools, and AFM-KITE conditions under the same LLM Agent configuration.

## Project layout

```text
agent.py             Interactive LLM Agent entry point
afm_kite_core/       Deterministic AFM Tools and execution core
afm-kite/            AFM Analysis Skill and workflow cards
experiments/         Component-ablation evaluation runner
data/                Local AFM/KPFM inputs; ignored by Git
outputs/             Generated analysis artifacts; ignored by Git
```

The manuscript evaluation corpus covers four sample categories: grating, wafer, array, and granular samples. 

## Installation

The Agent and evaluation runner require Python 3.11 or newer. From the repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Create the local credential file from the committed template and set the model credentials before using the Agent:

```bash
cp api.env.example api.env
# Edit api.env and set DEEPSEEK_API_KEY.
```

`api.env`, raw measurement data, generated outputs, and experiment batches are excluded from version control. Do not commit credentials or private measurement data.

## Run the LLM Agent

Run from the repository root and provide one or more input files under the local project directory:

```bash
python agent.py data/wafer/wafer1.txt
python agent.py data/sample-a.txt data/sample-b.txt
```

When no command-line inputs are supplied, edit `DEFAULT_INPUT_PATHS` near the top of `agent.py`. The Agent writes run artifacts under `outputs/runs/`.

## Run the component-ablation evaluation

The evaluation runner maps its implementation methods to the paper conditions as follows:

| Runner method | Paper condition | Added component |
|---|---|---|
| `basic_agent` | Baseline | Shared generic computation |
| `skill_agent` | Skill | AFM Analysis Skill |
| `tool_agent` | Tools | Deterministic AFM Tools |
| `skill_tool_agent` | AFM-KITE | AFM Analysis Skill + Deterministic AFM Tools |

Inspect a planned draw without invoking the model:

```bash
python -m experiments.run_experiments \
  --experiment grating --case-id grating-grating1 \
  --methods basic_agent skill_agent tool_agent skill_tool_agent \
  --successful-runs 3 --dry-run
```

Run one case across all four conditions:

```bash
python -m experiments.run_experiments \
  --experiment grating --case-id grating-grating1 \
  --methods basic_agent skill_agent tool_agent skill_tool_agent \
  --successful-runs 3 --runs-per-batch 1
```

Run all cases in a registered group:

```bash
python -m experiments.run_experiments \
  --experiment grating --all-cases \
  --methods basic_agent skill_agent tool_agent skill_tool_agent \
  --successful-runs 3
```

Use `--experiment` with `grating`, `wafer`, `hs100_mg` (array), `ag_tio2` (granular), or the optional `perovskite` extension. Use `--case`, `--case-id`, or `--all-cases` to select the input scope. Detailed experiment configuration, resume behavior, and artifact layout are documented in [experiments/README.md](experiments/README.md).

## Outputs and further documentation

Interactive runs are stored under `outputs/runs/`. Evaluation batches are stored under `experiments_outputs/`. The [AFM Analysis Skill](afm-kite/SKILL.md) contains the workflow routing rules and measurement semantics used by the Agent.
