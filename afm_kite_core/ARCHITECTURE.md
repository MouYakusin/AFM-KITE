# AFM-KITE Core Architecture

`afm_kite_core` is the deterministic execution layer of AFM-KITE. It separates public Deterministic AFM Tools, workflow orchestration, established scientific algorithms, data I/O, artifact infrastructure, and the LLM Agent runtime. Domain workflow selection and measurement semantics remain in the AFM Analysis Skill.

```text
Agent / Python caller
  -> tools/ public callable
  -> operations/ deterministic workflow
  -> algorithms/ in-memory numerical functions
  -> data_io/ file and metadata loading
  -> artifacts/ persistence and provenance

Agent entry point
  -> runtime/ scoping and Run lifecycle
  -> tools/ public callable
```

## Dependency rules

- `algorithms/` accepts NumPy arrays, masks, profiles, and numeric metadata. It implements established numerical operations and must not import Agent, LangChain, tool, workspace, or artifact-policy modules.
- `data_io/` parses and loads files. It must not choose scientific workflows.
- Public modules under `tools/` define stable generic names, signatures, and docstrings. They validate public arguments and call one concrete stage implementation.
- Private `operations/` modules coordinate file I/O, algorithms, and generated artifacts. The package is not a public import surface and its functions are never registered directly with an Agent. Defaults live only in public wrappers; operation entry points receive every argument explicitly, return Python mappings, and let failures propagate to the public tool boundary.
- `artifacts/` owns output policy, atomic writers, provenance, plotting setup, and workspace safeguards.
- `runtime/` owns Agent profiles, scoped tool adapters, Run workspaces, recording, message traces, and token usage.
- `contracts.py` owns shared `Literal` types and lightweight artifact shapes; `serialization.py` owns recursive JSON conversion.
- `data_io/paths.py` owns project-relative input and output resolution. The data-I/O layer does not import operations, tools, artifacts, or runtime.
- `tools/registry.py` is the only maintained model-exposed Deterministic AFM Tool list.
- `data_ingest` has a one-physical-file boundary and emits one data manifest; `data_group_build` is the separate multi-file relationship and compatibility boundary.
- Feature-local comparison is composed from `feature_select`, `feature_line_build`, `line_profile_sample`, `line_profile_measure`, and the optional `line_profile_render`; condition comparisons use this same generic staged path rather than a separate one-call wrapper.
- `tools/boundary.py` is the only public result-normalisation boundary. It also assigns a per-call invocation ID and routes generated files below `<output_dir>/<invocation_id>/`. It owns public error conversion and JSON-safe serialization; private operations do not duplicate those concerns.
- `artifacts/provenance.py` owns the common provenance contract, while `artifacts/writers.py` attaches active provenance to generated JSON files. Each successful file-producing invocation also writes `invocation.json`.
- `runtime/scoping.py` applies workspace and artifact policies only at the Agent boundary.
- `afm_kite_core.tools` is the public Deterministic AFM Tool import surface, with `tools/registry.py` as its model-exposed catalog.

## Adding a public tool

1. Add or reuse an in-memory function under `algorithms/`; it should accept arrays and return Python/NumPy values without reading or writing files.
2. Add any required parser or loader under `data_io/`.
3. Add the file/artifact workflow to an appropriate private `operations/` module.
4. Add the stable public wrapper to the corresponding public `tools/*.py` module.
5. Add the wrapper once to `tools/registry.py` and explicitly export it from `tools/__init__.py`.
6. Update the Skill workflow registry and tool reference.
7. Add algorithm unit tests, tool contract tests, and an end-to-end workflow test when the tool participates in a staged analysis.

## Public API policy

The public tool names and result fields remain stable:

```text
status
step
artifacts
summary
result         # domain values only; envelope and artifact fields are not repeated
invocation_id  # calls handled by the shared operation boundary
provenance     # tool, parameter, version, and input metadata
message  # on failure
```

Public tools are imported from `afm_kite_core.tools` and registered only in `tools/registry.py`.
