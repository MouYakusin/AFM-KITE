# Internal operations

This package contains deterministic workflow implementations behind the public Deterministic AFM Tools in `afm_kite_core.tools`. It coordinates established numerical operations, data I/O, and generated artifacts, but it is not a model-facing or stable public API. The AFM Analysis Skill supplies workflow semantics and parameter rules; these modules execute the resulting stages.

| Public module | Internal implementation |
|---|---|
| `ingest.py` | `ingest/`, `inspection.py` |
| `data_group.py` | `grouping/` |
| `preprocessing.py` | `preprocessing.py` |
| `regions.py` | `segmentation/` |
| `residual.py` | `residual/` |
| `periodic.py` | `periodic/`, `roughness.py`, `profiles.py` |
| `features.py` | `features/` |
| `line_profiles.py` | `line_profiles/` |

New code should import public tools from `afm_kite_core.tools` and pure array operations from `afm_kite_core.algorithms`. No external integration should import this package directly.

Keep each implementation in the module matching its public capability. Shared array calculations belong in `algorithms/`, while shared file parsing belongs in `data_io/`. Operation entry points return dictionaries and raise failures; the public tool boundary alone converts values and errors to the public result contract.
