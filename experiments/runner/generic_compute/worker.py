"""Isolated NumPy/SciPy worker for the shared experiment baseline.

The generic-compute controller launches this module in a separate Python process.
The worker exposes data libraries and a small builtin set.
It does not expose a general shell or persistent script-writing capability.
"""

from __future__ import annotations

import ast
import builtins
import contextlib
import inspect
import json
import math
import sys
from pathlib import Path
from typing import Any

try:  # POSIX-only, so every limit below remains best-effort.
    import resource
except ImportError:  # pragma: no cover - Windows deployment
    resource = None

import numpy as np
import scipy

MAX_CODE_CHARACTERS = 50_000
MAX_REQUEST_CHARACTERS = 200_000
MAX_INPUT_FILES = 32
MAX_INPUT_FILE_BYTES = 512 * 1024 * 1024
MAX_ARRAY_ELEMENTS = 50_000_000
MAX_INLINE_ITEMS = 32
MAX_RESULT_DEPTH = 12
MAX_RESULT_CONTAINER_ITEMS = 256
MAX_RESULT_TOTAL_ITEMS = 4_096
MAX_RESULT_STRING_CHARACTERS = 20_000
MAX_STDOUT_CHARACTERS = 20_000
MAX_RESPONSE_BYTES = 1 * 1024 * 1024
MAX_OUTPUT_FILES = 64
MAX_OUTPUT_FILE_BYTES = 256 * 1024 * 1024
MAX_TOTAL_OUTPUT_BYTES = 512 * 1024 * 1024
MAX_MEMORY_BYTES = 4 * 1024 * 1024 * 1024
MAX_CPU_SECONDS = 175
FORBIDDEN_NAMES = {
    "__import__",
    "breakpoint",
    "compile",
    "delattr",
    "dir",
    "eval",
    "exec",
    "getattr",
    "globals",
    "help",
    "input",
    "locals",
    "setattr",
    "vars",
}
FORBIDDEN_ATTRIBUTES = {
    "DataSource",
    "ctypeslib",
    "datasets",
    "distutils",
    "dump",
    "dumps",
    "f2py",
    "fromfile",
    "io",
    "lib",
    "load_npz",
    "memmap",
    "save_npz",
    "testing",
    "tofile",
}
FORBIDDEN_NODES = (
    ast.AsyncFor,
    ast.AsyncFunctionDef,
    ast.AsyncWith,
    ast.Await,
    ast.Global,
    ast.Nonlocal,
)
SAFE_IMPORT_MODULES = {
    "json",
    "math",
    "numpy",
    "numpy.fft",
    "numpy.linalg",
    "numpy.polynomial",
    "scipy",
    "scipy.fft",
    "scipy.interpolate",
    "scipy.linalg",
    "scipy.ndimage",
    "scipy.optimize",
    "scipy.signal",
    "scipy.spatial",
    "scipy.stats",
}
SAFE_IMPORT_ROOT_BINDINGS = {
    "json": "json",
    "math": "math",
    "numpy": "np",
    "scipy": "scipy",
}


class UnsafeCodeError(ValueError):
    """Raised when submitted generic-compute code exceeds the safe subset."""


class UnsafeResultError(ValueError):
    """Raised when a result cannot be represented inside the response limits."""


def _safe_module_expression(module: str) -> ast.expr:
    parts = module.split(".")
    root_binding = SAFE_IMPORT_ROOT_BINDINGS.get(parts[0])
    if root_binding is None:
        raise UnsafeCodeError(
            f"Import {module!r} is unavailable in generic_compute"
        )
    expression: ast.expr = ast.Name(id=root_binding, ctx=ast.Load())
    for part in parts[1:]:
        expression = ast.Attribute(value=expression, attr=part, ctx=ast.Load())
    return expression


def _validate_import_module(module: str) -> None:
    if module not in SAFE_IMPORT_MODULES:
        allowed = ", ".join(sorted(SAFE_IMPORT_MODULES))
        raise UnsafeCodeError(
            f"Import {module!r} is unavailable in generic_compute. "
            f"Allowed scientific imports: {allowed}"
        )


class _SafeImportRewriter(ast.NodeTransformer):
    """Translate allowlisted imports into bindings to preloaded safe modules."""

    def visit_Import(self, node: ast.Import) -> list[ast.stmt]:
        assignments: list[ast.stmt] = []
        for alias in node.names:
            _validate_import_module(alias.name)
            if alias.asname:
                target = alias.asname
                source_module = alias.name
            else:
                target = alias.name.split(".")[0]
                source_module = alias.name.split(".")[0]
            assignment = ast.Assign(
                targets=[ast.Name(id=target, ctx=ast.Store())],
                value=_safe_module_expression(source_module),
            )
            assignments.append(ast.copy_location(assignment, node))
        return assignments

    def visit_ImportFrom(self, node: ast.ImportFrom) -> list[ast.stmt]:
        if node.level or not node.module:
            raise UnsafeCodeError("Relative imports are unavailable in generic_compute")
        _validate_import_module(node.module)
        assignments: list[ast.stmt] = []
        module_expression = _safe_module_expression(node.module)
        for alias in node.names:
            if alias.name == "*":
                raise UnsafeCodeError("Wildcard imports are unavailable in generic_compute")
            if node.module == "scipy":
                _validate_import_module(f"scipy.{alias.name}")
            assignment = ast.Assign(
                targets=[ast.Name(id=alias.asname or alias.name, ctx=ast.Store())],
                value=ast.Attribute(
                    value=module_expression,
                    attr=alias.name,
                    ctx=ast.Load(),
                ),
            )
            assignments.append(ast.copy_location(assignment, node))
        return assignments


class _BoundedTextWriter:
    """Small stdout sink that never retains more than the configured limit."""

    def __init__(self, limit: int) -> None:
        self._limit = limit
        self._chunks: list[str] = []
        self._stored = 0
        self.omitted_characters = 0

    def write(self, value: Any) -> int:
        text = str(value)
        available = max(0, self._limit - self._stored)
        if available:
            retained = text[:available]
            self._chunks.append(retained)
            self._stored += len(retained)
        self.omitted_characters += max(0, len(text) - available)
        return len(text)

    def flush(self) -> None:
        return None

    def getvalue(self) -> str:
        return "".join(self._chunks)

    @property
    def truncated(self) -> bool:
        return self.omitted_characters > 0


class _ResultState:
    def __init__(self) -> None:
        self.total_items = 0
        self.active_ids: set[int] = set()

    def consume(self, count: int) -> None:
        self.total_items += count
        if self.total_items > MAX_RESULT_TOTAL_ITEMS:
            raise UnsafeResultError(
                f"result exceeds the {MAX_RESULT_TOTAL_ITEMS}-item total limit"
            )


class _OutputQuota:
    def __init__(self) -> None:
        self.bytes_written = 0

    def consume(self, count: int) -> None:
        self.bytes_written += max(0, count)
        if self.bytes_written > MAX_TOTAL_OUTPUT_BYTES:
            raise OSError(
                f"Total output exceeds the {MAX_TOTAL_OUTPUT_BYTES}-byte limit"
            )


class _QuotaFile:
    """Delegate a file object while accounting for writes in this invocation."""

    def __init__(self, wrapped: Any, quota: _OutputQuota) -> None:
        self._wrapped = wrapped
        self._quota = quota

    def __enter__(self) -> "_QuotaFile":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self._wrapped.close()

    def __iter__(self):
        return iter(self._wrapped)

    def __next__(self):
        return next(self._wrapped)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped, name)

    def write(self, value: Any) -> Any:
        if isinstance(value, bytes):
            size = len(value)
        else:
            encoding = getattr(self._wrapped, "encoding", None) or "utf-8"
            size = len(str(value).encode(encoding, errors="replace"))
        self._quota.consume(size)
        return self._wrapped.write(value)

    def writelines(self, values: Any) -> None:
        for value in values:
            self.write(value)


def _validate_code(code: str) -> ast.Module:
    tree = ast.parse(code, mode="exec")
    tree = _SafeImportRewriter().visit(tree)
    ast.fix_missing_locations(tree)
    for node in ast.walk(tree):
        if isinstance(node, FORBIDDEN_NODES):
            raise UnsafeCodeError(f"Unsupported syntax: {type(node).__name__}")
        if isinstance(node, ast.Name):
            if node.id in FORBIDDEN_NAMES or node.id.startswith("__"):
                raise UnsafeCodeError(f"Forbidden name: {node.id}")
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("_") or node.attr in FORBIDDEN_ATTRIBUTES:
                raise UnsafeCodeError(f"Forbidden attribute: {node.attr}")
        if isinstance(node, (ast.FunctionDef, ast.Lambda)):
            decorators = getattr(node, "decorator_list", [])
            if decorators:
                raise UnsafeCodeError("Decorators are not supported")
    return tree


def _apply_resource_limits() -> dict[str, Any]:
    """Apply process-local OS limits without making unsupported limits fatal."""
    if resource is None:
        return {"available": False, "applied": [], "skipped": ["resource module"]}

    applied: list[str] = []
    skipped: list[str] = []

    def set_limit(name: str, value: int) -> None:
        identifier = getattr(resource, name, None)
        if identifier is None:
            skipped.append(name)
            return
        try:
            _, hard = resource.getrlimit(identifier)
            infinity = getattr(resource, "RLIM_INFINITY", -1)
            target = value if hard in (-1, infinity) else min(value, hard)
            resource.setrlimit(identifier, (target, hard))
            applied.append(name)
        except (OSError, ValueError) as exc:
            skipped.append(f"{name}: {type(exc).__name__}")

    set_limit("RLIMIT_CPU", MAX_CPU_SECONDS)
    set_limit("RLIMIT_FSIZE", MAX_OUTPUT_FILE_BYTES)
    set_limit("RLIMIT_NOFILE", 64)
    set_limit("RLIMIT_CORE", 0)
    set_limit("RLIMIT_AS", MAX_MEMORY_BYTES)
    set_limit("RLIMIT_DATA", MAX_MEMORY_BYTES)
    return {"available": True, "applied": applied, "skipped": skipped}


def _is_within(path: Path, root: Path) -> bool:
    path = path.resolve()
    root = root.resolve()
    return path == root or root in path.parents


def _readable_path(file: Any, readable_files: set[Path], output_dir: Path) -> Path:
    if not isinstance(file, (str, Path)):
        raise TypeError("File paths must be strings or pathlib.Path values")
    path = Path(file).expanduser().resolve()
    if path not in readable_files and not _is_within(path, output_dir):
        raise PermissionError(f"Read path was not declared: {path}")
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size > MAX_INPUT_FILE_BYTES:
        raise ValueError(f"Input file exceeds the size limit: {path}")
    return path


def _safe_open_factory(readable_files: set[Path], output_dir: Path):
    original_open = builtins.open
    written_files: set[Path] = set()
    quota = _OutputQuota()

    def safe_open(file: Any, mode: str = "r", *args: Any, **kwargs: Any):
        if not isinstance(file, (str, Path)):
            raise TypeError("File paths must be strings or pathlib.Path values")
        path = Path(file).expanduser().resolve()
        writing = any(flag in mode for flag in ("w", "a", "x", "+"))
        if writing:
            if not _is_within(path, output_dir):
                raise PermissionError(f"Writes must stay inside {output_dir}: {path}")
            if path not in written_files:
                if len(written_files) >= MAX_OUTPUT_FILES:
                    raise PermissionError(
                        f"Output file limit of {MAX_OUTPUT_FILES} was exceeded"
                    )
                written_files.add(path)
            path.parent.mkdir(parents=True, exist_ok=True)
        else:
            path = _readable_path(path, readable_files, output_dir)
        opened = original_open(path, mode, *args, **kwargs)
        return _QuotaFile(opened, quota) if writing else opened

    return safe_open


class _SafeNumpyProxy:
    """Expose NumPy while forcing the risky loading options off."""

    def __init__(self, readable_files: set[Path], output_dir: Path) -> None:
        self._readable_files = readable_files
        self._output_dir = output_dir

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_") or name in FORBIDDEN_ATTRIBUTES:
            raise AttributeError(f"Forbidden NumPy attribute: {name}")
        return getattr(np, name)

    def loadtxt(self, file: Any, *args: Any, **kwargs: Any) -> np.ndarray:
        path = _readable_path(file, self._readable_files, self._output_dir)
        return np.loadtxt(path, *args, **kwargs)

    def genfromtxt(self, file: Any, *args: Any, **kwargs: Any) -> np.ndarray:
        path = _readable_path(file, self._readable_files, self._output_dir)
        return np.genfromtxt(path, *args, **kwargs)

    def fromregex(self, file: Any, *args: Any, **kwargs: Any) -> np.ndarray:
        path = _readable_path(file, self._readable_files, self._output_dir)
        return np.fromregex(path, *args, **kwargs)

    def recfromcsv(self, file: Any, *args: Any, **kwargs: Any) -> np.ndarray:
        path = _readable_path(file, self._readable_files, self._output_dir)
        return np.recfromcsv(path, *args, **kwargs)

    def recfromtxt(self, file: Any, *args: Any, **kwargs: Any) -> np.ndarray:
        path = _readable_path(file, self._readable_files, self._output_dir)
        return np.recfromtxt(path, *args, **kwargs)

    def load(
        self,
        file: Any,
        *,
        mmap_mode: str | None = None,
        allow_pickle: bool = False,
        fix_imports: bool = True,
        encoding: str = "ASCII",
        max_header_size: int = 10_000,
    ) -> np.ndarray:
        if allow_pickle:
            raise PermissionError("np.load(..., allow_pickle=True) is forbidden")
        if mmap_mode is not None:
            raise PermissionError("np.load memory mapping is forbidden")
        path = _readable_path(file, self._readable_files, self._output_dir)
        load_kwargs = {
            "mmap_mode": None,
            "allow_pickle": False,
            "fix_imports": fix_imports,
            "encoding": encoding,
            "max_header_size": max_header_size,
        }
        supported = inspect.signature(np.load).parameters
        loaded = np.load(
            path,
            **{name: value for name, value in load_kwargs.items() if name in supported},
        )
        if not isinstance(loaded, np.ndarray):
            close = getattr(loaded, "close", None)
            if close is not None:
                close()
            raise TypeError("np.load accepts numeric .npy arrays only in generic_compute")
        if loaded.dtype.hasobject:
            raise TypeError("Object arrays are forbidden in generic_compute")
        if loaded.size > MAX_ARRAY_ELEMENTS:
            raise ValueError(
                f"Loaded array exceeds the {MAX_ARRAY_ELEMENTS}-element limit"
            )
        return loaded

    @staticmethod
    def _validate_saved_arrays(values: Any) -> None:
        total_elements = 0
        for value in values:
            array = np.asarray(value)
            if array.dtype.hasobject:
                raise TypeError("Object arrays are forbidden in generic_compute")
            total_elements += int(array.size)
            if total_elements > MAX_ARRAY_ELEMENTS:
                raise ValueError(
                    f"Saved arrays exceed the {MAX_ARRAY_ELEMENTS}-element limit"
                )

    def save(
        self,
        file: Any,
        array: Any,
        allow_pickle: bool = False,
        fix_imports: bool = True,
    ) -> None:
        if allow_pickle:
            raise PermissionError("np.save(..., allow_pickle=True) is forbidden")
        self._validate_saved_arrays((array,))
        path = Path(file)
        if path.suffix != ".npy":
            path = Path(str(path) + ".npy")
        save_kwargs = {"allow_pickle": False, "fix_imports": fix_imports}
        supported = inspect.signature(np.save).parameters
        with builtins.open(path, "wb") as handle:
            np.save(
                handle,
                array,
                **{
                    name: value
                    for name, value in save_kwargs.items()
                    if name in supported
                },
            )

    def savez(self, file: Any, *args: Any, **kwargs: Any) -> None:
        self._savez(np.savez, file, args, kwargs)

    def savez_compressed(self, file: Any, *args: Any, **kwargs: Any) -> None:
        self._savez(np.savez_compressed, file, args, kwargs)

    def _savez(
        self,
        saver: Any,
        file: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        self._validate_saved_arrays((*args, *kwargs.values()))
        path = Path(file)
        if path.suffix != ".npz":
            path = Path(str(path) + ".npz")
        with builtins.open(path, "wb") as handle:
            saver(handle, *args, **kwargs)


def _finite_stat(array: np.ndarray, reducer: Any) -> float | None:
    finite = np.asarray(array)
    finite = finite[np.isfinite(finite)]
    return float(reducer(finite)) if finite.size else None


def _safe_mapping_key(value: Any) -> str:
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return str(value)
    raise UnsafeResultError(
        f"result mapping key type is unsupported: {type(value).__name__}"
    )


def _json_safe(
    value: Any,
    *,
    depth: int = 0,
    state: _ResultState | None = None,
) -> Any:
    if depth > MAX_RESULT_DEPTH:
        raise UnsafeResultError(f"result exceeds the depth limit of {MAX_RESULT_DEPTH}")
    if state is None:
        state = _ResultState()
    if isinstance(value, np.generic):
        return _json_safe(value.item(), depth=depth, state=state)
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise UnsafeResultError("object arrays are forbidden in results")
        if value.size <= MAX_INLINE_ITEMS:
            return _json_safe(value.tolist(), depth=depth + 1, state=state)
        state.consume(1)
        summary: dict[str, Any] = {
            "omitted_from_result": True,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "item_count": int(value.size),
        }
        if value.dtype.kind in "biuf":
            summary.update(
                {
                    "finite_min": _finite_stat(value, np.min),
                    "finite_max": _finite_stat(value, np.max),
                    "finite_mean": _finite_stat(value, np.mean),
                }
            )
        elif value.dtype.kind == "c":
            summary["finite_count"] = int(np.count_nonzero(np.isfinite(value)))
        return summary
    if isinstance(value, dict):
        if len(value) > MAX_RESULT_CONTAINER_ITEMS:
            raise UnsafeResultError(
                f"result mapping exceeds the {MAX_RESULT_CONTAINER_ITEMS}-item limit"
            )
        identifier = id(value)
        if identifier in state.active_ids:
            raise UnsafeResultError("result contains a recursive reference")
        state.consume(len(value))
        state.active_ids.add(identifier)
        try:
            return {
                _safe_mapping_key(key): _json_safe(
                    item, depth=depth + 1, state=state
                )
                for key, item in value.items()
            }
        finally:
            state.active_ids.remove(identifier)
    if isinstance(value, (list, tuple, set)):
        if len(value) > MAX_RESULT_CONTAINER_ITEMS:
            raise UnsafeResultError(
                f"result collection exceeds the {MAX_RESULT_CONTAINER_ITEMS}-item limit"
            )
        identifier = id(value)
        if identifier in state.active_ids:
            raise UnsafeResultError("result contains a recursive reference")
        state.consume(len(value))
        state.active_ids.add(identifier)
        try:
            return [
                _json_safe(item, depth=depth + 1, state=state) for item in value
            ]
        finally:
            state.active_ids.remove(identifier)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, str):
        if len(value) > MAX_RESULT_STRING_CHARACTERS:
            raise UnsafeResultError(
                "result string exceeds the character limit of "
                f"{MAX_RESULT_STRING_CHARACTERS}"
            )
        return value
    if value is None or isinstance(value, (int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    raise UnsafeResultError(f"result type is unsupported: {type(value).__name__}")


def _safe_builtins(safe_open: Any) -> dict[str, Any]:
    names = (
        "Exception",
        "ValueError",
        "RuntimeError",
        "abs",
        "all",
        "any",
        "bool",
        "dict",
        "enumerate",
        "filter",
        "float",
        "int",
        "isinstance",
        "len",
        "list",
        "map",
        "max",
        "min",
        "open",
        "pow",
        "print",
        "range",
        "reversed",
        "round",
        "set",
        "slice",
        "sorted",
        "str",
        "sum",
        "tuple",
        "zip",
    )
    safe = {name: getattr(builtins, name) for name in names if name != "open"}
    safe["open"] = safe_open
    return safe


def _execute(payload: dict[str, Any]) -> dict[str, Any]:
    code = str(payload.get("code", ""))
    if not code.strip():
        raise ValueError("code must be non-empty")
    if len(code) > MAX_CODE_CHARACTERS:
        raise ValueError("code exceeds the generic-compute size limit")
    tree = _validate_code(code)
    raw_input_paths = payload.get("input_paths", [])
    if not isinstance(raw_input_paths, list):
        raise TypeError("input_paths must be a list")
    if len(raw_input_paths) > MAX_INPUT_FILES:
        raise ValueError(f"input_paths exceeds the {MAX_INPUT_FILES}-file limit")
    input_paths = tuple(str(Path(path).resolve()) for path in raw_input_paths)
    output_dir = Path(payload["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    readable_files = {Path(path).resolve() for path in input_paths}
    missing = sorted(str(path) for path in readable_files if not path.is_file())
    if missing:
        raise FileNotFoundError("Declared input file(s) missing: " + ", ".join(missing))
    oversized = sorted(
        str(path) for path in readable_files if path.stat().st_size > MAX_INPUT_FILE_BYTES
    )
    if oversized:
        raise ValueError("Declared input file(s) exceed the size limit: " + ", ".join(oversized))

    resource_limits = _apply_resource_limits()
    safe_open = _safe_open_factory(readable_files, output_dir)

    namespace: dict[str, Any] = {
        "__builtins__": _safe_builtins(safe_open),
        "np": _SafeNumpyProxy(readable_files, output_dir),
        "scipy": scipy,
        "math": math,
        "json": json,
        "input_paths": input_paths,
        "output_dir": str(output_dir),
    }
    stdout = _BoundedTextWriter(MAX_STDOUT_CHARACTERS)
    original_open = builtins.open
    try:
        builtins.open = safe_open
        with contextlib.redirect_stdout(stdout):
            exec(compile(tree, "<generic_compute>", "exec"), namespace, namespace)
    finally:
        builtins.open = original_open

    if "result" not in namespace:
        raise ValueError("generic_compute code must assign its final values to `result`")
    return {
        "status": "ok",
        "summary": {
            "result": _json_safe(namespace["result"]),
            "stdout": stdout.getvalue(),
            "stdout_truncated": stdout.truncated,
            "stdout_omitted_characters": stdout.omitted_characters,
        },
        "execution": {
            "numpy_version": np.__version__,
            "scipy_version": scipy.__version__,
            "resource_limits": resource_limits,
        },
    }


def _error_response(exc: Exception) -> dict[str, Any]:
    message = str(exc)
    if len(message) > MAX_RESULT_STRING_CHARACTERS:
        message = message[:MAX_RESULT_STRING_CHARACTERS] + "…"
    return {
        "status": "error",
        "message": message,
        "error_type": type(exc).__name__,
        "summary": {},
    }


def main() -> int:
    try:
        raw_request = sys.stdin.read(MAX_REQUEST_CHARACTERS + 1)
        if len(raw_request) > MAX_REQUEST_CHARACTERS:
            raise ValueError("Worker request exceeds the size limit")
        payload = json.loads(raw_request)
        if not isinstance(payload, dict):
            raise TypeError("Worker request must be a JSON object")
        response = _execute(payload)
    except Exception as exc:
        response = _error_response(exc)
    try:
        encoded = json.dumps(response, ensure_ascii=False, allow_nan=False)
        if len(encoded.encode("utf-8")) > MAX_RESPONSE_BYTES:
            raise UnsafeResultError("Worker response exceeds the byte limit")
    except Exception as exc:
        response = _error_response(exc)
        encoded = json.dumps(response, ensure_ascii=False, allow_nan=False)
    sys.stdout.write(encoded)
    return 0 if response["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
