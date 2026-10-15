"""Per-Run model token usage tracing for LangChain-backed Agents."""

from __future__ import annotations

import threading
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

from .messages import json_safe

def _integer(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _nested_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    nested = value.get(name, {})
    return nested if isinstance(nested, Mapping) else {}


def _first_generation(response: Any) -> Any | None:
    generations = getattr(response, "generations", None)
    if not generations:
        return None
    first_group = generations[0]
    return first_group[0] if first_group else None


def _usage_from_response(response: Any) -> dict[str, Any]:
    llm_output = getattr(response, "llm_output", None)
    llm_output = llm_output if isinstance(llm_output, Mapping) else {}
    raw_usage = llm_output.get("token_usage", {})
    raw_usage = raw_usage if isinstance(raw_usage, Mapping) else {}

    generation = _first_generation(response)
    message = getattr(generation, "message", None)
    standard_usage = getattr(message, "usage_metadata", None)
    standard_usage = standard_usage if isinstance(standard_usage, Mapping) else {}

    input_tokens = _integer(raw_usage.get("prompt_tokens"))
    output_tokens = _integer(raw_usage.get("completion_tokens"))
    total_tokens = _integer(raw_usage.get("total_tokens"))
    if input_tokens is None:
        input_tokens = _integer(standard_usage.get("input_tokens"))
    if output_tokens is None:
        output_tokens = _integer(standard_usage.get("output_tokens"))
    if total_tokens is None:
        total_tokens = _integer(standard_usage.get("total_tokens"))
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens

    prompt_details = _nested_mapping(raw_usage, "prompt_tokens_details")
    completion_details = _nested_mapping(raw_usage, "completion_tokens_details")
    cache_hit_tokens = _integer(raw_usage.get("prompt_cache_hit_tokens"))
    cache_miss_tokens = _integer(raw_usage.get("prompt_cache_miss_tokens"))
    if cache_hit_tokens is None:
        cache_hit_tokens = _integer(prompt_details.get("cached_tokens"))
    if cache_hit_tokens is None:
        input_details = _nested_mapping(standard_usage, "input_token_details")
        cache_hit_tokens = _integer(input_details.get("cache_read"))
    if cache_miss_tokens is None and input_tokens is not None and cache_hit_tokens is not None:
        cache_miss_tokens = max(input_tokens - cache_hit_tokens, 0)

    reasoning_tokens = _integer(completion_details.get("reasoning_tokens"))
    if reasoning_tokens is None:
        output_details = _nested_mapping(standard_usage, "output_token_details")
        reasoning_tokens = _integer(output_details.get("reasoning"))

    generation_info = getattr(generation, "generation_info", None)
    generation_info = generation_info if isinstance(generation_info, Mapping) else {}
    response_metadata = getattr(message, "response_metadata", None)
    response_metadata = response_metadata if isinstance(response_metadata, Mapping) else {}

    return {
        "usage_available": any(
            value is not None for value in (input_tokens, output_tokens, total_tokens)
        ),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "prompt_cache_hit_tokens": cache_hit_tokens,
        "prompt_cache_miss_tokens": cache_miss_tokens,
        "reasoning_tokens": reasoning_tokens,
        "model": str(
            llm_output.get("model_name")
            or response_metadata.get("model_name")
            or ""
        ),
        "model_provider": str(
            response_metadata.get("model_provider")
            or llm_output.get("model_provider")
            or ""
        ),
        "response_id": str(
            llm_output.get("id")
            or response_metadata.get("id")
            or ""
        ),
        "finish_reason": str(
            generation_info.get("finish_reason")
            or response_metadata.get("finish_reason")
            or ""
        ),
        "raw_usage": json_safe(raw_usage),
    }


def _start_model_name(
    serialized: Mapping[str, Any] | None,
    metadata: Mapping[str, Any] | None,
) -> str:
    metadata = metadata or {}
    serialized = serialized or {}
    kwargs = serialized.get("kwargs", {})
    kwargs = kwargs if isinstance(kwargs, Mapping) else {}
    return str(
        metadata.get("ls_model_name")
        or kwargs.get("model")
        or kwargs.get("model_name")
        or serialized.get("name")
        or ""
    )


class TokenUsageCallbackHandler(BaseCallbackHandler):
    """Collect raw provider usage and normalized token counts for one Run."""

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self._active: dict[str, dict[str, Any]] = {}
        self._calls: list[dict[str, Any]] = []

    def _start(
        self,
        serialized: Mapping[str, Any] | None,
        *,
        run_id: Any,
        parent_run_id: Any = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        key = str(run_id)
        with self._lock:
            if key in self._active:
                return
            self._active[key] = {
                "sequence": len(self._calls) + len(self._active) + 1,
                "llm_run_id": key,
                "parent_run_id": str(parent_run_id) if parent_run_id else "",
                "model": _start_model_name(serialized, metadata),
                "status": "running",
            }

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        *,
        run_id: Any,
        parent_run_id: Any = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self._start(
            serialized,
            run_id=run_id,
            parent_run_id=parent_run_id,
            metadata=metadata,
        )

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: Any,
        parent_run_id: Any = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self._start(
            serialized,
            run_id=run_id,
            parent_run_id=parent_run_id,
            metadata=metadata,
        )

    def _finish(self, run_id: Any) -> dict[str, Any]:
        key = str(run_id)
        with self._lock:
            record = self._active.pop(
                key,
                {
                    "sequence": len(self._calls) + 1,
                    "llm_run_id": key,
                    "parent_run_id": "",
                    "model": "",
                    "status": "running",
                },
            )
        return record

    def _append(self, record: dict[str, Any]) -> None:
        with self._lock:
            record["sequence"] = len(self._calls) + 1
            self._calls.append(record)

    def on_llm_end(
        self,
        response: Any,
        *,
        run_id: Any,
        parent_run_id: Any = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        record = self._finish(run_id)
        usage = _usage_from_response(response)
        if usage["model"]:
            record["model"] = usage["model"]
        record.update(usage)
        record.update(
            {
                "status": "completed",
            }
        )
        self._append(record)

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: Any,
        parent_run_id: Any = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        record = self._finish(run_id)
        record.update(
            {
                "status": "failed",
                "usage_available": False,
                "input_tokens": None,
                "output_tokens": None,
                "total_tokens": None,
                "prompt_cache_hit_tokens": None,
                "prompt_cache_miss_tokens": None,
                "reasoning_tokens": None,
                "model_provider": "",
                "response_id": "",
                "finish_reason": "",
                "raw_usage": {},
                "error": {
                    "type": type(error).__name__,
                    "message": str(error),
                },
            }
        )
        self._append(record)

    def finalize_pending(self, reason: str) -> None:
        """Preserve calls that started but never emitted an end/error callback."""
        with self._lock:
            pending = list(self._active.values())
            self._active.clear()
        for record in pending:
            record.update(
                {
                    "status": "interrupted",
                    "usage_available": False,
                    "input_tokens": None,
                    "output_tokens": None,
                    "total_tokens": None,
                    "prompt_cache_hit_tokens": None,
                    "prompt_cache_miss_tokens": None,
                    "reasoning_tokens": None,
                    "model_provider": "",
                    "response_id": "",
                    "finish_reason": "",
                    "raw_usage": {},
                    "error": {
                        "type": "RunTerminated",
                        "message": reason,
                    },
                }
            )
            self._append(record)

    @property
    def calls(self) -> list[dict[str, Any]]:
        with self._lock:
            return deepcopy(self._calls)

    def summary(self) -> dict[str, Any]:
        calls = self.calls
        completed = [item for item in calls if item["status"] == "completed"]
        failed = [item for item in calls if item["status"] == "failed"]
        interrupted = [item for item in calls if item["status"] == "interrupted"]
        available = [item for item in calls if item.get("usage_available")]
        cache_available = [
            item
            for item in available
            if item.get("prompt_cache_hit_tokens") is not None
            or item.get("prompt_cache_miss_tokens") is not None
        ]

        def total(name: str) -> int:
            return sum(int(item.get(name) or 0) for item in available)

        cache_hit_tokens = total("prompt_cache_hit_tokens")
        cache_miss_tokens = total("prompt_cache_miss_tokens")
        cache_denominator = cache_hit_tokens + cache_miss_tokens
        return {
            "model_call_count": len(calls),
            "completed_model_call_count": len(completed),
            "failed_model_call_count": len(failed),
            "interrupted_model_call_count": len(interrupted),
            "usage_available_call_count": len(available),
            "missing_usage_call_count": len(calls) - len(available),
            "input_tokens": total("input_tokens"),
            "output_tokens": total("output_tokens"),
            "total_tokens": total("total_tokens"),
            "prompt_cache_hit_tokens": cache_hit_tokens,
            "prompt_cache_miss_tokens": cache_miss_tokens,
            "cache_usage_available_call_count": len(cache_available),
            "cache_usage_missing_call_count": len(available) - len(cache_available),
            "prompt_cache_hit_rate": (
                round(cache_hit_tokens / cache_denominator, 6)
                if cache_denominator
                else None
            ),
            "reasoning_tokens": total("reasoning_tokens"),
            "models": sorted({str(item["model"]) for item in calls if item.get("model")}),
        }
