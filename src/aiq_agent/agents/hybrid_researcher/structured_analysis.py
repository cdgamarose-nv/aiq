# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded headless ReAct worker for enterprise retrieval and analysis."""

from __future__ import annotations

import asyncio
import json
import logging
import posixpath
import shlex
from collections.abc import Awaitable
from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gsf.errors import GSFErrorCode
from gsf.errors import GSFToolError
from gsf.models import TextToSQLRequest
from gsf.models import TextToSQLResponse
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware import ToolCallLimitMiddleware
from langchain.agents.middleware.types import ModelResponse
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

from aiq_agent.agents.deep_researcher.sandbox import SandboxProvider
from aiq_agent.common import load_prompt
from aiq_agent.common import render_prompt_template

from .models import GSF_PROVENANCE_MAX_COLUMNS
from .models import GSF_PROVENANCE_MAX_ERROR_CHARS
from .models import GSF_PROVENANCE_MAX_IDENTIFIER_CHARS
from .models import GSF_PROVENANCE_MAX_METADATA_CHARS
from .models import GSF_PROVENANCE_MAX_METADATA_ITEMS
from .models import STRUCTURED_CONCLUSION_MAX_CHARS
from .models import STRUCTURED_MAX_LIMITATIONS
from .models import GSFQueryError
from .models import GSFQueryProvenance
from .models import GSFQuerySuccess
from .models import GSFResultColumnSummary
from .models import GSFSemanticContextSummary
from .models import ResearchTaskResult
from .models import StructuredAnalysisRequest
from .models import StructuredAnalysisResult
from .models import StructuredConclusionText
from .models import StructuredLimitationText

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent / "prompts"
_MANIFEST_FILENAME = "manifest.json"

GSFInvoke = Callable[[Any], Awaitable[Any]]
SandboxFactory = Callable[[str], SandboxProvider]


class StructuredAnalysisError(RuntimeError):
    """Base error for the structured-analysis worker."""


class StructuredAnalysisTimeoutError(StructuredAnalysisError):
    """The complete structured trajectory exceeded its deadline."""


class StructuredAnalysisUploadError(StructuredAnalysisError):
    """A complete GSF response could not be persisted for Python."""


class _QueryGSFInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=4_096)


class _ExecutePythonInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str = Field(min_length=1)


class _StructuredConclusion(BaseModel):
    """Model-authored portion of a structured-analysis result."""

    model_config = ConfigDict(extra="forbid")

    sufficiency: str = Field(pattern=r"^(sufficient|limited|insufficient)$")
    content: StructuredConclusionText
    limitations: tuple[StructuredLimitationText, ...] = Field(default=(), max_length=STRUCTURED_MAX_LIMITATIONS)


@dataclass(frozen=True)
class _GSFQueryAttempt:
    """Worker-local GSF outcome retained only while one trajectory is active."""

    question: str
    outcome: TextToSQLResponse | GSFToolError


class _SequentialGSFMiddleware(AgentMiddleware):
    """Reject every GSF call when a model emits multiple calls in one turn."""

    def __init__(self) -> None:
        self._rejected_call_ids: set[str] = set()

    async def awrap_model_call(self, request, handler):
        response = await handler(request)
        if not isinstance(response, ModelResponse):
            return response
        for message in response.result:
            if not isinstance(message, AIMessage):
                continue
            calls = [call for call in message.tool_calls if call.get("name") == "query_gsf"]
            if len(calls) > 1:
                self._rejected_call_ids.update(str(call.get("id", "")) for call in calls)
        return response

    async def awrap_tool_call(self, request, handler):
        call_id = str(request.tool_call.get("id", ""))
        if call_id not in self._rejected_call_ids:
            return await handler(request)
        self._rejected_call_ids.discard(call_id)
        return ToolMessage(
            content=json.dumps(
                {
                    "status": "error",
                    "code": "invalid_request",
                    "retryable": False,
                    "message": "Multiple query_gsf calls in one model turn are not allowed; observe each result first.",
                }
            ),
            tool_call_id=call_id or "parallel-gsf-rejected",
            name="query_gsf",
            status="error",
        )


class StructuredAnalysisWorker:
    """Run one observation-driven GSF and pandas trajectory."""

    def __init__(
        self,
        *,
        llm: BaseChatModel,
        gsf_invoke: GSFInvoke,
        sandbox_factory: SandboxFactory,
        database_name: str | None = None,
        sql_max_rows: int = 1_000,
        callbacks: Sequence[BaseCallbackHandler] = (),
        prompt_template: str | None = None,
        timeout_seconds: float = 600,
        execute_timeout_seconds: int = 60,
        max_gsf_calls: int = 4,
        max_python_calls: int = 4,
        max_code_chars: int = 40_000,
        max_output_chars: int = 40_000,
        model_result_rows: int = 25,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if execute_timeout_seconds < 1:
            raise ValueError("execute_timeout_seconds must be positive")
        if max_gsf_calls < 1 or max_python_calls < 1:
            raise ValueError("tool-call limits must be positive")
        if sql_max_rows < 1 or model_result_rows < 1:
            raise ValueError("row limits must be positive")
        self._llm = llm
        self._gsf_invoke = gsf_invoke
        self._sandbox_factory = sandbox_factory
        self._database_name = database_name
        self._sql_max_rows = sql_max_rows
        self._callbacks = list(callbacks)
        self._prompt_template = prompt_template or load_prompt(_PROMPTS_DIR, "structured_analysis")
        self._timeout_seconds = timeout_seconds
        self._execute_timeout_seconds = execute_timeout_seconds
        self._max_gsf_calls = max_gsf_calls
        self._max_python_calls = max_python_calls
        self._max_code_chars = max_code_chars
        self._max_output_chars = max_output_chars
        self._model_result_rows = model_result_rows

    async def run(self, request: StructuredAnalysisRequest) -> StructuredAnalysisResult:
        """Execute one bounded trajectory and return only compact terminal evidence."""
        provider = self._sandbox_factory(f"hybrid-structured-{request.task_id}")
        terminate = False
        attempts: list[_GSFQueryAttempt] = []
        question_keys: set[str] = set()
        tool_lock = asyncio.Lock()
        effective_database = request.database_name or self._database_name

        try:
            async with asyncio.timeout(self._timeout_seconds):
                query_gsf = self._build_gsf_tool(
                    provider,
                    request,
                    attempts,
                    question_keys,
                    tool_lock,
                    effective_database,
                )
                execute_python = self._build_python_tool(provider, request, attempts, tool_lock)
                prompt = render_prompt_template(
                    self._prompt_template,
                    max_gsf_calls=self._max_gsf_calls,
                    max_python_calls=self._max_python_calls,
                    max_conclusion_chars=STRUCTURED_CONCLUSION_MAX_CHARS,
                )
                agent = create_agent(
                    model=self._llm,
                    tools=[query_gsf, execute_python],
                    system_prompt=prompt,
                    middleware=[
                        _SequentialGSFMiddleware(),
                        ToolCallLimitMiddleware(
                            tool_name="query_gsf",
                            run_limit=self._max_gsf_calls,
                            exit_behavior="error",
                        ),
                        ToolCallLimitMiddleware(
                            tool_name="execute_python",
                            run_limit=self._max_python_calls,
                            exit_behavior="error",
                        ),
                    ],
                    response_format=_StructuredConclusion,
                )
                config: dict[str, Any] = {"run_name": "hybrid-structured-analysis-worker"}
                if self._callbacks:
                    config["callbacks"] = self._callbacks
                result = await agent.ainvoke(
                    {"messages": [{"role": "user", "content": json.dumps(_request_context(request))}]},
                    config=config,
                )
                structured = result.get("structured_response") if isinstance(result, Mapping) else None
                conclusion = _StructuredConclusion.model_validate(structured)
                return StructuredAnalysisResult(
                    sufficiency=conclusion.sufficiency,
                    conclusion=conclusion.content,
                    gsf_provenance=tuple(_provenance_summary(attempt) for attempt in attempts),
                    limitations=conclusion.limitations,
                )
        except TimeoutError as exc:
            terminate = True
            raise StructuredAnalysisTimeoutError(
                f"Structured analysis exceeded its {self._timeout_seconds:g}-second deadline"
            ) from exc
        except asyncio.CancelledError:
            terminate = True
            raise
        finally:
            cleanup = provider.terminate if terminate else provider.close
            try:
                await asyncio.shield(asyncio.to_thread(cleanup))
            except Exception as exc:  # noqa: BLE001 - cleanup must not hide task outcome
                logger.warning(
                    "Structured-analysis sandbox cleanup failed (task_id=%s error_type=%s)",
                    request.task_id,
                    type(exc).__name__,
                )

    def _build_gsf_tool(
        self,
        provider: SandboxProvider,
        request: StructuredAnalysisRequest,
        attempts: list[_GSFQueryAttempt],
        question_keys: set[str],
        tool_lock: asyncio.Lock,
        database_name: str | None,
    ):
        @tool("query_gsf", args_schema=_QueryGSFInput)
        async def query_gsf(question: str) -> str:
            """Ask one natural-language enterprise question and inspect the validated result before asking another."""
            key = " ".join(question.split()).casefold()
            async with tool_lock:
                if attempts and isinstance(attempts[-1].outcome, GSFToolError):
                    error = GSFToolError(
                        code=GSFErrorCode.INVALID_REQUEST,
                        retryable=False,
                        message=(
                            "A prior GSF failure provides no semantic observation for a corrected follow-up. "
                            "Stop and report the limitation."
                        ),
                    )
                    attempts.append(_GSFQueryAttempt(question=question, outcome=error))
                    return error.model_dump_json(exclude_none=True)
                if key in question_keys:
                    error = GSFToolError(
                        code=GSFErrorCode.INVALID_REQUEST,
                        retryable=False,
                        message="An exact repeated GSF question is not allowed.",
                    )
                    attempts.append(_GSFQueryAttempt(question=question, outcome=error))
                    return error.model_dump_json(exclude_none=True)
                if len(attempts) >= self._max_gsf_calls:
                    raise StructuredAnalysisError(f"query_gsf allows at most {self._max_gsf_calls} calls")
                question_keys.add(key)

            raw = await self._gsf_invoke(
                TextToSQLRequest(
                    question=question,
                    database_name=database_name,
                    max_rows=self._sql_max_rows,
                )
            )
            outcome = _parse_gsf_outcome(raw)
            async with tool_lock:
                if isinstance(outcome, GSFToolError):
                    attempts.append(_GSFQueryAttempt(question=question, outcome=outcome))
                    return outcome.model_dump_json(exclude_none=True)
                attempt = _GSFQueryAttempt(question=question, outcome=outcome)
                manifest_path = await asyncio.to_thread(
                    _persist_gsf_inputs,
                    provider,
                    request.task_id,
                    (*attempts, attempt),
                )
                attempts.append(attempt)
            return json.dumps(
                _model_projection(outcome, manifest_path, self._model_result_rows),
                ensure_ascii=False,
                default=str,
            )

        return query_gsf

    def _build_python_tool(
        self,
        provider: SandboxProvider,
        request: StructuredAnalysisRequest,
        attempts: list[_GSFQueryAttempt],
        tool_lock: asyncio.Lock,
    ):
        python_calls = 0

        @tool("execute_python", args_schema=_ExecutePythonInput)
        async def execute_python(code: str) -> str:
            """Run pandas/Python over complete successful GSF responses listed in the generated manifest."""
            nonlocal python_calls
            async with tool_lock:
                python_calls += 1
                if python_calls > self._max_python_calls:
                    raise StructuredAnalysisError(f"execute_python allows at most {self._max_python_calls} calls")
                if not any(isinstance(attempt.outcome, TextToSQLResponse) for attempt in attempts):
                    raise StructuredAnalysisError("execute_python is unavailable until one GSF query succeeds")
            if len(code) > self._max_code_chars:
                raise StructuredAnalysisError(f"Python code exceeds {self._max_code_chars} characters")

            task_dir = posixpath.join(provider.workdir, "hybrid", request.task_id)
            script_path = posixpath.join(task_dir, f"analysis-{python_calls}.py")
            output_path = posixpath.join(task_dir, f"analysis-{python_calls}.output.json")
            _upload_or_raise(provider, [(script_path, code.encode("utf-8"))])
            response = await asyncio.to_thread(
                provider.execute,
                f"python3 {shlex.quote(script_path)}",
                timeout=self._execute_timeout_seconds,
            )
            output, worker_truncated = _truncate_text(response.output, self._max_output_chars)
            payload = {
                "exit_code": response.exit_code,
                "output": output,
                "truncated": bool(response.truncated or worker_truncated),
            }
            _upload_or_raise(provider, [(output_path, json.dumps(payload, ensure_ascii=False).encode("utf-8"))])
            return json.dumps({**payload, "code_artifact": script_path, "output_artifact": output_path})

        return execute_python


def _request_context(request: StructuredAnalysisRequest) -> dict[str, Any]:
    catalog = request.catalog_context
    dependencies: list[dict[str, Any]] = []
    for run in request.dependency_results:
        if isinstance(run.result, ResearchTaskResult):
            conclusion = run.result.notes.summary
            limitations = list(run.result.notes.gaps)
            provenance = [source.locator for source in run.result.notes.sources]
        else:
            conclusion = run.result.conclusion
            limitations = list(run.result.limitations)
            provenance = [item.model_dump(mode="json") for item in run.result.gsf_provenance]
        dependencies.append(
            {
                "task_id": run.task_id,
                "kind": run.kind,
                "conclusion": conclusion,
                "limitations": limitations,
                "provenance": provenance,
            }
        )
    return {
        "overall_objective": request.objective,
        "structured_task": request.task_objective,
        "catalog_context": {
            "truncated": catalog.truncated,
            "uncovered_entities": catalog.uncovered_entities or [],
            "candidates": [
                {"label": candidate.label, "attribute": candidate.attribute, "term": candidate.term}
                for candidate in catalog.candidates
            ],
        },
        "dependency_conclusions": dependencies,
    }


def _provenance_summary(attempt: _GSFQueryAttempt) -> GSFQueryProvenance:
    """Compact one worker-local outcome before it crosses into workflow state."""
    outcome = attempt.outcome
    if isinstance(outcome, GSFToolError):
        message, message_truncated = _bounded_text(outcome.message, GSF_PROVENANCE_MAX_ERROR_CHARS)
        request_id, request_id_truncated = _optional_bounded_text(
            outcome.request_id,
            GSF_PROVENANCE_MAX_IDENTIFIER_CHARS,
        )
        return GSFQueryError(
            question=attempt.question,
            request_id=request_id,
            code=outcome.code.value,
            retryable=outcome.retryable,
            message=message or "GSF query failed.",
            metadata_truncated=request_id_truncated or message_truncated,
        )

    request_id, request_id_truncated = _optional_bounded_text(
        outcome.request_id,
        GSF_PROVENANCE_MAX_IDENTIFIER_CHARS,
    )
    citation_key, citation_key_truncated = _bounded_text(
        outcome.citation_key,
        GSF_PROVENANCE_MAX_IDENTIFIER_CHARS,
    )
    columns: list[GSFResultColumnSummary] = []
    columns_truncated = len(outcome.columns) > GSF_PROVENANCE_MAX_COLUMNS
    for column in outcome.columns[:GSF_PROVENANCE_MAX_COLUMNS]:
        name, name_truncated = _bounded_text(column.name, GSF_PROVENANCE_MAX_METADATA_CHARS)
        data_type, data_type_truncated = _optional_bounded_text(
            column.data_type,
            GSF_PROVENANCE_MAX_METADATA_CHARS,
        )
        columns.append(GSFResultColumnSummary(name=name, data_type=data_type))
        columns_truncated = columns_truncated or name_truncated or data_type_truncated
    assumptions, assumptions_truncated = _bounded_items(outcome.assumptions or ())
    warnings, warnings_truncated = _bounded_items(outcome.warnings or ())
    semantic_context, semantic_context_truncated = _semantic_context_summary(outcome)
    return GSFQuerySuccess(
        question=attempt.question,
        request_id=request_id,
        citation_key=citation_key,
        returned_row_count=outcome.returned_row_count,
        result_truncated=outcome.truncated,
        columns=tuple(columns),
        semantic_context=semantic_context,
        assumptions=assumptions,
        warnings=warnings,
        metadata_truncated=any(
            (
                request_id_truncated,
                citation_key_truncated,
                columns_truncated,
                semantic_context_truncated,
                assumptions_truncated,
                warnings_truncated,
            )
        ),
    )


def _semantic_context_summary(response: TextToSQLResponse) -> tuple[GSFSemanticContextSummary | None, bool]:
    context = response.semantic_context
    if context is None:
        return None, False
    grain, truncated = _optional_bounded_text(context.grain, GSF_PROVENANCE_MAX_METADATA_CHARS)
    truncated = truncated or bool(context.metrics)
    projected: dict[str, tuple[str, ...]] = {}
    for name, values in (
        ("units", context.units),
        ("filters", context.filters),
        ("rules", context.rules),
        ("omissions", context.omissions),
    ):
        projected[name], field_truncated = _bounded_items(values)
        truncated = truncated or field_truncated
    return GSFSemanticContextSummary(grain=grain, **projected), truncated


def _bounded_items(values: Iterable[Any]) -> tuple[tuple[str, ...], bool]:
    items = list(values)
    projected: list[str] = []
    truncated = len(items) > GSF_PROVENANCE_MAX_METADATA_ITEMS
    for value in items[:GSF_PROVENANCE_MAX_METADATA_ITEMS]:
        text, item_truncated = _bounded_text(str(value), GSF_PROVENANCE_MAX_METADATA_CHARS)
        projected.append(text)
        truncated = truncated or item_truncated
    return tuple(projected), truncated


def _bounded_text(value: str, max_chars: int) -> tuple[str, bool]:
    if len(value) <= max_chars:
        return value, False
    suffix = " [truncated]"
    return f"{value[: max_chars - len(suffix)]}{suffix}", True


def _optional_bounded_text(value: str | None, max_chars: int) -> tuple[str | None, bool]:
    if not value:
        return None, value is not None
    return _bounded_text(value, max_chars)


def _parse_gsf_outcome(raw: Any) -> TextToSQLResponse | GSFToolError:
    if isinstance(raw, TextToSQLResponse | GSFToolError):
        return raw
    if isinstance(raw, str):
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise StructuredAnalysisError("GSF returned invalid JSON") from exc
    else:
        payload = raw
    if not isinstance(payload, Mapping):
        raise StructuredAnalysisError("GSF returned a non-object response")
    if payload.get("status") == "error":
        return GSFToolError.model_validate(payload)
    return TextToSQLResponse.model_validate(payload)


def _persist_gsf_inputs(
    provider: SandboxProvider,
    task_id: str,
    attempts: Sequence[_GSFQueryAttempt],
) -> str:
    input_dir = posixpath.join(provider.workdir, "hybrid", task_id, "gsf")
    uploads: list[tuple[str, bytes]] = []
    entries: list[dict[str, Any]] = []
    for index, attempt in enumerate(attempts, start=1):
        if not isinstance(attempt.outcome, TextToSQLResponse):
            continue
        path = posixpath.join(input_dir, f"response-{index}.json")
        payload = attempt.outcome.model_dump_json(indent=2).encode("utf-8")
        uploads.append((path, payload))
        entries.append({"attempt": index, "question": attempt.question, "path": path, "bytes": len(payload)})
    manifest_path = posixpath.join(input_dir, _MANIFEST_FILENAME)
    manifest = {"version": 1, "manifest_path": manifest_path, "successful_gsf_responses": entries}
    uploads.append((manifest_path, json.dumps(manifest, indent=2).encode("utf-8")))
    _upload_or_raise(provider, uploads)
    return manifest_path


def _upload_or_raise(provider: SandboxProvider, uploads: list[tuple[str, bytes]]) -> None:
    responses = provider.upload_files(uploads)
    if len(responses) != len(uploads) or any(getattr(response, "error", None) is not None for response in responses):
        raise StructuredAnalysisUploadError("The structured-analysis sandbox rejected an artifact")


def _model_projection(response: TextToSQLResponse, manifest_path: str, max_rows: int) -> dict[str, Any]:
    return {
        "status": "success",
        "request_id": response.request_id,
        "sql": response.sql,
        "columns": [column.model_dump(mode="json") for column in response.columns],
        "returned_row_count": response.returned_row_count,
        "rows": response.rows[:max_rows],
        "rows_projection_truncated": len(response.rows) > max_rows,
        "result_truncated": response.truncated,
        "semantic_context": response.semantic_context.model_dump(mode="json") if response.semantic_context else None,
        "warnings": response.warnings or [],
        "assumptions": response.assumptions or [],
        "manifest_path": manifest_path,
        "citation_key": response.citation_key,
    }


def _truncate_text(value: Any, max_chars: int) -> tuple[str, bool]:
    text = value if isinstance(value, str) else str(value)
    if len(text) <= max_chars:
        return text, False
    suffix = "\n[output truncated by structured-analysis worker]"
    retained = max(0, max_chars - len(suffix))
    return f"{text[:retained]}{suffix}"[:max_chars], True


__all__ = [
    "GSFInvoke",
    "SandboxFactory",
    "StructuredAnalysisError",
    "StructuredAnalysisTimeoutError",
    "StructuredAnalysisUploadError",
    "StructuredAnalysisWorker",
]
