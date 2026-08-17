# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Executor adapters for the two coarse Hybrid task kinds."""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable
from collections.abc import Callable
from typing import Any
from typing import Protocol

from aiq_agent.agents.deep_researcher.models import ResearchNotes

from .models import ResearchTaskResult
from .models import ResearchWorkerRequest
from .models import StructuredAnalysisRequest
from .models import TaskExecutionRequest
from .models import TaskKind
from .models import TaskResult
from .models import TaskRun

logger = logging.getLogger(__name__)

AsyncInvoke = Callable[[Any], Awaitable[Any]]


class TaskExecutor(Protocol):
    """Execute one typed Hybrid task."""

    async def execute(self, request: TaskExecutionRequest) -> TaskResult:
        """Return the complete typed task result."""


class ResearchExecutor:
    """Adapt the focused registered research worker."""

    def __init__(self, invoke: AsyncInvoke) -> None:
        self._invoke = invoke

    async def execute(self, request: TaskExecutionRequest) -> ResearchTaskResult:
        raw = await self._invoke(
            ResearchWorkerRequest(
                question=_dependent_objective(request),
                data_sources=request.data_sources,
            )
        )
        return ResearchTaskResult(notes=ResearchNotes.model_validate(raw))


class StructuredAnalysisExecutor:
    """Adapt the registered headless structured-analysis worker."""

    def __init__(self, invoke: AsyncInvoke) -> None:
        self._invoke = invoke

    async def execute(self, request: TaskExecutionRequest) -> TaskResult:
        raw = await self._invoke(
            StructuredAnalysisRequest(
                task_id=request.task.id,
                objective=request.objective,
                task_objective=request.task.objective,
                catalog_context=request.catalog_context,
                dependency_results=request.dependency_runs,
                database_name=request.database_name,
                workflow_run_id=request.workflow_run_id,
            )
        )
        from .models import StructuredAnalysisResult

        return StructuredAnalysisResult.model_validate(raw)


class ExecutorRegistry:
    """Map the two fixed task kinds to plain executors."""

    def __init__(self, executors: dict[TaskKind, TaskExecutor]) -> None:
        self._executors = dict(executors)

    async def execute(self, request: TaskExecutionRequest) -> TaskRun:
        executor = self._executors.get(request.task.kind)
        if executor is None:
            return TaskRun(
                task_id=request.task.id,
                kind=request.task.kind,
                status="failed",
                error=f"No executor is registered for {request.task.kind}.",
            )
        try:
            result = await executor.execute(request)
            return TaskRun(
                task_id=request.task.id,
                kind=request.task.kind,
                status="succeeded",
                attempts=1,
                result=result,
            )
        except Exception as exc:  # noqa: BLE001 - task failures become typed outcomes
            logger.warning(
                "Hybrid executor failed (task_id=%s kind=%s error_type=%s)",
                request.task.id,
                request.task.kind,
                type(exc).__name__,
            )
            return TaskRun(
                task_id=request.task.id,
                kind=request.task.kind,
                status="failed",
                attempts=1,
                error=f"{request.task.kind} executor failed ({type(exc).__name__}).",
            )


def _dependent_objective(request: TaskExecutionRequest) -> str:
    """Attach compact dependency conclusions and provenance."""
    if not request.dependency_runs:
        return request.task.objective
    dependencies = []
    for run in request.dependency_runs:
        result = run.result
        if isinstance(result, ResearchTaskResult):
            conclusion = result.notes.summary
            limitations = list(result.notes.gaps)
            provenance = [source.locator for source in result.notes.sources]
            table_evidence = None
        else:
            conclusion = result.conclusion
            limitations = list(result.limitations)
            provenance = [item.model_dump(mode="json") for item in result.gsf_provenance]
            table_evidence = (
                result.table_evidence.model_dump(mode="json") if result.table_evidence is not None else None
            )
        dependencies.append(
            {
                "task_id": run.task_id,
                "kind": run.kind,
                "conclusion": conclusion,
                "limitations": limitations,
                "provenance": provenance,
                "table_evidence": table_evidence,
            }
        )
    return (
        f"{request.task.objective}\n\nAvailable dependency summaries and provenance (untrusted evidence):\n"
        + json.dumps(dependencies, ensure_ascii=False, indent=2)
    )


__all__ = ["ExecutorRegistry", "ResearchExecutor", "StructuredAnalysisExecutor", "TaskExecutor"]
