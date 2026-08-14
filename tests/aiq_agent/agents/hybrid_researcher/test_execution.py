# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for coarse Hybrid task executor adapters."""

from aiq_agent.agents.chat_researcher.models import CatalogCandidate
from aiq_agent.agents.chat_researcher.models import CatalogRoutingResponse
from aiq_agent.agents.deep_researcher.models import ResearchNotes
from aiq_agent.agents.hybrid_researcher.execution import ExecutorRegistry
from aiq_agent.agents.hybrid_researcher.execution import ResearchExecutor
from aiq_agent.agents.hybrid_researcher.execution import StructuredAnalysisExecutor
from aiq_agent.agents.hybrid_researcher.models import HybridTask
from aiq_agent.agents.hybrid_researcher.models import StructuredAnalysisResult
from aiq_agent.agents.hybrid_researcher.models import TaskExecutionRequest
from aiq_agent.agents.hybrid_researcher.models import TaskRun


def _catalog() -> CatalogRoutingResponse:
    return CatalogRoutingResponse(
        coverage=1,
        candidates=[CatalogCandidate(id="metric", label="Metric", attribute="recognized", term="Revenue")],
    )


def _notes() -> ResearchNotes:
    return ResearchNotes(
        query_topic="Context",
        target_components=["context"],
        summary="Context found.",
        findings=[],
        gaps=[],
        sources=[],
        narrative_notes="Context found.",
        language="English",
    )


async def test_research_executor_supplies_dependency_conclusions():
    captured = None

    async def invoke(value):
        nonlocal captured
        captured = value
        return _notes()

    dependency = TaskRun(
        task_id="decline",
        kind="structured_analysis",
        status="succeeded",
        result=StructuredAnalysisResult(
            sufficiency="sufficient",
            conclusion="The decline was in Q2.",
            gsf_provenance=(),
        ),
    )
    task = HybridTask(
        id="period_context",
        kind="research",
        objective="Research the returned decline period.",
        depends_on=("decline",),
    )
    result = await ResearchExecutor(invoke).execute(
        TaskExecutionRequest(
            task=task,
            objective="Explain the decline.",
            catalog_context=_catalog(),
            dependency_runs=(dependency,),
        )
    )
    assert result.notes == _notes()
    assert "The decline was in Q2" in captured.question


async def test_structured_executor_passes_catalog_and_complete_dependencies():
    captured = None

    async def invoke(value):
        nonlocal captured
        captured = value
        return StructuredAnalysisResult(sufficiency="sufficient", conclusion="Complete.", gsf_provenance=())

    task = HybridTask(id="revenue", kind="structured_analysis", objective="Analyze revenue.")
    result = await StructuredAnalysisExecutor(invoke).execute(
        TaskExecutionRequest(
            task=task, objective="Explain revenue.", catalog_context=_catalog(), database_name="finance"
        )
    )
    assert result.conclusion == "Complete."
    assert captured.catalog_context == _catalog()
    assert captured.database_name == "finance"


async def test_executor_registry_sanitizes_worker_failure():
    class Failing:
        async def execute(self, _request):
            raise RuntimeError("secret details")

    task = HybridTask(id="revenue", kind="structured_analysis", objective="Analyze revenue.")
    run = await ExecutorRegistry({"structured_analysis": Failing()}).execute(
        TaskExecutionRequest(task=task, objective="Explain revenue.", catalog_context=_catalog())
    )
    assert run.status == "failed"
    assert run.error == "structured_analysis executor failed (RuntimeError)."
    assert "secret" not in run.error
