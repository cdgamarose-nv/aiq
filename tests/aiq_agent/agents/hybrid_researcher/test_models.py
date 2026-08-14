# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for append-only Hybrid Research contracts."""

import operator
from typing import get_args
from typing import get_type_hints

import pytest
from pydantic import ValidationError

from aiq_agent.agents.chat_researcher.models import CatalogCandidate
from aiq_agent.agents.chat_researcher.models import CatalogRoutingResponse
from aiq_agent.agents.hybrid_researcher.models import ContinuationDecision
from aiq_agent.agents.hybrid_researcher.models import HybridResearchState
from aiq_agent.agents.hybrid_researcher.models import HybridTask
from aiq_agent.agents.hybrid_researcher.models import HybridTaskPlan
from aiq_agent.agents.hybrid_researcher.models import StructuredAnalysisResult
from aiq_agent.agents.hybrid_researcher.models import TaskExecutionRequest
from aiq_agent.agents.hybrid_researcher.models import TaskRun


def _catalog() -> CatalogRoutingResponse:
    return CatalogRoutingResponse(
        coverage=1,
        candidates=[CatalogCandidate(id="metric", label="Metric", attribute="recognized", term="Revenue")],
    )


def test_plan_allows_multiple_tasks_of_both_kinds_and_cross_kind_dependencies():
    plan = HybridTaskPlan(
        objective="Compare performance and context.",
        tasks=(
            HybridTask(id="market", kind="research", objective="Research the market."),
            HybridTask(id="revenue", kind="structured_analysis", objective="Analyze revenue."),
            HybridTask(id="customers", kind="structured_analysis", objective="Analyze customers."),
            HybridTask(
                id="revenue_period",
                kind="research",
                objective="Research the observed revenue period.",
                depends_on=("revenue",),
            ),
        ),
    )
    assert [task.kind for task in plan.tasks].count("research") == 2
    assert [task.kind for task in plan.tasks].count("structured_analysis") == 2


def test_task_contract_is_frozen_and_rejects_invalid_ids():
    task = HybridTask(id="market_context", kind="research", objective="Research context.")
    with pytest.raises(ValidationError):
        task.objective = "Mutated."
    with pytest.raises(ValidationError):
        HybridTask(id="MarketContext", kind="research", objective="Research context.")


def test_task_run_and_result_are_frozen_ledger_records():
    result = StructuredAnalysisResult(
        sufficiency="sufficient",
        conclusion="Complete.",
        gsf_provenance=(),
    )
    run = TaskRun(task_id="analysis", kind="structured_analysis", status="succeeded", result=result)
    with pytest.raises(ValidationError):
        result.conclusion = "Mutated."
    with pytest.raises(ValidationError):
        run.status = "failed"


def test_continuation_payload_matches_action():
    task = HybridTask(id="next_task", kind="research", objective="Research the discovered entity.")
    assert ContinuationDecision(action="append", new_tasks=(task,), reasoning="The entity is now known.").new_tasks
    with pytest.raises(ValidationError, match="append decisions require"):
        ContinuationDecision(action="append", reasoning="More work is needed.")
    with pytest.raises(ValidationError, match="only valid"):
        ContinuationDecision(action="finish", new_tasks=(task,), reasoning="Done.")


def test_execution_request_requires_exact_successful_dependencies():
    dependency = TaskRun(
        task_id="discover",
        kind="structured_analysis",
        status="failed",
        error="Worker failed.",
    )
    task = HybridTask(
        id="research_entity",
        kind="research",
        objective="Research the entity.",
        depends_on=("discover",),
    )
    with pytest.raises(ValidationError, match="must all be successful"):
        TaskExecutionRequest(
            task=task,
            objective="Objective.",
            catalog_context=_catalog(),
            dependency_runs=(dependency,),
        )


def test_task_run_enforces_typed_result_and_status():
    result = StructuredAnalysisResult(
        sufficiency="limited",
        conclusion="Only part of the period was returned.",
        gsf_provenance=(),
        limitations=("One period is missing.",),
    )
    assert (
        TaskRun(
            task_id="revenue",
            kind="structured_analysis",
            status="succeeded",
            result=result,
        ).result
        == result
    )
    with pytest.raises(ValidationError, match="requires a result"):
        TaskRun(task_id="revenue", kind="structured_analysis", status="succeeded")


def test_state_uses_parallel_append_reducer_and_requires_catalog_candidates():
    state = HybridResearchState(question="Compare revenue.", catalog_context=_catalog())
    annotation = get_type_hints(HybridResearchState, include_extras=True)["task_runs"]
    assert operator.add in get_args(annotation)
    assert state.task_runs == []
    assert state.plan_extensions == 0
    with pytest.raises(ValidationError, match="requires catalog candidates"):
        HybridResearchState(
            question="Compare revenue.",
            catalog_context=CatalogRoutingResponse(coverage=0, candidates=[]),
        )
