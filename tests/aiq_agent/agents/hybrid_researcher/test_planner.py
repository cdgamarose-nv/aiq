# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for coarse planning and append-only continuation."""

from datetime import UTC
from datetime import datetime
from typing import Any

import pytest

from aiq_agent.agents.chat_researcher.models import CatalogCandidate
from aiq_agent.agents.chat_researcher.models import CatalogRoutingResponse
from aiq_agent.agents.hybrid_researcher.models import BoundedTableEvidence
from aiq_agent.agents.hybrid_researcher.models import ContinuationDecision
from aiq_agent.agents.hybrid_researcher.models import GSFQuerySuccess
from aiq_agent.agents.hybrid_researcher.models import GSFResultColumnSummary
from aiq_agent.agents.hybrid_researcher.models import HybridResearchState
from aiq_agent.agents.hybrid_researcher.models import HybridTask
from aiq_agent.agents.hybrid_researcher.models import HybridTaskPlan
from aiq_agent.agents.hybrid_researcher.models import StructuredAnalysisResult
from aiq_agent.agents.hybrid_researcher.models import TaskRun
from aiq_agent.agents.hybrid_researcher.planner import ContinuationPlanner
from aiq_agent.agents.hybrid_researcher.planner import InitialTaskPlanner
from aiq_agent.agents.hybrid_researcher.planner import InvalidTaskGraphError
from aiq_agent.agents.hybrid_researcher.planner import validate_task_graph


class _Runnable:
    def __init__(self, responses: list[Any]) -> None:
        self.responses = iter(responses)
        self.calls = []

    async def ainvoke(self, state, config=None):
        self.calls.append((state, config))
        return {"structured_response": next(self.responses)}


class _Model:
    def __init__(self, responses: list[Any]) -> None:
        self.runnable = _Runnable(responses)
        self.schema = None
        self.response_format = None


@pytest.fixture(autouse=True)
def _tool_call_planner_agent(monkeypatch):
    def fake_create_agent(*, model, response_format, **_kwargs):
        model.schema = response_format.schema
        model.response_format = response_format
        return model.runnable

    monkeypatch.setattr("aiq_agent.agents.hybrid_researcher.planner.create_agent", fake_create_agent)


def _state(**overrides: Any) -> HybridResearchState:
    values = {
        "question": "Find the largest decline and research its period.",
        "clarified_question": "Find the largest decline and research its period.",
        "clarification_reference_datetime": datetime(2026, 8, 13, tzinfo=UTC),
        "catalog_context": CatalogRoutingResponse(
            request_id="private",
            coverage=1,
            candidates=[CatalogCandidate(id="private-id", label="Metric", attribute="recognized", term="Revenue")],
        ),
        "user_info": {"email": "private@example.com"},
    }
    values.update(overrides)
    return HybridResearchState(**values)


async def test_initial_planner_allows_parallel_multiple_kinds_and_known_dependency():
    plan = HybridTaskPlan(
        objective="Find the largest decline and research its period.",
        tasks=(
            HybridTask(id="decline", kind="structured_analysis", objective="Identify the largest decline."),
            HybridTask(id="benchmark", kind="structured_analysis", objective="Analyze the independent benchmark."),
            HybridTask(id="policy", kind="research", objective="Research relevant policy changes."),
            HybridTask(
                id="decline_period",
                kind="research",
                objective="Research the period returned by the decline analysis.",
                depends_on=("decline",),
            ),
        ),
    )
    model = _Model([plan])
    planner = InitialTaskPlanner(model)
    assert await planner(_state()) == plan
    context = planner.prompt_context(_state())
    serialized = str(context)
    assert "private@example.com" not in serialized
    assert "private-id" not in serialized
    assert "request_id" not in serialized
    assert context["catalog_context"]["candidates"][0]["term"] == "Revenue"
    assert model.schema is HybridTaskPlan
    assert model.response_format.handle_errors is False


async def test_single_task_keeps_clarified_objective_without_procedural_drift():
    state = _state()
    proposed = HybridTaskPlan(
        objective=state.clarified_question,
        tasks=(
            HybridTask(
                id="decline",
                kind="structured_analysis",
                objective="Fetch every intermediate row, then calculate the decline.",
            ),
        ),
    )
    plan = await InitialTaskPlanner(_Model([proposed]))(state)
    assert plan.tasks[0].objective == state.clarified_question


def test_graph_validation_rejects_exact_duplicates_cycles_missing_dependencies_and_total_limit():
    first = HybridTask(id="first", kind="research", objective="Research context.")
    with pytest.raises(InvalidTaskGraphError, match="exact duplicate"):
        validate_task_graph(
            (first, HybridTask(id="second", kind="research", objective="  research CONTEXT.  ")),
            max_total_tasks=12,
        )
    with pytest.raises(InvalidTaskGraphError, match="missing dependencies"):
        validate_task_graph(
            (HybridTask(id="first", kind="research", objective="Research.", depends_on=("missing",)),),
            max_total_tasks=12,
        )
    with pytest.raises(InvalidTaskGraphError, match="cycle"):
        validate_task_graph(
            (
                HybridTask(id="first", kind="research", objective="One.", depends_on=("second",)),
                HybridTask(id="second", kind="structured_analysis", objective="Two.", depends_on=("first",)),
            ),
            max_total_tasks=12,
        )
    with pytest.raises(InvalidTaskGraphError, match="maximum"):
        validate_task_graph(
            (first,), previous_tasks=(HybridTask(id="old", kind="research", objective="Old."),), max_total_tasks=1
        )


async def test_continuation_observes_exhausted_ledger_and_appends_unknown_fanout():
    plan = HybridTaskPlan(
        objective="Find the largest decline and research its period.",
        tasks=(HybridTask(id="discover", kind="structured_analysis", objective="Discover entities."),),
    )
    run = TaskRun(
        task_id="discover",
        kind="structured_analysis",
        status="succeeded",
        result=StructuredAnalysisResult(
            sufficiency="limited",
            conclusion="Entities Alpha and Beta were discovered.",
            gsf_provenance=(),
            limitations=("External context is missing.",),
        ),
    )
    decision = ContinuationDecision(
        action="append",
        new_tasks=(
            HybridTask(id="alpha", kind="research", objective="Research Alpha.", depends_on=("discover",)),
            HybridTask(id="beta", kind="research", objective="Research Beta.", depends_on=("discover",)),
        ),
        reasoning="The discovered entities need external evidence.",
    )
    state = _state(plan=plan, task_runs=[run])
    model = _Model([decision])
    planner = ContinuationPlanner(model)
    assert await planner(state, max_plan_extensions=2) == decision
    context = planner.prompt_context(state, max_plan_extensions=2)
    assert context["task_ledger"][0]["run"]["conclusion"] == run.result.conclusion
    assert context["extension_waves_remaining"] == 2
    assert model.schema is ContinuationDecision


def test_continuation_receives_compact_terminal_structured_evidence():
    plan = HybridTaskPlan(
        objective="Find the largest decline and research its period.",
        tasks=(HybridTask(id="discover", kind="structured_analysis", objective="Discover entities."),),
    )
    run = TaskRun(
        task_id="discover",
        kind="structured_analysis",
        status="succeeded",
        result=StructuredAnalysisResult(
            sufficiency="sufficient",
            conclusion="Entities Alpha and Beta were discovered.",
            gsf_provenance=(
                GSFQuerySuccess(
                    question="Discover entities.",
                    request_id="entities-1",
                    citation_key="GSF request entities-1",
                    returned_row_count=1_000,
                    result_truncated=False,
                ),
            ),
        ),
    )
    context = ContinuationPlanner(_Model([])).prompt_context(
        _state(plan=plan, task_runs=[run]),
        max_plan_extensions=2,
    )
    projection = context["task_ledger"][0]["run"]
    assert projection["conclusion"] == "Entities Alpha and Beta were discovered."
    assert projection["gsf_provenance"][0]["returned_row_count"] == 1_000
    assert "rows" not in str(projection)
    assert "artifact" not in str(projection)


def test_continuation_receives_exact_rows_only_through_bounded_table_contract():
    plan = HybridTaskPlan(
        objective="Find the largest decline and research its period.",
        tasks=(HybridTask(id="discover", kind="structured_analysis", objective="Discover entities."),),
    )
    run = TaskRun(
        task_id="discover",
        kind="structured_analysis",
        status="succeeded",
        result=StructuredAnalysisResult(
            sufficiency="sufficient",
            conclusion="Two entities were returned.",
            gsf_provenance=(),
            table_evidence=BoundedTableEvidence(
                citation_key="GSF request entities-1",
                columns=(GSFResultColumnSummary(name="entity"),),
                rows=({"entity": "Alpha"}, {"entity": "Beta"}),
                returned_row_count=2,
            ),
        ),
    )
    context = ContinuationPlanner(_Model([])).prompt_context(
        _state(plan=plan, task_runs=[run]),
        max_plan_extensions=2,
    )
    projection = context["task_ledger"][0]["run"]
    assert projection["table_evidence"]["rows"] == [{"entity": "Alpha"}, {"entity": "Beta"}]
    assert "sql" not in str(projection)


async def test_continuation_rejects_mutating_or_duplicate_existing_task():
    existing = HybridTask(id="existing", kind="research", objective="Research context.")
    plan = HybridTaskPlan(objective=_state().clarified_question, tasks=(existing,))
    run = TaskRun(task_id="existing", kind="research", status="failed", error="Research failed.")
    duplicate = ContinuationDecision(
        action="append",
        new_tasks=(HybridTask(id="renamed", kind="research", objective="Research context."),),
        reasoning="Retry it.",
    )
    planner = ContinuationPlanner(_Model([duplicate]))
    with pytest.raises(InvalidTaskGraphError, match="exact duplicate"):
        await planner(_state(plan=plan, task_runs=[run]), max_plan_extensions=2)
