# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Graph tests for hierarchical Hybrid Research orchestration."""

import asyncio

from aiq_agent.agents.chat_researcher.models import CatalogCandidate
from aiq_agent.agents.chat_researcher.models import CatalogRoutingResponse
from aiq_agent.agents.deep_researcher.models import ResearchNotes
from aiq_agent.agents.hybrid_researcher.agent import HybridResearchAgent
from aiq_agent.agents.hybrid_researcher.execution import ExecutorRegistry
from aiq_agent.agents.hybrid_researcher.models import ContinuationDecision
from aiq_agent.agents.hybrid_researcher.models import HybridResearchState
from aiq_agent.agents.hybrid_researcher.models import HybridTask
from aiq_agent.agents.hybrid_researcher.models import HybridTaskPlan
from aiq_agent.agents.hybrid_researcher.models import ResearchTaskResult
from aiq_agent.agents.hybrid_researcher.models import StructuredAnalysisResult


def _state() -> HybridResearchState:
    return HybridResearchState(
        question="Compare revenue with market context.",
        catalog_context=CatalogRoutingResponse(
            coverage=1,
            candidates=[CatalogCandidate(id="metric", label="Metric", attribute="recognized", term="Revenue")],
        ),
    )


def _notes(value: str) -> ResearchNotes:
    return ResearchNotes(
        query_topic=value,
        target_components=["context"],
        summary=value,
        findings=[],
        gaps=[],
        sources=[],
        narrative_notes=value,
        language="English",
    )


async def test_parallel_tasks_dependency_levels_and_single_terminal_continuation():
    roots_started = set()
    roots_ready = asyncio.Event()
    events = []

    class Executor:
        async def execute(self, request):
            events.append(f"start:{request.task.id}")
            if request.task.id in {"market", "revenue"}:
                roots_started.add(request.task.id)
                if len(roots_started) == 2:
                    roots_ready.set()
                await asyncio.wait_for(roots_ready.wait(), 1)
            events.append(f"finish:{request.task.id}")
            if request.task.kind == "research":
                return ResearchTaskResult(notes=_notes(request.task.id))
            return StructuredAnalysisResult(
                sufficiency="sufficient",
                conclusion=request.task.id,
                gsf_provenance=(),
            )

    plan = HybridTaskPlan(
        objective=_state().question,
        tasks=(
            HybridTask(id="market", kind="research", objective="Research market."),
            HybridTask(id="revenue", kind="structured_analysis", objective="Analyze revenue."),
            HybridTask(
                id="comparison",
                kind="structured_analysis",
                objective="Compare the observed results.",
                depends_on=("market", "revenue"),
            ),
        ),
    )
    continuation_calls = 0

    async def continuation(_state, **_kwargs):
        nonlocal continuation_calls
        continuation_calls += 1
        return ContinuationDecision(action="finish", reasoning="Evidence covers the objective.")

    async def writer(_state):
        return "Complete answer."

    executor = Executor()
    agent = HybridResearchAgent(
        None,
        ExecutorRegistry({"research": executor, "structured_analysis": executor}),
        enable_clarifier=False,
        plan_builder=lambda _state: asyncio.sleep(0, result=plan),
        continuation_builder=continuation,
        writer=writer,
    )
    result = await agent.run(_state())
    assert events.index("start:market") < events.index("finish:revenue")
    assert events.index("start:revenue") < events.index("finish:market")
    assert events.index("start:comparison") > events.index("finish:market")
    assert events.index("start:comparison") > events.index("finish:revenue")
    assert continuation_calls == 1
    assert result.final_answer == "Complete answer."


async def test_result_driven_append_preserves_existing_runs_and_two_waves():
    calls = []

    class Executor:
        async def execute(self, request):
            calls.append(request.task.id)
            if request.task.kind == "research":
                return ResearchTaskResult(notes=_notes(request.task.id))
            return StructuredAnalysisResult(
                sufficiency="limited",
                conclusion="Entities Alpha and Beta.",
                gsf_provenance=(),
            )

    initial = HybridTaskPlan(
        objective=_state().question,
        tasks=(HybridTask(id="discover", kind="structured_analysis", objective="Discover entities."),),
    )
    decisions = iter(
        [
            ContinuationDecision(
                action="append",
                new_tasks=(
                    HybridTask(id="alpha", kind="research", objective="Research Alpha.", depends_on=("discover",)),
                ),
                reasoning="Alpha needs context.",
            ),
            ContinuationDecision(
                action="append",
                new_tasks=(
                    HybridTask(id="beta", kind="research", objective="Research Beta.", depends_on=("discover",)),
                ),
                reasoning="Beta also needs context.",
            ),
            ContinuationDecision(action="finish", reasoning="Both entities are covered."),
        ]
    )

    async def continuation(_state, **_kwargs):
        return next(decisions)

    agent = HybridResearchAgent(
        None,
        ExecutorRegistry({"research": Executor(), "structured_analysis": Executor()}),
        enable_clarifier=False,
        plan_builder=lambda _state: asyncio.sleep(0, result=initial),
        continuation_builder=continuation,
        writer=lambda _state: asyncio.sleep(0, result="Done."),
        max_plan_extensions=2,
    )
    result = await agent.run(_state())
    assert calls == ["discover", "alpha", "beta"]
    assert [task.id for task in result.plan.tasks] == ["discover", "alpha", "beta"]
    assert result.plan_extensions == 2
    assert result.final_answer == "Done."


async def test_third_append_fails_closed_and_writer_never_runs():
    initial = HybridTaskPlan(
        objective=_state().question,
        tasks=(HybridTask(id="discover", kind="structured_analysis", objective="Discover entities."),),
    )

    class Executor:
        async def execute(self, request):
            if request.task.kind == "research":
                return ResearchTaskResult(notes=_notes(request.task.id))
            return StructuredAnalysisResult(sufficiency="limited", conclusion="Entity.", gsf_provenance=())

    index = 0

    async def continuation(_state, **_kwargs):
        nonlocal index
        index += 1
        return ContinuationDecision(
            action="append",
            new_tasks=(
                HybridTask(
                    id=f"wave_{index}",
                    kind="research",
                    objective=f"Research entity wave {index}.",
                    depends_on=("discover",),
                ),
            ),
            reasoning="More context.",
        )

    writer_called = False

    async def writer(_state):
        nonlocal writer_called
        writer_called = True
        return "Should not run."

    executor = Executor()
    agent = HybridResearchAgent(
        None,
        ExecutorRegistry({"research": executor, "structured_analysis": executor}),
        enable_clarifier=False,
        plan_builder=lambda _state: asyncio.sleep(0, result=initial),
        continuation_builder=continuation,
        writer=writer,
        max_plan_extensions=2,
    )
    result = await agent.run(_state())
    assert result.plan_extensions == 2
    assert result.terminal_message is not None
    assert not writer_called


async def test_failed_dependency_blocks_descendant_while_unrelated_task_completes():
    called = []

    class Executor:
        async def execute(self, request):
            called.append(request.task.id)
            if request.task.id == "revenue":
                raise RuntimeError("failed")
            if request.task.kind == "research":
                return ResearchTaskResult(notes=_notes("market"))
            raise AssertionError("blocked descendant ran")

    plan = HybridTaskPlan(
        objective=_state().question,
        tasks=(
            HybridTask(id="revenue", kind="structured_analysis", objective="Analyze revenue."),
            HybridTask(id="market", kind="research", objective="Research market."),
            HybridTask(
                id="dependent",
                kind="structured_analysis",
                objective="Analyze revenue comparison.",
                depends_on=("revenue",),
            ),
        ),
    )
    executor = Executor()
    agent = HybridResearchAgent(
        None,
        ExecutorRegistry({"research": executor, "structured_analysis": executor}),
        enable_clarifier=False,
        plan_builder=lambda _state: asyncio.sleep(0, result=plan),
        continuation_builder=lambda _state, **_kwargs: asyncio.sleep(
            0,
            result=ContinuationDecision(action="fail", reasoning="Required structured evidence failed."),
        ),
    )
    result = await agent.run(_state())
    assert set(called) == {"revenue", "market"}
    assert [(run.task_id, run.status) for run in result.task_runs] == [
        ("revenue", "failed"),
        ("market", "succeeded"),
        ("dependent", "blocked"),
    ]
