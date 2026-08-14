# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for deterministic Hybrid task scheduling."""

from aiq_agent.agents.hybrid_researcher.models import HybridTask
from aiq_agent.agents.hybrid_researcher.models import HybridTaskPlan
from aiq_agent.agents.hybrid_researcher.models import StructuredAnalysisResult
from aiq_agent.agents.hybrid_researcher.models import TaskRun
from aiq_agent.agents.hybrid_researcher.scheduler import derive_blocked_runs
from aiq_agent.agents.hybrid_researcher.scheduler import ledger_exhausted
from aiq_agent.agents.hybrid_researcher.scheduler import select_ready_tasks


def _plan() -> HybridTaskPlan:
    return HybridTaskPlan(
        objective="Objective.",
        tasks=(
            HybridTask(id="market", kind="research", objective="Research market."),
            HybridTask(id="revenue", kind="structured_analysis", objective="Analyze revenue."),
            HybridTask(
                id="followup",
                kind="structured_analysis",
                objective="Compare the observed revenue result.",
                depends_on=("revenue",),
            ),
        ),
    )


def _success() -> TaskRun:
    return TaskRun(
        task_id="revenue",
        kind="structured_analysis",
        status="succeeded",
        result=StructuredAnalysisResult(
            sufficiency="sufficient",
            conclusion="Revenue returned.",
            gsf_provenance=(),
        ),
    )


def test_independent_tasks_dispatch_together_and_dependents_wait():
    assert [task.id for task in select_ready_tasks(_plan(), [], max_parallel_tasks=4)] == ["market", "revenue"]
    assert [task.id for task in select_ready_tasks(_plan(), [_success()], max_parallel_tasks=4)] == [
        "market",
        "followup",
    ]


def test_failure_blocks_descendants_without_blocking_unrelated_tasks():
    failed = TaskRun(task_id="revenue", kind="structured_analysis", status="failed", error="Worker failed.")
    blocked = derive_blocked_runs(_plan(), [failed])
    assert [(run.task_id, run.status) for run in blocked] == [("followup", "blocked")]
    assert [task.id for task in select_ready_tasks(_plan(), [failed, *blocked], max_parallel_tasks=4)] == ["market"]


def test_ledger_exhaustion_requires_every_task_terminal():
    plan = _plan()
    failed = TaskRun(task_id="revenue", kind="structured_analysis", status="failed", error="Worker failed.")
    blocked = derive_blocked_runs(plan, [failed])
    assert not ledger_exhausted(plan, [failed, *blocked])
    market_failed = TaskRun(task_id="market", kind="research", status="failed", error="Research failed.")
    assert ledger_exhausted(plan, [failed, *blocked, market_failed])
