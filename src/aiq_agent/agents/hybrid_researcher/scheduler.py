# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic readiness scheduling for the cumulative Hybrid task ledger."""

from __future__ import annotations

from collections.abc import Iterable

from .models import HybridTask
from .models import HybridTaskPlan
from .models import TaskRun


def runs_by_task(task_runs: Iterable[TaskRun]) -> dict[str, TaskRun]:
    """Return the sole terminal record for each append-only task."""
    result: dict[str, TaskRun] = {}
    for run in task_runs:
        if run.task_id in result:
            raise ValueError(f"Task {run.task_id} has more than one terminal run")
        result[run.task_id] = run
    return result


def select_ready_tasks(
    plan: HybridTaskPlan,
    task_runs: Iterable[TaskRun],
    *,
    max_parallel_tasks: int,
) -> tuple[HybridTask, ...]:
    """Select pending tasks whose declared dependencies all succeeded."""
    if max_parallel_tasks < 1:
        raise ValueError("max_parallel_tasks must be positive")
    by_task = runs_by_task(task_runs)
    ready = [
        task
        for task in plan.tasks
        if task.id not in by_task
        and all(dependency in by_task and by_task[dependency].status == "succeeded" for dependency in task.depends_on)
    ]
    return tuple(ready[:max_parallel_tasks])


def derive_blocked_runs(plan: HybridTaskPlan, task_runs: Iterable[TaskRun]) -> tuple[TaskRun, ...]:
    """Mark descendants of failed or blocked dependencies as blocked."""
    by_task = runs_by_task(task_runs)
    blocked: list[TaskRun] = []
    changed = True
    while changed:
        changed = False
        for task in plan.tasks:
            if task.id in by_task:
                continue
            has_failed_dependency = any(
                dependency in by_task and by_task[dependency].status != "succeeded" for dependency in task.depends_on
            )
            if has_failed_dependency:
                run = TaskRun(
                    task_id=task.id,
                    kind=task.kind,
                    status="blocked",
                    error="A required dependency did not succeed.",
                )
                by_task[task.id] = run
                blocked.append(run)
                changed = True
    return tuple(blocked)


def ledger_exhausted(plan: HybridTaskPlan, task_runs: Iterable[TaskRun]) -> bool:
    """Return whether every task in the cumulative plan is terminal."""
    return len(runs_by_task(task_runs)) == len(plan.tasks)


__all__ = ["derive_blocked_runs", "ledger_exhausted", "runs_by_task", "select_ready_tasks"]
