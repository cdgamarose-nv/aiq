# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Initial coarse planning and append-only continuation decisions."""

from __future__ import annotations

import asyncio
import json
import logging
from collections import Counter
from collections.abc import Mapping
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from typing import TypeVar

from langchain.agents import create_agent
from langchain.agents.structured_output import StructuredOutputError
from langchain.agents.structured_output import ToolStrategy
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel
from pydantic import ValidationError

from aiq_agent.common import load_prompt
from aiq_agent.common import render_prompt_template

from .models import ContinuationDecision
from .models import HybridResearchState
from .models import HybridTask
from .models import HybridTaskPlan
from .models import ResearchTaskResult
from .models import StructuredAnalysisResult
from .models import TaskRun

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent / "prompts"
_T = TypeVar("_T", bound=BaseModel)


class PlannerError(RuntimeError):
    """Base error at a structured planner boundary."""


class PlannerConfigurationError(PlannerError):
    """The configured model cannot return the required structured output."""


class InvalidPlannerResponseError(PlannerError):
    """The model did not return the requested structured contract."""


class InvalidTaskGraphError(PlannerError):
    """A schema-valid task graph cannot be scheduled safely."""


class PlannerTimeoutError(PlannerError):
    """A planning or continuation call exceeded its deadline."""


def _normalized_objective(value: str) -> str:
    return " ".join(value.split()).casefold()


def _task_signature(task: HybridTask) -> tuple[str, str, tuple[str, ...]]:
    return (task.kind, _normalized_objective(task.objective), task.depends_on)


def validate_task_graph(
    tasks: Sequence[HybridTask],
    *,
    max_total_tasks: int,
    previous_tasks: Sequence[HybridTask] = (),
) -> None:
    """Validate identity, dependency, duplicate, cardinality, and acyclicity guards."""
    all_tasks = (*previous_tasks, *tasks)
    if not tasks:
        raise InvalidTaskGraphError("A task graph must contain at least one task.")
    if len(all_tasks) > max_total_tasks:
        raise InvalidTaskGraphError(f"Task graph contains {len(all_tasks)} tasks; maximum is {max_total_tasks}.")

    task_ids = [task.id for task in all_tasks]
    duplicate_ids = sorted(task_id for task_id, count in Counter(task_ids).items() if count > 1)
    if duplicate_ids:
        raise InvalidTaskGraphError(f"Task graph contains duplicate task IDs: {', '.join(duplicate_ids)}.")

    signatures = [_task_signature(task) for task in all_tasks]
    duplicate_signatures = [signature for signature, count in Counter(signatures).items() if count > 1]
    if duplicate_signatures:
        raise InvalidTaskGraphError("Task graph contains an exact duplicate task.")

    known_ids = set(task_ids)
    for task in all_tasks:
        duplicate_dependencies = sorted(
            dependency for dependency, count in Counter(task.depends_on).items() if count > 1
        )
        if duplicate_dependencies:
            raise InvalidTaskGraphError(
                f"Task {task.id} contains duplicate dependencies: {', '.join(duplicate_dependencies)}."
            )
        if task.id in task.depends_on:
            raise InvalidTaskGraphError(f"Task {task.id} cannot depend on itself.")
        missing = sorted(set(task.depends_on) - known_ids)
        if missing:
            raise InvalidTaskGraphError(f"Task {task.id} has missing dependencies: {', '.join(missing)}.")

    remaining = {task.id: set(task.depends_on) for task in all_tasks}
    while remaining:
        ready = {task_id for task_id, dependencies in remaining.items() if not dependencies}
        if not ready:
            raise InvalidTaskGraphError(
                f"Task graph contains a dependency cycle involving: {', '.join(sorted(remaining))}."
            )
        remaining = {
            task_id: dependencies - ready for task_id, dependencies in remaining.items() if task_id not in ready
        }


class _StructuredDecisionInvoker:
    """Shared tool-call structured-output boundary with one correction attempt."""

    def __init__(
        self,
        llm: BaseChatModel,
        schema: type[_T],
        *,
        template_name: str,
        timeout: float,
        callbacks: Sequence[BaseCallbackHandler],
    ) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self._schema = schema
        self._policy = render_prompt_template(load_prompt(_PROMPTS_DIR, template_name))
        self._timeout = timeout
        self._callbacks = tuple(callbacks)
        try:
            self._structured_agent = create_agent(
                model=llm,
                tools=[],
                system_prompt=self._policy,
                response_format=ToolStrategy(schema, handle_errors=False),
            )
        except (NotImplementedError, ValueError) as exc:
            raise PlannerConfigurationError(
                f"The configured model does not support {schema.__name__} tool-call output."
            ) from exc
        self._correction = (
            f"The structured response was missing or invalid. Call the {schema.__name__} output tool exactly once "
            "with a schema-valid payload and no prose."
        )

    def _run_config(self) -> RunnableConfig | None:
        return {"callbacks": list(self._callbacks)} if self._callbacks else None

    async def invoke(self, context: dict[str, Any]) -> _T:
        """Place stable policy first and the untrusted runtime JSON last."""
        messages = [HumanMessage(content=json.dumps(context, ensure_ascii=False, separators=(",", ":"), default=str))]
        try:
            async with asyncio.timeout(self._timeout):
                try:
                    return await self._invoke_agent(messages)
                except (
                    InvalidPlannerResponseError,
                    OutputParserException,
                    StructuredOutputError,
                    ValidationError,
                ):
                    logger.warning("Retrying %s once as a forced output tool", self._schema.__name__)
                    return await self._invoke_agent(
                        [*messages, HumanMessage(content=self._correction)],
                    )
        except TimeoutError as exc:
            raise PlannerTimeoutError(f"{self._schema.__name__} generation exceeded its timeout.") from exc

    async def _invoke_agent(self, messages: list[HumanMessage]) -> _T:
        response = await self._structured_agent.ainvoke(
            {"messages": messages},
            config=self._run_config(),
        )
        if not isinstance(response, Mapping) or response.get("structured_response") is None:
            raise InvalidPlannerResponseError("The planner returned no structured response.")
        try:
            return self._schema.model_validate(response["structured_response"])
        except ValidationError as exc:
            raise InvalidPlannerResponseError(
                f"The model did not return one valid {self._schema.__name__} object."
            ) from exc


class InitialTaskPlanner:
    """Create the smallest complete coarse task graph."""

    def __init__(
        self,
        llm: BaseChatModel,
        *,
        timeout: float = 90,
        max_total_tasks: int = 12,
        callbacks: Sequence[BaseCallbackHandler] = (),
    ) -> None:
        if max_total_tasks < 1:
            raise ValueError("max_total_tasks must be positive")
        self._invoker = _StructuredDecisionInvoker(
            llm,
            HybridTaskPlan,
            template_name="planner",
            timeout=timeout,
            callbacks=callbacks,
        )
        self._max_total_tasks = max_total_tasks

    def prompt_context(self, state: HybridResearchState) -> dict[str, Any]:
        """Project catalog semantics without treating them as observed data."""
        objective = state.clarified_question
        if objective is None:
            raise InvalidTaskGraphError("Initial planning requires a clarified objective.")
        catalog = state.catalog_context
        return {
            "objective": objective,
            "current_datetime": state.clarification_reference_datetime.isoformat(timespec="seconds"),
            "retained_proposed_defaults": state.proposed_defaults,
            "catalog_context": {
                "truncated": catalog.truncated,
                "uncovered_entities": catalog.uncovered_entities or [],
                "candidates": [
                    {"label": candidate.label, "attribute": candidate.attribute, "term": candidate.term}
                    for candidate in catalog.candidates
                ],
            },
            "available_task_kinds": ["research", "structured_analysis"],
            "max_total_tasks": self._max_total_tasks,
        }

    async def __call__(self, state: HybridResearchState) -> HybridTaskPlan:
        plan = HybridTaskPlan.model_validate(await self._invoker.invoke(self.prompt_context(state)))
        assert state.clarified_question is not None
        # The objective is supplied to the planner as context, but it is not a
        # planner decision.  Long benchmark questions can exceed the model's
        # practical tool-call argument budget when echoed back verbatim.  Keep
        # the authoritative value from state instead of rejecting a plan whose
        # redundant echo was shortened by the model.
        plan = plan.model_copy(update={"objective": state.clarified_question})
        if len(plan.tasks) == 1:
            only_task = plan.tasks[0].model_copy(update={"objective": plan.objective})
            plan = plan.model_copy(update={"tasks": (only_task,)})
        validate_task_graph(plan.tasks, max_total_tasks=self._max_total_tasks)
        return plan


def _compact_run(run: TaskRun) -> dict[str, Any]:
    item: dict[str, Any] = {
        "task_id": run.task_id,
        "kind": run.kind,
        "status": run.status,
        "attempts": run.attempts,
    }
    if run.error:
        item["error"] = run.error
    if isinstance(run.result, ResearchTaskResult):
        notes = run.result.notes
        item["conclusion"] = notes.summary
        item["limitations"] = list(notes.gaps)
        item["provenance"] = [source.locator for source in notes.sources]
    elif isinstance(run.result, StructuredAnalysisResult):
        item.update(run.result.model_dump(mode="json", exclude={"kind"}))
    return item


class ContinuationPlanner:
    """Choose finish, append, or fail after the current ledger is exhausted."""

    def __init__(
        self,
        llm: BaseChatModel,
        *,
        timeout: float = 90,
        max_total_tasks: int = 12,
        callbacks: Sequence[BaseCallbackHandler] = (),
    ) -> None:
        if max_total_tasks < 1:
            raise ValueError("max_total_tasks must be positive")
        self._invoker = _StructuredDecisionInvoker(
            llm,
            ContinuationDecision,
            template_name="continuation",
            timeout=timeout,
            callbacks=callbacks,
        )
        self._max_total_tasks = max_total_tasks

    def prompt_context(self, state: HybridResearchState, *, max_plan_extensions: int) -> dict[str, Any]:
        """Build the complete compact ledger and remaining append budget."""
        if state.plan is None or state.clarified_question is None:
            raise InvalidTaskGraphError("Continuation requires a plan and clarified objective.")
        return {
            "original_objective": state.question,
            "clarified_objective": state.clarified_question,
            "task_ledger": [
                {
                    "task": task.model_dump(mode="json"),
                    "run": _compact_run(next(run for run in state.task_runs if run.task_id == task.id)),
                }
                for task in state.plan.tasks
            ],
            "extension_waves_used": state.plan_extensions,
            "extension_waves_remaining": max(0, max_plan_extensions - state.plan_extensions),
            "task_slots_remaining": max(0, self._max_total_tasks - len(state.plan.tasks)),
        }

    async def __call__(self, state: HybridResearchState, *, max_plan_extensions: int) -> ContinuationDecision:
        decision = ContinuationDecision.model_validate(
            await self._invoker.invoke(self.prompt_context(state, max_plan_extensions=max_plan_extensions))
        )
        if decision.action == "append":
            assert state.plan is not None
            validate_task_graph(
                decision.new_tasks,
                max_total_tasks=self._max_total_tasks,
                previous_tasks=state.plan.tasks,
            )
        return decision


__all__ = [
    "ContinuationPlanner",
    "InitialTaskPlanner",
    "InvalidPlannerResponseError",
    "InvalidTaskGraphError",
    "PlannerConfigurationError",
    "PlannerError",
    "PlannerTimeoutError",
    "validate_task_graph",
]
