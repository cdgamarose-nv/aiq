# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LangGraph orchestration for append-only hierarchical Hybrid Research."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Awaitable
from collections.abc import Callable
from typing import Any
from typing import Literal

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Send
from pydantic import ValidationError

from aiq_agent.agents.chat_researcher.models import WorkflowClarificationRequired
from aiq_agent.agents.clarifier.utils import extract_user_response
from nat.builder.context import Context
from nat.data_models.interactive import HumanPromptText

from .clarifier import ClarificationTurnLimitError
from .clarifier import Clarifier
from .clarifier import InvalidClarificationDecisionError
from .execution import ExecutorRegistry
from .models import ClarificationDecision
from .models import ClarificationTurn
from .models import ContinuationDecision
from .models import HybridResearchState
from .models import HybridTaskPlan
from .models import TaskExecutionRequest
from .planner import validate_task_graph
from .scheduler import derive_blocked_runs
from .scheduler import ledger_exhausted
from .scheduler import runs_by_task
from .scheduler import select_ready_tasks
from .writer import HybridWriter

PromptUser = Callable[[str], Awaitable[str]]
PlanBuilder = Callable[[HybridResearchState], Awaitable[HybridTaskPlan]]
ContinuationBuilder = Callable[..., Awaitable[ContinuationDecision]]
EntryRoute = Literal["accept_original_question", "evaluate_clarity"]
AssessmentRoute = Literal["plan", "ask_clarification", "clarification_required"]
PostReplyRoute = Literal["evaluate_clarity", "plan"]
ContinuationRoute = Literal["append", "finish", "fail"]

logger = logging.getLogger(__name__)

_AFFIRMATIVE_REPLY = re.compile(
    r"\s*(?:y|yes|confirm(?:ed)?|accept(?:ed)?|looks good|that(?:'s| is) fine|"
    r"use (?:these|the) defaults|proceed)\s*[.!]?\s*",
    re.IGNORECASE,
)


class HybridResearchAgent:
    """Plan coarse work, schedule it deterministically, and append only after exhaustion."""

    def __init__(
        self,
        clarifier: Clarifier | None,
        executor_registry: ExecutorRegistry,
        *,
        enable_clarifier: bool = True,
        max_clarification_turns: int = 3,
        max_parallel_tasks: int = 4,
        max_total_tasks: int = 12,
        max_plan_extensions: int = 2,
        verbose: bool = False,
        prompt_user: PromptUser | None = None,
        plan_builder: PlanBuilder | None = None,
        continuation_builder: ContinuationBuilder | None = None,
        writer: HybridWriter | None = None,
        database_name: str | None = None,
        checkpointer: BaseCheckpointSaver | None = None,
    ) -> None:
        if not 1 <= max_clarification_turns <= 10:
            raise ValueError("max_clarification_turns must be between one and ten")
        if max_parallel_tasks < 1:
            raise ValueError("max_parallel_tasks must be positive")
        if max_total_tasks < 1:
            raise ValueError("max_total_tasks must be positive")
        if max_plan_extensions < 0:
            raise ValueError("max_plan_extensions cannot be negative")
        if enable_clarifier and clarifier is None:
            raise ValueError("clarifier is required when clarification is enabled")
        self._clarifier = clarifier
        self._executor_registry = executor_registry
        self._enable_clarifier = enable_clarifier
        self._max_clarification_turns = max_clarification_turns
        self._max_parallel_tasks = max_parallel_tasks
        self._max_total_tasks = max_total_tasks
        self._max_plan_extensions = max_plan_extensions
        self._verbose = verbose
        self._prompt_user = prompt_user
        self._plan_builder = plan_builder
        self._continuation_builder = continuation_builder
        self._writer = writer
        self._database_name = database_name
        self._checkpointer = checkpointer
        self._graph = self._build_graph()

    def _build_graph(self) -> CompiledStateGraph:
        graph = StateGraph(HybridResearchState)
        graph.add_node("accept_original_question", self.accept_original_question)
        graph.add_node("evaluate_clarity", self.evaluate_clarity)
        graph.add_node("ask_clarification", self.ask_clarification)
        graph.add_node("create_plan", self.create_plan)
        graph.add_node("advance_execution", self.advance_execution)
        graph.add_node("execute_work_item", self.execute_work_item)
        graph.add_node("decide_continuation", self.decide_continuation)
        graph.add_node("append_tasks", self.append_tasks)
        graph.add_node("write_answer", self.write_answer)
        graph.add_node("finalize_failure", self.finalize_failure)
        graph.set_conditional_entry_point(
            self.route_from_entry,
            {
                "accept_original_question": "accept_original_question",
                "evaluate_clarity": "evaluate_clarity",
            },
        )
        graph.add_edge("accept_original_question", "create_plan")
        graph.add_conditional_edges(
            "evaluate_clarity",
            self.route_after_assessment,
            {
                "plan": "create_plan",
                "ask_clarification": "ask_clarification",
                "clarification_required": END,
            },
        )
        graph.add_conditional_edges(
            "ask_clarification",
            self.route_after_clarification_reply,
            {"evaluate_clarity": "evaluate_clarity", "plan": "create_plan"},
        )
        graph.add_edge("create_plan", "advance_execution")
        graph.add_conditional_edges("advance_execution", self.dispatch_or_continue)
        graph.add_edge("execute_work_item", "advance_execution")
        graph.add_conditional_edges(
            "decide_continuation",
            self.route_after_continuation,
            {"append": "append_tasks", "finish": "write_answer", "fail": "finalize_failure"},
        )
        graph.add_edge("append_tasks", "advance_execution")
        graph.add_edge("write_answer", END)
        graph.add_edge("finalize_failure", END)
        return graph.compile(checkpointer=self._checkpointer)

    def route_from_entry(self, _state: HybridResearchState) -> EntryRoute:
        return "evaluate_clarity" if self._enable_clarifier else "accept_original_question"

    @staticmethod
    def accept_original_question(state: HybridResearchState) -> dict[str, Any]:
        return {
            "clarification_decision": None,
            "clarified_question": state.question,
            "clarification_required": None,
        }

    async def evaluate_clarity(self, state: HybridResearchState) -> dict[str, Any]:
        if self._clarifier is None:
            raise RuntimeError("Clarifier is not configured")
        try:
            decision = ClarificationDecision.model_validate(await self._clarifier(state))
        except InvalidClarificationDecisionError:
            raise
        except (TypeError, ValueError, ValidationError) as exc:
            raise InvalidClarificationDecisionError("The clarifier model returned an invalid decision.") from exc
        update: dict[str, Any] = {
            "clarification_decision": decision,
            "clarified_question": decision.clarified_question,
            "clarification_required": None,
        }
        if decision.status == "ready":
            return update
        if len(state.clarification_history) >= self._max_clarification_turns:
            raise ClarificationTurnLimitError(self._max_clarification_turns, decision)
        if state.skip_clarifier:
            assert decision.clarification_question is not None
            update["clarification_required"] = WorkflowClarificationRequired(
                clarification_question=decision.clarification_question,
                missing_dimensions=decision.missing_dimensions or ("business_context",),
            )
        return update

    @staticmethod
    def route_after_assessment(state: HybridResearchState) -> AssessmentRoute:
        if state.clarified_question is not None:
            return "plan"
        if state.clarification_required is not None:
            return "clarification_required"
        return "ask_clarification"

    @staticmethod
    def route_after_clarification_reply(state: HybridResearchState) -> PostReplyRoute:
        return "plan" if state.clarified_question is not None else "evaluate_clarity"

    async def ask_clarification(self, state: HybridResearchState) -> dict[str, Any]:
        decision = state.clarification_decision
        if decision is None or decision.status != "needs_clarification" or decision.clarification_question is None:
            raise ValueError("ask_clarification requires a persisted clarification decision")
        if self._prompt_user is not None:
            reply = await self._prompt_user(decision.clarification_question)
        else:
            response = await Context.get().user_interaction_manager.prompt_user_input(
                HumanPromptText(
                    text=decision.clarification_question,
                    required=True,
                    placeholder="Please provide the missing analysis details...",
                )
            )
            reply = extract_user_response(response)
        is_confirmation = bool(decision.proposed_defaults) and _AFFIRMATIVE_REPLY.fullmatch(reply) is not None
        turn = ClarificationTurn(
            clarification_question=decision.clarification_question,
            user_reply=reply,
            proposed_defaults=decision.proposed_defaults,
        )
        retained_defaults = {**state.proposed_defaults, **decision.proposed_defaults}
        update: dict[str, Any] = {
            "clarification_history": (*state.clarification_history, turn),
            "proposed_defaults": retained_defaults,
            "clarification_decision": None,
            "clarified_question": None,
            "clarification_required": None,
        }
        if is_confirmation:
            clarified = _question_with_confirmed_defaults(state.question, retained_defaults)
            update["clarification_decision"] = ClarificationDecision(
                status="ready",
                clarified_question=clarified,
            )
            update["clarified_question"] = clarified
        return update

    async def create_plan(self, state: HybridResearchState) -> dict[str, Any]:
        if state.clarified_question is None:
            raise ValueError("create_plan requires a clarified objective")
        if self._plan_builder is None:
            raise RuntimeError("Hybrid Research planning is not configured")
        plan = HybridTaskPlan.model_validate(await self._plan_builder(state))
        if plan.objective != state.clarified_question:
            raise ValueError("Initial plan objective must exactly match the clarified objective")
        validate_task_graph(plan.tasks, max_total_tasks=self._max_total_tasks)
        if self._verbose:
            logger.info(
                "Hybrid initial plan created (tasks=%s)",
                [{"id": task.id, "kind": task.kind, "depends_on": list(task.depends_on)} for task in plan.tasks],
            )
        return {"plan": plan}

    def advance_execution(self, state: HybridResearchState) -> dict[str, Any]:
        if state.plan is None:
            raise ValueError("advance_execution requires a plan")
        blocked = derive_blocked_runs(state.plan, state.task_runs)
        return {"task_runs": list(blocked)} if blocked else {}

    def dispatch_or_continue(self, state: HybridResearchState) -> str | list[Send]:
        plan = state.plan
        if plan is None:
            raise ValueError("dispatch_or_continue requires a plan")
        ready = select_ready_tasks(plan, state.task_runs, max_parallel_tasks=self._max_parallel_tasks)
        if ready:
            by_task = runs_by_task(state.task_runs)
            if self._verbose:
                logger.info("Hybrid task batch dispatched (tasks=%s)", [f"{task.id}:{task.kind}" for task in ready])
            return [
                Send(
                    "execute_work_item",
                    TaskExecutionRequest(
                        task=task,
                        objective=plan.objective,
                        catalog_context=state.catalog_context,
                        dependency_runs=tuple(by_task[dependency] for dependency in task.depends_on),
                        data_sources=state.data_sources,
                        database_name=self._database_name,
                    ).model_dump(mode="python"),
                )
                for task in ready
            ]
        if ledger_exhausted(plan, state.task_runs):
            return "decide_continuation"
        logger.warning("Hybrid task ledger cannot make progress")
        return "finalize_failure"

    async def execute_work_item(self, payload: TaskExecutionRequest | dict[str, Any]) -> dict[str, Any]:
        request = TaskExecutionRequest.model_validate(payload)
        with Context.get().push_active_function(
            f"hybrid_task_{request.task.id}",
            json.dumps({"type": "hybrid.task", "phase": "started", "task": request.task.model_dump(mode="json")}),
        ) as event:
            run = await self._executor_registry.execute(request)
            event.set_output(
                json.dumps(
                    {
                        "type": "hybrid.task",
                        "phase": "completed",
                        "run": run.model_dump(mode="json"),
                    }
                )
            )
        return {"task_runs": [run]}

    async def decide_continuation(self, state: HybridResearchState) -> dict[str, Any]:
        if self._continuation_builder is None:
            raise RuntimeError("Hybrid Research continuation planning is not configured")
        decision = ContinuationDecision.model_validate(
            await self._continuation_builder(state, max_plan_extensions=self._max_plan_extensions)
        )
        if decision.action == "append":
            if state.plan is None:
                raise ValueError("An append decision requires the current plan")
            validate_task_graph(
                decision.new_tasks,
                max_total_tasks=self._max_total_tasks,
                previous_tasks=state.plan.tasks,
            )
        if self._verbose:
            logger.info(
                "Hybrid continuation decided (action=%s new_tasks=%s)",
                decision.action,
                len(decision.new_tasks),
            )
        return {"continuation_history": (*state.continuation_history, decision)}

    def route_after_continuation(self, state: HybridResearchState) -> ContinuationRoute:
        if not state.continuation_history:
            raise ValueError("Continuation routing requires a decision")
        action = state.continuation_history[-1].action
        if action == "append" and state.plan_extensions >= self._max_plan_extensions:
            return "fail"
        return action

    @staticmethod
    def append_tasks(state: HybridResearchState) -> dict[str, Any]:
        if state.plan is None or not state.continuation_history:
            raise ValueError("append_tasks requires a plan and continuation decision")
        decision = state.continuation_history[-1]
        if decision.action != "append":
            raise ValueError("append_tasks requires an append decision")
        return {
            "plan": state.plan.model_copy(update={"tasks": (*state.plan.tasks, *decision.new_tasks)}),
            "plan_extensions": state.plan_extensions + 1,
        }

    async def write_answer(self, state: HybridResearchState) -> dict[str, Any]:
        if self._writer is None:
            raise RuntimeError("Hybrid Research writer is not configured")
        return {"final_answer": await self._writer(state), "terminal_message": None}

    def finalize_failure(self, _state: HybridResearchState) -> dict[str, Any]:
        return {
            "terminal_message": "Hybrid Research could not complete this request responsibly. Please try again.",
            "final_answer": None,
        }

    async def run(self, state: HybridResearchState, *, thread_id: str | None = None) -> HybridResearchState:
        if self._checkpointer is not None and thread_id is None:
            raise ValueError("thread_id is required when Hybrid Research uses a checkpointer")
        config: RunnableConfig | None = None
        if thread_id is not None:
            config = {"configurable": {"thread_id": thread_id}}
        return HybridResearchState.model_validate(await self._graph.ainvoke(state, config=config))

    @property
    def graph(self) -> CompiledStateGraph:
        return self._graph


def _question_with_confirmed_defaults(question: str, defaults: dict[str, Any]) -> str:
    serialized = json.dumps(defaults, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return f"{question.rstrip()}\n\nConfirmed business defaults: {serialized}"


__all__ = ["ContinuationBuilder", "HybridResearchAgent", "PlanBuilder", "PromptUser"]
