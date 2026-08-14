# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for Hybrid Research clarification and graph routing."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import UTC
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage
from pydantic import ValidationError

from aiq_agent.agents.chat_researcher.models import CatalogCandidate
from aiq_agent.agents.chat_researcher.models import CatalogRoutingResponse
from aiq_agent.agents.chat_researcher.models import WorkflowClarificationRequired
from aiq_agent.agents.hybrid_researcher.agent import HybridResearchAgent
from aiq_agent.agents.hybrid_researcher.clarifier import ClarificationTurnLimitError
from aiq_agent.agents.hybrid_researcher.clarifier import ClarifierConfigurationError
from aiq_agent.agents.hybrid_researcher.clarifier import ClarifierTimeoutError
from aiq_agent.agents.hybrid_researcher.clarifier import InvalidClarificationDecisionError
from aiq_agent.agents.hybrid_researcher.clarifier import StructuredClarifier
from aiq_agent.agents.hybrid_researcher.execution import ExecutorRegistry
from aiq_agent.agents.hybrid_researcher.models import ClarificationDecision
from aiq_agent.agents.hybrid_researcher.models import ContinuationDecision
from aiq_agent.agents.hybrid_researcher.models import HybridResearchState
from aiq_agent.agents.hybrid_researcher.models import HybridTask
from aiq_agent.agents.hybrid_researcher.models import HybridTaskPlan
from aiq_agent.agents.hybrid_researcher.models import StructuredAnalysisResult
from nat.builder.context import Context


def _catalog_context(*, request_id: str = "catalog-request-1") -> CatalogRoutingResponse:
    return CatalogRoutingResponse(
        request_id=request_id,
        coverage=1.0,
        candidates=[
            CatalogCandidate(
                id="metric.revenue",
                label="Metric",
                attribute="recognized_revenue",
                term="Revenue",
            )
        ],
    )


def _state(**overrides: Any) -> HybridResearchState:
    values = {
        "question": "Compare revenue with market conditions.",
        "catalog_context": _catalog_context(),
    }
    values.update(overrides)
    return HybridResearchState(**values)


def _needs_clarification() -> ClarificationDecision:
    return ClarificationDecision.model_validate(
        {
            "status": "needs_clarification",
            "missing_dimensions": ["time_window"],
            "clarification_question": "Which reporting period should I use?",
            "clarified_question": None,
            "proposed_defaults": {"time_window": "calendar year 2026"},
        }
    )


def _ready() -> ClarificationDecision:
    return ClarificationDecision.model_validate(
        {
            "status": "ready",
            "missing_dimensions": [],
            "clarification_question": None,
            "clarified_question": "Compare 2026 revenue with external market conditions.",
            "proposed_defaults": {},
        }
    )


async def _test_plan(state: HybridResearchState) -> HybridTaskPlan:
    return HybridTaskPlan(
        objective=state.clarified_question or state.question,
        tasks=(HybridTask(id="revenue", kind="structured_analysis", objective="Query the objective."),),
    )


def _agent(clarifier, **kwargs) -> HybridResearchAgent:
    class Executor:
        async def execute(self, _request):
            return StructuredAnalysisResult(
                sufficiency="sufficient",
                conclusion="Revenue result.",
                gsf_provenance=(),
            )

    async def continuation(_state, **_kwargs):
        return ContinuationDecision(action="finish", reasoning="The result is sufficient.")

    async def writer(_state):
        return "Revenue result."

    kwargs.setdefault("plan_builder", _test_plan)
    kwargs.setdefault("continuation_builder", continuation)
    kwargs.setdefault("writer", writer)
    return HybridResearchAgent(clarifier, ExecutorRegistry({"structured_analysis": Executor()}), **kwargs)


class _SequenceClarifier:
    def __init__(
        self,
        decisions: list[ClarificationDecision | dict | Callable[[HybridResearchState], ClarificationDecision]],
    ) -> None:
        self._decisions = iter(decisions)
        self.states: list[HybridResearchState] = []

    async def __call__(self, state: HybridResearchState) -> ClarificationDecision | dict:
        self.states.append(state)
        decision = next(self._decisions)
        return decision(state) if callable(decision) else decision


class _FakeRunnable:
    def __init__(self, responses: list[Any], *, delay: float = 0) -> None:
        self._responses = iter(responses)
        self.delay = delay
        self.calls: list[tuple[Any, Any]] = []

    async def ainvoke(self, messages, config=None):
        self.calls.append((messages, config))
        if self.delay:
            await asyncio.sleep(self.delay)
        response = next(self._responses)
        if isinstance(response, Exception):
            raise response
        return response


class _StructuredCapableModel:
    def __init__(self, structured_responses: list[Any], raw_responses: list[Any] | None = None) -> None:
        self.structured = _FakeRunnable(structured_responses)
        self._raw_responses = iter(raw_responses or [])
        self.raw_calls: list[tuple[Any, Any]] = []
        self.schema = None

    def with_structured_output(self, schema):
        self.schema = schema
        return self.structured

    async def ainvoke(self, messages, config=None):
        self.raw_calls.append((messages, config))
        response = next(self._raw_responses)
        if isinstance(response, Exception):
            raise response
        return response


@pytest.mark.parametrize(
    "payload",
    [
        {"status": "ready"},
        {"status": "needs_clarification"},
    ],
)
def test_clarification_decision_requires_only_the_selected_route_output(payload):
    with pytest.raises(ValidationError):
        ClarificationDecision.model_validate(payload)


def test_reference_datetime_must_be_timezone_aware():
    with pytest.raises(ValidationError, match="timezone-aware"):
        _state(clarification_reference_datetime=datetime(2026, 8, 6, 12, 0))


def test_clarification_schema_keeps_non_route_fields_optional():
    schema = ClarificationDecision.model_json_schema()
    assert schema["required"] == ["status"]


def test_ready_decision_tolerates_unused_clarification_fields():
    decision = ClarificationDecision.model_validate(
        {
            "status": "ready",
            "missing_dimensions": [],
            "clarification_question": "Repeated clarified question",
            "clarified_question": "Return revenue.",
            "proposed_defaults": {"currency": "USD"},
        }
    )

    assert decision.status == "ready"
    assert decision.clarified_question == "Return revenue."


async def test_clarifier_loop_is_part_of_the_hybrid_graph():
    def finish(state: HybridResearchState) -> ClarificationDecision:
        assert state.clarification_history[0].user_reply == "Use calendar year 2026."
        assert state.proposed_defaults == {"time_window": "calendar year 2026"}
        return _ready()

    clarifier = _SequenceClarifier([_needs_clarification(), finish])
    prompts: list[str] = []

    async def prompt_user(question: str) -> str:
        prompts.append(question)
        return "Use calendar year 2026."

    result = await _agent(clarifier, prompt_user=prompt_user).run(_state())

    assert prompts == ["Which reporting period should I use?"]
    assert result.clarified_question == "Compare 2026 revenue with external market conditions."
    assert len(result.clarification_history) == 1


@pytest.mark.parametrize("reply", ["yes", "Confirm.", "use these defaults", "Looks good!"])
async def test_exact_confirmation_proceeds_without_another_clarifier_call(reply: str):
    clarifier = _SequenceClarifier([_needs_clarification()])

    async def prompt_user(_: str) -> str:
        return reply

    result = await _agent(clarifier, prompt_user=prompt_user).run(_state())

    assert len(clarifier.states) == 1
    assert len(result.clarification_history) == 1
    assert result.proposed_defaults == {"time_window": "calendar year 2026"}
    assert result.clarified_question == (
        'Compare revenue with market conditions.\n\nConfirmed business defaults: {"time_window":"calendar year 2026"}'
    )


async def test_confirmation_with_a_correction_is_reassessed_by_the_clarifier():
    clarifier = _SequenceClarifier([_needs_clarification(), _ready()])

    async def prompt_user(_: str) -> str:
        return "Yes, but use fiscal year 2026."

    result = await _agent(clarifier, prompt_user=prompt_user).run(_state())

    assert len(clarifier.states) == 2
    assert result.clarified_question == "Compare 2026 revenue with external market conditions."


async def test_disabled_clarifier_uses_original_question_without_assessment_or_prompt():
    async def unexpected_prompt(_: str) -> str:
        raise AssertionError("disabled clarifier must not prompt")

    result = await _agent(
        None,
        enable_clarifier=False,
        prompt_user=unexpected_prompt,
    ).run(_state())

    assert result.clarified_question == "Compare revenue with market conditions."
    assert result.clarification_decision is None
    assert result.clarification_history == ()


async def test_headless_ambiguity_returns_public_clarification_outcome_without_prompting():
    async def unexpected_prompt(_: str) -> str:
        raise AssertionError("headless clarification must not prompt")

    result = await _agent(
        _SequenceClarifier([_needs_clarification()]),
        prompt_user=unexpected_prompt,
    ).run(_state(skip_clarifier=True))

    assert result.clarified_question is None
    assert result.clarification_required == WorkflowClarificationRequired(
        clarification_question="Which reporting period should I use?",
        missing_dimensions=("time_window",),
    )


async def test_turn_limit_stops_before_planning():
    async def prompt_user(_: str) -> str:
        return "I am not sure."

    agent = _agent(
        _SequenceClarifier([_needs_clarification(), _needs_clarification()]),
        max_clarification_turns=1,
        prompt_user=prompt_user,
    )

    with pytest.raises(ClarificationTurnLimitError, match="1 clarification turn"):
        await agent.run(_state())


async def test_invalid_clarifier_payload_fails_closed():
    agent = _agent(_SequenceClarifier([{"status": "ready", "clarified_question": None}]))

    with pytest.raises(InvalidClarificationDecisionError, match="invalid decision"):
        await agent.run(_state())


def test_structured_clarifier_sends_only_decision_useful_context():
    model = _StructuredCapableModel([_ready()])
    clarifier = StructuredClarifier(model)
    state = _state(
        clarification_reference_datetime=datetime(2026, 8, 6, 12, 0, tzinfo=UTC),
        user_info={"email": "private@example.com"},
    )

    context = clarifier.prompt_context(state)
    dynamic_message = clarifier.messages(state)[1].content

    assert context["current_datetime"] == "2026-08-06T12:00:00+00:00"
    assert context["catalog_context"]["candidates"] == [
        {"label": "Metric", "attribute": "recognized_revenue", "term": "Revenue"}
    ]
    assert "user_info" not in dynamic_message
    assert "coverage" not in dynamic_message
    assert '"id"' not in dynamic_message
    assert "remaining_turns" not in dynamic_message
    assert state.catalog_context.coverage == 1.0
    assert state.catalog_context.candidates[0].id == "metric.revenue"


async def test_structured_clarifier_uses_schema_result_without_raw_fallback():
    decision = ClarificationDecision.model_validate(
        {
            "status": "ready",
            "missing_dimensions": [],
            "clarification_question": None,
            "clarified_question": "Return 2026 revenue.",
            "proposed_defaults": {},
        }
    )
    model = _StructuredCapableModel([decision])
    clarifier = StructuredClarifier(model, template="Static clarification policy.")

    result = await clarifier(_state(question="Return revenue."))

    assert result == decision
    assert model.schema is ClarificationDecision
    assert len(model.structured.calls) == 1
    assert model.raw_calls == []
    assert model.structured.calls[0][0][0].content == "Static clarification policy."


async def test_structured_clarifier_corrects_once_with_exact_json():
    response = (
        '{"status":"ready","missing_dimensions":[],"clarification_question":null,'
        '"clarified_question":"Return 2026 revenue.","proposed_defaults":{}}'
    )
    model = _StructuredCapableModel([None], [AIMessage(content=response)])
    clarifier = StructuredClarifier(model)

    result = await clarifier(_state(question="Return revenue."))

    assert result.clarified_question == "Return 2026 revenue."
    assert len(model.raw_calls) == 1
    assert "Do not include Markdown fences" in model.raw_calls[0][0][-1].content


async def test_structured_clarifier_corrects_a_validation_error_once():
    with pytest.raises(ValidationError) as invalid:
        ClarificationDecision.model_validate({"status": "ready"})
    corrected = _ready()
    model = _StructuredCapableModel(
        [invalid.value],
        [AIMessage(content=corrected.model_dump_json())],
    )

    assert await StructuredClarifier(model)(_state()) == corrected
    assert len(model.raw_calls) == 1


async def test_structured_clarifier_rejects_fenced_json_correction():
    response = (
        '```json\n{"status":"ready","missing_dimensions":[],"clarification_question":null,'
        '"clarified_question":"Return 2026 revenue.","proposed_defaults":{}}\n```'
    )
    model = _StructuredCapableModel([None], [AIMessage(content=response)])
    clarifier = StructuredClarifier(model)

    with pytest.raises(InvalidClarificationDecisionError, match="one valid ClarificationDecision"):
        await clarifier(_state(question="Return revenue."))


async def test_structured_clarifier_applies_one_total_timeout():
    model = _StructuredCapableModel([_ready()])
    model.structured.delay = 0.05
    clarifier = StructuredClarifier(model, timeout=0.001)

    with pytest.raises(ClarifierTimeoutError, match="configured timeout"):
        await clarifier(_state(question="Return revenue."))


def test_structured_clarifier_fails_fast_for_unsupported_model():
    class UnsupportedModel:
        def with_structured_output(self, _schema):
            raise NotImplementedError

    with pytest.raises(ClarifierConfigurationError, match="does not support"):
        StructuredClarifier(UnsupportedModel())


async def test_hybrid_prompt_node_uses_nat_required_text_prompt(monkeypatch):
    manager = MagicMock()
    manager.prompt_user_input = AsyncMock()
    response = MagicMock()
    response.content.text = "Use calendar year 2026."
    manager.prompt_user_input.return_value = response
    context = MagicMock()
    context.user_interaction_manager = manager
    monkeypatch.setattr(Context, "get", lambda: context)

    state = _state(clarification_decision=_needs_clarification())
    update = await _agent(_SequenceClarifier([_ready()])).ask_clarification(state)

    prompt = manager.prompt_user_input.await_args.args[0]
    assert prompt.text == "Which reporting period should I use?"
    assert prompt.required is True
    assert update["clarification_history"][0].user_reply == "Use calendar year 2026."
