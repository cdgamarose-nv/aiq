# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the implemented Hybrid Research clarifier boundary."""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from aiq_agent.agents.chat_researcher.models import CatalogCandidate
from aiq_agent.agents.chat_researcher.models import CatalogRoutingResponse
from aiq_agent.agents.chat_researcher.models import ChatResearcherState
from aiq_agent.agents.chat_researcher.models import WorkflowClarificationRequired
from aiq_agent.agents.deep_researcher.deepagents_runtime import DeepResearchSandboxConfig
from aiq_agent.agents.hybrid_researcher import register as hybrid_register
from aiq_agent.agents.hybrid_researcher.config import HybridResearchAgentConfig
from aiq_agent.agents.hybrid_researcher.config import HybridResearchWorkerConfig
from aiq_agent.agents.hybrid_researcher.config import StructuredAnalysisWorkerConfig
from aiq_agent.agents.hybrid_researcher.models import HybridResearchState
from aiq_agent.agents.hybrid_researcher.models import ResearchWorkerRequest
from aiq_agent.agents.hybrid_researcher.register import clarifier_boundary_update
from aiq_agent.agents.hybrid_researcher.register import hybrid_boundary_update


def _catalog_context() -> CatalogRoutingResponse:
    return CatalogRoutingResponse(
        request_id="catalog-request-1",
        coverage=1,
        candidates=[
            CatalogCandidate(
                id="metric.revenue",
                label="Metric",
                attribute="recognized_revenue",
                term="Revenue",
            )
        ],
    )


def test_headless_clarification_maps_to_public_workflow_outcome():
    required = WorkflowClarificationRequired(
        clarification_question="Which reporting period should I use?",
        missing_dimensions=("time_window",),
    )
    state = HybridResearchState(
        question="Compare revenue.",
        catalog_context=_catalog_context(),
        clarification_required=required,
    )

    update = clarifier_boundary_update(state)

    assert update is not None
    assert update["messages"][0].content == required.clarification_question
    assert update["workflow_outcome"] == WorkflowClarificationRequired(
        clarification_question=required.clarification_question,
        missing_dimensions=required.missing_dimensions,
    )


def test_plan_ready_question_stays_inside_hybrid_workflow():
    state = HybridResearchState(
        question="Compare revenue.",
        catalog_context=_catalog_context(),
        clarified_question="Compare 2026 revenue.",
    )

    assert clarifier_boundary_update(state) is None


def test_resolved_sandbox_reference_must_meet_analysis_requirements():
    builder = MagicMock()
    builder.get_function_config.return_value = DeepResearchSandboxConfig(
        provider="modal",
        packages=("pandas",),
        network="open",
    )

    with pytest.raises(ValueError, match="requires sandbox network='blocked'"):
        hybrid_register._resolve_sandbox_config("hybrid_analysis_sandbox", builder)


def test_terminal_failure_maps_to_typed_failure_without_claiming_success():
    state = HybridResearchState(
        question="Compare revenue.",
        catalog_context=_catalog_context(),
        clarified_question="Compare 2026 revenue.",
        terminal_message="Hybrid Research could not complete this request responsibly.",
    )

    update = hybrid_boundary_update(state)

    assert update["messages"][0].content == state.terminal_message
    assert update["workflow_outcome"].status == "failed"


@tool
async def _web_search(query: str) -> str:
    """Search the web."""
    return query


@tool
async def _gsf_sql(query: str) -> str:
    """Query GSF."""
    return query


async def test_registered_research_worker_always_excludes_gsf_tools():
    _web_search.name = "web_search_tool"
    _gsf_sql.name = "gsf__text_to_sql"
    builder = MagicMock()
    builder.get_llm = AsyncMock(return_value=MagicMock())
    builder.get_tools = AsyncMock(return_value=[_web_search, _gsf_sql])

    class Worker:
        async def run(self, request: ResearchWorkerRequest) -> ResearchWorkerRequest:
            return request

    worker = Worker()

    with patch.object(hybrid_register, "HybridResearchWorker", return_value=worker) as worker_class:
        registration = hybrid_register.hybrid_research_worker.__wrapped__(
            HybridResearchWorkerConfig(researcher_llm="researcher_llm", exclude_tools=[]),
            builder,
        )
        function_info = await anext(registration)
        await registration.aclose()

    assert function_info.single_fn is not None
    assert worker_class.call_args.kwargs["tools"] == [_web_search]
    assert worker_class.call_args.kwargs["max_source_tool_calls"] == 6
    assert worker_class.call_args.kwargs["max_concurrent_source_tool_calls"] == 2
    assert worker_class.call_args.kwargs["max_source_tool_batch_size"] == 2
    assert worker_class.call_args.kwargs["timeout_seconds"] == 180


async def test_registered_hybrid_agent_maps_chat_state_to_skeleton_state():
    research_function = MagicMock()
    research_function.ainvoke = AsyncMock()
    structured_function = MagicMock()
    structured_function.ainvoke = AsyncMock()
    builder = MagicMock()
    builder.get_llm = AsyncMock(return_value=MagicMock())
    builder.get_function = AsyncMock(side_effect=[research_function, structured_function])
    fake_agent = MagicMock()
    fake_agent.run = AsyncMock(
        side_effect=lambda state, thread_id: state.model_copy(
            update={
                "clarified_question": state.question,
                "final_answer": "Revenue increased 25%.",
            }
        )
    )
    config = HybridResearchAgentConfig(
        enable_clarifier=False,
        planner_llm="planner_llm",
        writer_llm="writer_llm",
        research_worker="hybrid_research_worker",
        structured_analysis_worker="structured_analysis_worker",
        verbose=True,
    )

    with (
        patch.object(hybrid_register, "get_checkpointer", AsyncMock(return_value=MagicMock())),
        patch.object(hybrid_register, "HybridResearchAgent", return_value=fake_agent) as agent_class,
        patch.object(hybrid_register, "_workflow_run_id", return_value="run-1"),
    ):
        registration = hybrid_register.hybrid_research_agent.__wrapped__(config, builder)
        function_info = await anext(registration)
        chat_state = ChatResearcherState(
            messages=[HumanMessage(content="Compare revenue.")],
            catalog_context=_catalog_context(),
            data_sources=["web_search", "gsf"],
        )
        update = await function_info.single_fn(chat_state)
        await registration.aclose()

    state = fake_agent.run.await_args.args[0]
    assert state.question == "Compare revenue."
    assert state.catalog_context == chat_state.catalog_context
    assert state.data_sources == ["web_search", "gsf"]
    assert agent_class.call_args.kwargs["verbose"] is True
    assert fake_agent.run.await_args.kwargs["thread_id"] == "hybrid:run-1"
    assert update["workflow_outcome"].status == "success"
    assert update["messages"][0].content == "Revenue increased 25%."


async def test_registered_structured_worker_receives_only_configured_gsf_and_sandbox():
    builder = MagicMock()
    builder.get_llm = AsyncMock(return_value=MagicMock())
    sql_function = MagicMock()
    sql_function.ainvoke = AsyncMock()
    builder.get_function = AsyncMock(return_value=sql_function)
    config = StructuredAnalysisWorkerConfig(
        structured_analysis_llm="structured_llm",
        sql_tool="gsf__text_to_sql",
        sandbox={"provider": "modal", "packages": ["pandas"], "network": "blocked"},
    )
    with (
        patch.object(hybrid_register, "create_sandbox_provider", return_value=MagicMock()),
        patch.object(hybrid_register, "StructuredAnalysisWorker") as worker_class,
    ):
        registration = hybrid_register.structured_analysis_worker.__wrapped__(config, builder)
        function_info = await anext(registration)
        await registration.aclose()
    assert function_info.single_fn is not None
    kwargs = worker_class.call_args.kwargs
    assert kwargs["gsf_invoke"] == sql_function.ainvoke
    assert kwargs["max_gsf_calls"] == 4
    assert kwargs["max_python_calls"] == 4
    assert kwargs["timeout_seconds"] == 600
