# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NAT registrations for hierarchical Hybrid Research and its focused workers."""

import logging
import uuid
from typing import Any

from langchain_core.messages import AIMessage

from aiq_agent.agents.chat_researcher.models import RESEARCH_WORKFLOW_FAILURE_ERROR
from aiq_agent.agents.chat_researcher.models import ChatResearcherState
from aiq_agent.agents.chat_researcher.models import WorkflowFailure
from aiq_agent.agents.chat_researcher.models import WorkflowSuccess
from aiq_agent.agents.deep_researcher.deepagents_runtime import DeepResearchSandboxConfig
from aiq_agent.agents.deep_researcher.deepagents_runtime import create_sandbox_provider
from aiq_agent.agents.deep_researcher.models import ResearchNotes
from aiq_agent.common import VerboseTraceCallback
from aiq_agent.common import get_all_tool_refs
from aiq_agent.common import get_checkpointer
from aiq_agent.common import get_latest_user_query
from aiq_agent.common import is_verbose
from nat.builder.builder import Builder
from nat.builder.context import Context
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function

from .agent import HybridResearchAgent
from .clarifier import StructuredClarifier
from .config import HybridResearchAgentConfig
from .config import HybridResearchWorkerConfig
from .config import StructuredAnalysisWorkerConfig
from .config import validate_analysis_sandbox
from .execution import ExecutorRegistry
from .execution import ResearchExecutor
from .execution import StructuredAnalysisExecutor
from .models import HybridResearchState
from .models import ResearchWorkerRequest
from .models import StructuredAnalysisRequest
from .models import StructuredAnalysisResult
from .planner import ContinuationPlanner
from .planner import InitialTaskPlanner
from .research_worker import HybridResearchWorker
from .structured_analysis import StructuredAnalysisWorker
from .writer import HybridWriter

logger = logging.getLogger(__name__)

_REQUIRED_RESEARCH_EXCLUSIONS = frozenset({"gsf__catalog_search", "gsf__text_to_sql"})


@register_function(config_type=HybridResearchWorkerConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def hybrid_research_worker(config: HybridResearchWorkerConfig, builder: Builder):
    """Register the focused unstructured research worker."""
    llm = await builder.get_llm(config.researcher_llm, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    tool_refs = config.tools or get_all_tool_refs()
    tools = await builder.get_tools(tool_names=tool_refs, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    excluded = {*config.exclude_tools, *_REQUIRED_RESEARCH_EXCLUSIONS}
    tools = [tool for tool in tools if getattr(tool, "name", "") not in excluded]
    callbacks = [VerboseTraceCallback()] if is_verbose(config.verbose) else []
    worker = HybridResearchWorker(
        llm=llm,
        tools=tools,
        callbacks=callbacks,
        max_source_tool_calls=config.max_source_tool_calls,
        max_concurrent_source_tool_calls=config.max_concurrent_source_tool_calls,
        max_source_tool_batch_size=config.max_source_tool_batch_size,
        timeout_seconds=config.timeout_seconds,
    )

    async def _run(request: ResearchWorkerRequest) -> ResearchNotes:
        return await worker.run(request)

    yield FunctionInfo.from_fn(_run, description="Run one focused source-grounded research trajectory.")


@register_function(config_type=StructuredAnalysisWorkerConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def structured_analysis_worker(config: StructuredAnalysisWorkerConfig, builder: Builder):
    """Register the headless bounded structured-analysis worker."""
    llm = await builder.get_llm(config.structured_analysis_llm, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    sql_function = await builder.get_function(config.sql_tool)
    sandbox_config = _resolve_sandbox_config(config.sandbox, builder)
    callbacks = [VerboseTraceCallback()] if is_verbose(config.verbose) else []
    worker = StructuredAnalysisWorker(
        llm=llm,
        gsf_invoke=sql_function.ainvoke,
        sandbox_factory=lambda job_id: create_sandbox_provider(sandbox_config, job_id),
        database_name=config.database_name,
        sql_max_rows=config.sql_max_rows,
        callbacks=callbacks,
        timeout_seconds=config.structured_analysis_timeout_seconds,
        execute_timeout_seconds=config.python_execute_timeout_seconds,
        max_gsf_calls=config.structured_max_gsf_calls,
        max_python_calls=config.structured_max_python_calls,
        max_code_chars=config.max_code_chars,
        max_output_chars=config.max_output_chars,
        model_result_rows=config.model_result_rows,
    )

    async def _run(request: StructuredAnalysisRequest) -> StructuredAnalysisResult:
        return await worker.run(request)

    yield FunctionInfo.from_fn(
        _run,
        description="Analyze one enterprise-data objective with bounded GSF and sandboxed Python calls.",
    )


@register_function(config_type=HybridResearchAgentConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def hybrid_research_agent(config: HybridResearchAgentConfig, builder: Builder):
    """Register the complete append-only Hybrid Research workflow."""
    verbose = is_verbose(config.verbose)
    callbacks = [VerboseTraceCallback()] if verbose else []
    clarifier = None
    if config.enable_clarifier:
        assert config.clarifier_llm is not None
        clarifier_llm = await builder.get_llm(config.clarifier_llm, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
        clarifier = StructuredClarifier(clarifier_llm, timeout=config.llm_timeout, callbacks=callbacks)
    planner_llm = await builder.get_llm(config.planner_llm, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    writer_llm = await builder.get_llm(config.writer_llm, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    planner = InitialTaskPlanner(
        planner_llm,
        timeout=config.planner_timeout_seconds,
        max_total_tasks=config.max_total_tasks,
        callbacks=callbacks,
    )
    continuation = ContinuationPlanner(
        planner_llm,
        timeout=config.continuation_timeout_seconds,
        max_total_tasks=config.max_total_tasks,
        callbacks=callbacks,
    )
    writer = HybridWriter(
        writer_llm,
        timeout_seconds=config.writer_timeout_seconds,
        max_input_chars=config.writer_max_input_chars,
        enable_citation_verification=config.enable_citation_verification,
        callbacks=callbacks,
    )
    research_function = await builder.get_function(config.research_worker)
    structured_function = await builder.get_function(config.structured_analysis_worker)
    registry = ExecutorRegistry(
        {
            "research": ResearchExecutor(research_function.ainvoke),
            "structured_analysis": StructuredAnalysisExecutor(structured_function.ainvoke),
        }
    )
    checkpointer = await get_checkpointer(config.checkpoint_db)
    agent = HybridResearchAgent(
        clarifier,
        registry,
        enable_clarifier=config.enable_clarifier,
        max_clarification_turns=config.max_clarification_turns,
        max_parallel_tasks=config.max_parallel_tasks,
        max_total_tasks=config.max_total_tasks,
        max_plan_extensions=config.max_plan_extensions,
        verbose=verbose,
        plan_builder=planner,
        continuation_builder=continuation,
        writer=writer,
        database_name=config.database_name,
        checkpointer=checkpointer,
    )

    async def _run(chat_state: ChatResearcherState) -> dict[str, Any]:
        try:
            if chat_state.catalog_context is None:
                raise ValueError("Hybrid Research requires catalog_context from the entry router")
            question = chat_state.original_query or get_latest_user_query(chat_state.messages)
            state = HybridResearchState(
                question=question,
                catalog_context=chat_state.catalog_context,
                data_sources=chat_state.data_sources,
                skip_clarifier=chat_state.skip_clarifier,
                user_info=chat_state.user_info,
            )
            result = await agent.run(state, thread_id=f"hybrid:{_workflow_run_id()}")
            return hybrid_boundary_update(result)
        except Exception as exc:  # noqa: BLE001 - sanitize failures at parent workflow boundary
            logger.warning("Hybrid Research failed (error_type=%s)", type(exc).__name__)
            return {
                "messages": [AIMessage(content=RESEARCH_WORKFLOW_FAILURE_ERROR)],
                "workflow_outcome": WorkflowFailure(error=RESEARCH_WORKFLOW_FAILURE_ERROR),
            }

    yield FunctionInfo.from_fn(_run, description="Hierarchical Hybrid Research for catalog-supported questions.")


def _resolve_sandbox_config(
    sandbox: DeepResearchSandboxConfig | Any,
    builder: Builder,
) -> DeepResearchSandboxConfig:
    if isinstance(sandbox, DeepResearchSandboxConfig):
        return validate_analysis_sandbox(sandbox)
    resolved = builder.get_function_config(sandbox)
    if not isinstance(resolved, DeepResearchSandboxConfig):
        raise TypeError(f"{sandbox!r} must reference DeepResearchSandboxConfig, got {type(resolved).__name__}")
    return validate_analysis_sandbox(resolved)


def _workflow_run_id() -> str:
    try:
        value = Context.get().workflow_run_id
    except Exception:  # noqa: BLE001 - NAT context is optional in direct unit invocation
        value = None
    return str(value or uuid.uuid4())


def clarifier_boundary_update(state: HybridResearchState) -> dict[str, Any] | None:
    if state.clarification_required is not None:
        required = state.clarification_required
        return {"messages": [AIMessage(content=required.clarification_question)], "workflow_outcome": required}
    if state.clarified_question is not None:
        return None
    raise ValueError("Clarifier ended without a clarified objective or required-input outcome")


def hybrid_boundary_update(state: HybridResearchState) -> dict[str, Any]:
    clarifier_update = clarifier_boundary_update(state)
    if clarifier_update is not None:
        return clarifier_update
    if state.final_answer is not None:
        return {
            "messages": [AIMessage(content=state.final_answer)],
            "final_report": state.final_answer,
            "last_report_markdown": state.final_answer,
            "workflow_outcome": WorkflowSuccess(result=state.final_answer),
        }
    if state.terminal_message is None:
        raise ValueError("Hybrid Research ended without a terminal answer or failure")
    return {
        "messages": [AIMessage(content=state.terminal_message)],
        "workflow_outcome": WorkflowFailure(error=RESEARCH_WORKFLOW_FAILURE_ERROR),
    }


__all__ = [
    "clarifier_boundary_update",
    "hybrid_boundary_update",
    "hybrid_research_agent",
    "hybrid_research_worker",
    "structured_analysis_worker",
]
