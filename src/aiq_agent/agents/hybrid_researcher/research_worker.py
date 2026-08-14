# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused single-query research worker used by Hybrid DAG nodes."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool

from aiq_agent.agents.deep_researcher.custom_middleware import SourceRegistryMiddleware
from aiq_agent.agents.deep_researcher.factory import build_common_middleware_for_tools
from aiq_agent.agents.deep_researcher.factory import build_researcher_runnable
from aiq_agent.agents.deep_researcher.models import ResearchNotes
from aiq_agent.agents.deep_researcher.models import ResearchQuery
from aiq_agent.agents.deep_researcher.tools.research import invoke_researcher_runnable
from aiq_agent.agents.deep_researcher.tools.source_tool_batching import activate_source_tool_budget
from aiq_agent.agents.deep_researcher.tools.source_tool_batching import adapt_source_tools_for_research
from aiq_agent.agents.deep_researcher.tools.source_tool_batching import reset_source_tool_budget
from aiq_agent.common import filter_tools_by_sources
from aiq_agent.common import load_prompt
from aiq_agent.common import render_prompt_template

from .models import ResearchWorkerRequest

_PROMPTS_DIR = Path(__file__).parent / "prompts"


class HybridResearchWorker:
    """Run one focused research question without nesting the Deep Research workflow."""

    def __init__(
        self,
        *,
        llm: BaseChatModel,
        tools: list[BaseTool],
        callbacks: Sequence[BaseCallbackHandler] = (),
        prompt_template: str | None = None,
        max_source_tool_calls: int = 6,
        max_concurrent_source_tool_calls: int = 2,
        max_source_tool_batch_size: int = 2,
        timeout_seconds: float = 180,
    ) -> None:
        if max_source_tool_calls < 1:
            raise ValueError("max_source_tool_calls must be positive")
        if max_concurrent_source_tool_calls < 1:
            raise ValueError("max_concurrent_source_tool_calls must be positive")
        if max_source_tool_batch_size < 1:
            raise ValueError("max_source_tool_batch_size must be positive")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self._llm = llm
        self._tools = list(tools)
        self._callbacks = list(callbacks)
        self._prompt_template = prompt_template or load_prompt(_PROMPTS_DIR, "research_worker")
        self._max_source_tool_calls = max_source_tool_calls
        self._max_concurrent_source_tool_calls = max_concurrent_source_tool_calls
        self._max_source_tool_batch_size = max_source_tool_batch_size
        self._timeout_seconds = timeout_seconds

    async def run(self, request: ResearchWorkerRequest) -> ResearchNotes:
        """Research one question with the request-selected source tools."""
        selected_tools = filter_tools_by_sources(self._tools, request.data_sources)
        if not selected_tools:
            raise ValueError("No research source is available for this Hybrid research node")

        tools_info = [{"name": tool.name, "description": tool.description} for tool in selected_tools]
        researcher_tools = adapt_source_tools_for_research(
            selected_tools,
            source_tool_names={tool.name for tool in selected_tools},
            max_concurrent_source_tool_calls=self._max_concurrent_source_tool_calls,
            max_batch_size=self._max_source_tool_batch_size,
        )
        source_registry = SourceRegistryMiddleware(source_tool_names={tool.name for tool in researcher_tools})
        researcher_middleware = build_common_middleware_for_tools(
            tools=researcher_tools,
            source_registry_middleware=source_registry,
        )
        system_prompt = render_prompt_template(
            self._prompt_template,
            current_datetime=datetime.now().astimezone().isoformat(timespec="seconds"),
            tools=tools_info,
            max_source_tool_calls=self._max_source_tool_calls,
            max_concurrent_source_tool_calls=self._max_concurrent_source_tool_calls,
            max_source_tool_batch_size=self._max_source_tool_batch_size,
        )
        runnable = build_researcher_runnable(
            researcher_model=self._llm,
            researcher_tools=researcher_tools,
            researcher_middleware=researcher_middleware,
            system_prompt=system_prompt,
            include_filesystem_tools=False,
        )

        tool_names = [tool.name for tool in selected_tools]
        query = ResearchQuery(
            query=request.question,
            preferred_tools=tool_names[:1],
            fallback_tools=tool_names[1:],
            target_components=["research_context"],
            rationale="Gather source-grounded context for the Hybrid Research objective.",
        )
        query_json = json.dumps(query.model_dump(mode="json"), indent=2, ensure_ascii=False)
        invoke_state: dict[str, Any] = {
            "messages": [
                HumanMessage(
                    content=(
                        "Execute this focused ResearchQuery and return a structured ResearchNotes response.\n\n"
                        f"ResearchQuery JSON:\n{query_json}"
                    )
                )
            ]
        }
        invoke_config = {
            "run_name": "hybrid-research-worker",
            "callbacks": self._callbacks,
        }
        budget_token = activate_source_tool_budget(self._max_source_tool_calls)
        try:
            execution_timeout = asyncio.timeout(self._timeout_seconds)
            try:
                async with execution_timeout:
                    result = await invoke_researcher_runnable(
                        researcher_runnable=runnable,
                        invoke_state=invoke_state,
                        invoke_config=invoke_config,
                        query_label=request.question,
                    )
            except TimeoutError as exc:
                if execution_timeout.expired():
                    raise TimeoutError(
                        f"Hybrid research worker exceeded its {self._timeout_seconds:g}-second deadline"
                    ) from exc
                raise
            if execution_timeout.expired():
                raise TimeoutError(f"Hybrid research worker exceeded its {self._timeout_seconds:g}-second deadline")
            source_registry.register_research_note_sources([result])
            return result
        finally:
            reset_source_tool_budget(budget_token)


__all__ = ["HybridResearchWorker"]
