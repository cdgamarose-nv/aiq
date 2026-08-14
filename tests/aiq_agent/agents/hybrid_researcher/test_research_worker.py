# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the focused Hybrid research worker."""

import asyncio

import pytest
from langchain_core.tools import tool

from aiq_agent.agents.deep_researcher.custom_middleware import SourceRegistryMiddleware
from aiq_agent.agents.deep_researcher.models import ResearchNotes
from aiq_agent.agents.hybrid_researcher.models import ResearchWorkerRequest
from aiq_agent.agents.hybrid_researcher.research_worker import HybridResearchWorker


def _notes() -> ResearchNotes:
    return ResearchNotes(
        query_topic="Market context",
        target_components=["research_context"],
        summary="Summary",
        findings=[],
        gaps=[],
        sources=[],
        narrative_notes="Notes",
        language="English",
    )


@tool
async def web_search(query: str) -> str:
    """Search the web for external context."""
    return query


async def test_worker_builds_one_focused_research_query_without_batch_or_filesystem_contract(monkeypatch):
    captured = {}
    registered_notes = []

    class Runnable:
        async def ainvoke(self, state, config=None):
            captured["state"] = state
            captured["config"] = config
            return {"structured_response": _notes()}

    def build_runnable(**kwargs):
        captured["build"] = kwargs
        return Runnable()

    def adapt_tools(tools, **kwargs):
        captured["adapt"] = {"tools": tools, **kwargs}
        return tools

    def activate_budget(max_calls):
        captured["activated_budget"] = max_calls
        return "budget-token"

    def reset_budget(token):
        captured["reset_budget"] = token

    monkeypatch.setattr(
        "aiq_agent.agents.hybrid_researcher.research_worker.build_researcher_runnable",
        build_runnable,
    )
    monkeypatch.setattr(
        "aiq_agent.agents.hybrid_researcher.research_worker.adapt_source_tools_for_research",
        adapt_tools,
    )
    monkeypatch.setattr(
        "aiq_agent.agents.hybrid_researcher.research_worker.activate_source_tool_budget",
        activate_budget,
    )
    monkeypatch.setattr(
        "aiq_agent.agents.hybrid_researcher.research_worker.reset_source_tool_budget",
        reset_budget,
    )
    monkeypatch.setattr(
        SourceRegistryMiddleware,
        "register_research_note_sources",
        lambda _self, notes: registered_notes.append(notes),
    )
    worker = HybridResearchWorker(
        llm=object(),
        tools=[web_search],
        max_source_tool_calls=6,
        max_concurrent_source_tool_calls=2,
        max_source_tool_batch_size=2,
    )

    result = await worker.run(ResearchWorkerRequest(question="What changed in the market?"))

    assert result == _notes()
    assert captured["adapt"] == {
        "tools": [web_search],
        "source_tool_names": {"web_search"},
        "max_concurrent_source_tool_calls": 2,
        "max_batch_size": 2,
    }
    assert captured["activated_budget"] == 6
    assert captured["reset_budget"] == "budget-token"
    assert captured["build"]["include_filesystem_tools"] is False
    assert captured["build"]["researcher_tools"] == [web_search]
    middleware = captured["build"]["researcher_middleware"]
    assert [type(item).__name__ for item in middleware] == [
        "EmptyContentFixMiddleware",
        "ToolNameSanitizationMiddleware",
        "ToolRetryMiddleware",
        "SourceRegistryMiddleware",
        "ToolResultPruningMiddleware",
        "ModelRetryMiddleware",
    ]
    assert registered_notes == [[result]]
    assert "/shared/plan.json" not in captured["build"]["system_prompt"]
    assert "at most 6 concrete source requests" in captured["build"]["system_prompt"]
    message = captured["state"]["messages"][0].content
    assert "ResearchQuery JSON" in message
    assert '"query": "What changed in the market?"' in message
    assert '"preferred_tools": [\n    "web_search"' in message


async def test_worker_fails_when_request_selection_leaves_no_research_source(monkeypatch):
    monkeypatch.setattr(
        "aiq_agent.agents.hybrid_researcher.research_worker.filter_tools_by_sources",
        lambda _tools, _sources: [],
    )
    worker = HybridResearchWorker(llm=object(), tools=[web_search])

    with pytest.raises(ValueError, match="No research source"):
        await worker.run(ResearchWorkerRequest(question="What changed?", data_sources=["gsf"]))


async def test_worker_enforces_total_deadline_and_resets_source_budget(monkeypatch):
    reset_tokens = []

    class Runnable:
        async def ainvoke(self, _state, config=None):
            del config
            await asyncio.sleep(1)

    monkeypatch.setattr(
        "aiq_agent.agents.hybrid_researcher.research_worker.build_researcher_runnable",
        lambda **_kwargs: Runnable(),
    )
    monkeypatch.setattr(
        "aiq_agent.agents.hybrid_researcher.research_worker.adapt_source_tools_for_research",
        lambda tools, **_kwargs: tools,
    )
    monkeypatch.setattr(
        "aiq_agent.agents.hybrid_researcher.research_worker.activate_source_tool_budget",
        lambda _max_calls: "budget-token",
    )
    monkeypatch.setattr(
        "aiq_agent.agents.hybrid_researcher.research_worker.reset_source_tool_budget",
        reset_tokens.append,
    )
    worker = HybridResearchWorker(llm=object(), tools=[web_search], timeout_seconds=0.01)

    with pytest.raises(TimeoutError, match="0.01-second deadline"):
        await worker.run(ResearchWorkerRequest(question="What changed?"))

    assert reset_tokens == ["budget-token"]
