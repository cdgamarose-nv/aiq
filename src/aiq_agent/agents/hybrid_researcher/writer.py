# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tool-free cross-source final synthesis for Hybrid Research."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage

from aiq_agent.common import load_prompt
from aiq_agent.common import render_prompt_template
from aiq_agent.common.citation_verification import SourceEntry
from aiq_agent.common.citation_verification import SourceRegistry
from aiq_agent.common.citation_verification import get_session_registry
from aiq_agent.common.citation_verification import sanitize_report
from aiq_agent.common.citation_verification import verify_citations

from .models import GSFQuerySuccess
from .models import HybridResearchState
from .models import ResearchTaskResult
from .models import StructuredAnalysisResult
from .scheduler import runs_by_task

_PROMPTS_DIR = Path(__file__).parent / "prompts"


class HybridWriter:
    """Write once from the terminal append-only ledger without new tools."""

    def __init__(
        self,
        llm: BaseChatModel,
        *,
        template: str | None = None,
        timeout_seconds: float = 120,
        max_input_chars: int = 200_000,
        enable_citation_verification: bool = True,
        callbacks: Sequence[BaseCallbackHandler] = (),
    ) -> None:
        if timeout_seconds <= 0 or max_input_chars < 1:
            raise ValueError("writer limits must be positive")
        self._llm = llm
        self._policy = render_prompt_template(template or load_prompt(_PROMPTS_DIR, "writer"))
        self._timeout_seconds = timeout_seconds
        self._max_input_chars = max_input_chars
        self._enable_citation_verification = enable_citation_verification
        self._callbacks = list(callbacks)

    async def __call__(self, state: HybridResearchState) -> str:
        payload, sources = _writer_context(state)
        context = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str)
        if len(context) > self._max_input_chars:
            raise ValueError("Hybrid writer context exceeds its configured character limit")
        config = {"callbacks": self._callbacks} if self._callbacks else None
        try:
            async with asyncio.timeout(self._timeout_seconds):
                response = await self._llm.ainvoke(
                    [SystemMessage(content=self._policy), HumanMessage(content=context)],
                    config=config,
                )
        except TimeoutError as exc:
            raise TimeoutError("Hybrid writer exceeded its configured deadline") from exc
        answer = _message_text(response).strip()
        if not answer:
            raise ValueError("Hybrid writer returned an empty answer")
        registry = get_session_registry() or SourceRegistry()
        for source in sources:
            registry.add(source)
        if self._enable_citation_verification:
            answer = verify_citations(answer, registry, reference_sources=sources).verified_report
        return sanitize_report(answer).sanitized_report


def _writer_context(state: HybridResearchState) -> tuple[dict[str, Any], list[SourceEntry]]:
    plan = state.plan
    if plan is None or not state.continuation_history or state.continuation_history[-1].action != "finish":
        raise ValueError("Hybrid writer requires a terminal finish decision")
    by_task = runs_by_task(state.task_runs)
    evidence: list[dict[str, Any]] = []
    research_locators: set[str] = set()
    gsf_sources: list[SourceEntry] = []
    for task in plan.tasks:
        run = by_task.get(task.id)
        if run is None or run.status != "succeeded" or run.result is None:
            continue
        evidence.append(
            {
                "task_id": task.id,
                "kind": task.kind,
                "objective": task.objective,
                "depends_on": list(task.depends_on),
                "result": run.result.model_dump(mode="json"),
            }
        )
        if isinstance(run.result, ResearchTaskResult):
            research_locators.update(source.locator for source in run.result.notes.sources)
        elif isinstance(run.result, StructuredAnalysisResult):
            for provenance in run.result.gsf_provenance:
                if isinstance(provenance, GSFQuerySuccess):
                    gsf_sources.append(
                        SourceEntry(
                            citation_key=provenance.citation_key,
                            title=provenance.citation_key,
                            source_type="structured_data",
                            tool_name="gsf__text_to_sql",
                        )
                    )
    if not evidence:
        raise ValueError("Hybrid writer has no successful evidence")
    registry = get_session_registry()
    captured = registry.all_sources() if registry else []
    research_sources = _resolve_research_sources(registry, captured, research_locators)
    sources = _deduplicate_sources([*research_sources, *gsf_sources])
    return (
        {
            "original_objective": state.question,
            "clarified_objective": state.clarified_question or plan.objective,
            "reference_datetime": state.clarification_reference_datetime.isoformat(timespec="seconds"),
            "task_ledger": [
                {
                    "task": task.model_dump(mode="json"),
                    "status": by_task[task.id].status,
                    "error": by_task[task.id].error,
                }
                for task in plan.tasks
            ],
            "continuation_reasoning": state.continuation_history[-1].reasoning,
            "evidence": evidence,
            "sources": [
                {
                    "number": index,
                    "title": source.title,
                    "url": source.url,
                    "citation_key": source.citation_key,
                    "source_type": source.source_type,
                }
                for index, source in enumerate(sources, start=1)
            ],
        },
        sources,
    )


def _resolve_research_sources(
    registry: SourceRegistry | None,
    captured_sources: list[SourceEntry],
    locators: set[str],
) -> list[SourceEntry]:
    if registry is None:
        return []
    by_url = {source.url: source for source in captured_sources if source.url}
    by_key = {source.citation_key: source for source in captured_sources if source.citation_key}
    selected: list[SourceEntry] = []
    for locator in sorted(locators):
        if locator.startswith(("http://", "https://")):
            canonical = registry.resolve_url(locator)
            source = by_url.get(canonical) if canonical else None
        else:
            canonical = registry.resolve_citation_key(locator)
            source = by_key.get(canonical) if canonical else None
        if source is not None:
            selected.append(source)
    return selected


def _deduplicate_sources(sources: list[SourceEntry]) -> list[SourceEntry]:
    seen: set[str] = set()
    result: list[SourceEntry] = []
    for source in sources:
        key = source.url or source.citation_key
        if key and key not in seen:
            seen.add(key)
            result.append(source)
    return result


def _message_text(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            block if isinstance(block, str) else block.get("text", "")
            for block in content
            if isinstance(block, str) or isinstance(block, dict)
        )
    return str(content)


__all__ = ["HybridWriter"]
