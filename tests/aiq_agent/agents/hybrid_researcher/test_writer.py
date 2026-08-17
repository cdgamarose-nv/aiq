# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for final append-only Hybrid synthesis and provenance."""

import json

import pytest
from langchain_core.messages import AIMessage

from aiq_agent.agents.chat_researcher.models import CatalogCandidate
from aiq_agent.agents.chat_researcher.models import CatalogRoutingResponse
from aiq_agent.agents.hybrid_researcher.models import BoundedTableEvidence
from aiq_agent.agents.hybrid_researcher.models import ContinuationDecision
from aiq_agent.agents.hybrid_researcher.models import GSFQuerySuccess
from aiq_agent.agents.hybrid_researcher.models import GSFResultColumnSummary
from aiq_agent.agents.hybrid_researcher.models import HybridResearchState
from aiq_agent.agents.hybrid_researcher.models import HybridTask
from aiq_agent.agents.hybrid_researcher.models import HybridTaskPlan
from aiq_agent.agents.hybrid_researcher.models import StructuredAnalysisResult
from aiq_agent.agents.hybrid_researcher.models import TaskRun
from aiq_agent.agents.hybrid_researcher.writer import HybridWriter
from aiq_agent.common.citation_verification import SourceRegistry
from aiq_agent.common.citation_verification import reset_session_registry
from aiq_agent.common.citation_verification import set_session_registry


class _Model:
    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.calls = []

    async def ainvoke(self, messages, config=None):
        self.calls.append((messages, config))
        return AIMessage(content=self.answer)


def _state() -> HybridResearchState:
    task = HybridTask(id="revenue", kind="structured_analysis", objective="Analyze revenue.")
    return HybridResearchState(
        question="What was revenue?",
        clarified_question="What was revenue?",
        catalog_context=CatalogRoutingResponse(
            coverage=1,
            candidates=[CatalogCandidate(id="metric", label="Metric", attribute="recognized", term="Revenue")],
        ),
        plan=HybridTaskPlan(objective="What was revenue?", tasks=(task,)),
        task_runs=[
            TaskRun(
                task_id="revenue",
                kind="structured_analysis",
                status="succeeded",
                result=StructuredAnalysisResult(
                    sufficiency="limited",
                    conclusion="Revenue was 100.",
                    gsf_provenance=(
                        GSFQuerySuccess(
                            question="What was revenue?",
                            request_id="revenue-1",
                            citation_key="GSF request revenue-1",
                            returned_row_count=1,
                            result_truncated=False,
                        ),
                    ),
                    limitations=("Preliminary value.",),
                ),
            )
        ],
        continuation_history=(ContinuationDecision(action="finish", reasoning="Available evidence is sufficient."),),
    )


async def test_writer_receives_bounded_provenance_lineage_and_limitations():
    model = _Model("Revenue was 100 [1].\n\n**References:**\n- [1] GSF request revenue-1")
    writer = HybridWriter(model, template="Static policy.", enable_citation_verification=False)
    answer = await writer(_state())
    assert "Revenue was 100" in answer
    context = json.loads(model.calls[0][0][1].content)
    evidence = context["evidence"][0]
    assert evidence["depends_on"] == []
    assert evidence["result"]["limitations"] == ["Preliminary value."]
    provenance = evidence["result"]["gsf_provenance"][0]
    assert provenance["returned_row_count"] == 1
    assert "rows" not in provenance
    assert "sql" not in provenance
    assert "artifact" not in str(evidence)
    assert context["sources"][0]["citation_key"] == "GSF request revenue-1"


async def test_writer_context_is_compact_by_terminal_contract():
    state = _state()
    model = _Model("Revenue was 100 [1].\n\n**References:**\n- [1] GSF request revenue-1")
    await HybridWriter(model, template="Static policy.", enable_citation_verification=False)(state)
    context_text = model.calls[0][0][1].content
    assert len(context_text) < 20_000
    assert "rows" not in context_text
    assert "artifact" not in context_text


async def test_writer_receives_complete_bounded_table_rows():
    state = _state()
    result = state.task_runs[0].result
    assert isinstance(result, StructuredAnalysisResult)
    table = BoundedTableEvidence(
        citation_key="GSF request revenue-1",
        columns=(GSFResultColumnSummary(name="quarter"), GSFResultColumnSummary(name="revenue")),
        rows=({"quarter": "Q1", "revenue": 100}, {"quarter": "Q2", "revenue": 125}),
        returned_row_count=2,
    )
    state = state.model_copy(
        update={
            "task_runs": [
                state.task_runs[0].model_copy(update={"result": result.model_copy(update={"table_evidence": table})})
            ]
        }
    )
    model = _Model("Q1 was 100 and Q2 was 125 [1].\n\n**References:**\n- [1] GSF request revenue-1")
    await HybridWriter(model, template="Static policy.", enable_citation_verification=False)(state)
    context = json.loads(model.calls[0][0][1].content)
    assert context["evidence"][0]["result"]["table_evidence"]["rows"] == [
        {"quarter": "Q1", "revenue": 100},
        {"quarter": "Q2", "revenue": 125},
    ]


async def test_writer_registers_gsf_citation_identity():
    registry = SourceRegistry()
    token = set_session_registry(registry)
    try:
        await HybridWriter(_Model("Revenue was 100 [1]."), template="Static policy.")(_state())
    finally:
        reset_session_registry(token)
    assert registry.has_citation_key("GSF request revenue-1")


async def test_writer_requires_terminal_finish_and_never_silently_truncates():
    state = _state().model_copy(update={"continuation_history": ()})
    with pytest.raises(ValueError, match="finish decision"):
        await HybridWriter(_Model("unused"), template="Static policy.")(state)
    with pytest.raises(ValueError, match="exceeds"):
        await HybridWriter(_Model("unused"), template="Static policy.", max_input_chars=10)(_state())
