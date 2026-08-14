# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the bounded structured-analysis worker."""

import asyncio
import json
from types import SimpleNamespace
from typing import Any

import pytest
from deepagents.backends.protocol import ExecuteResponse
from gsf.errors import GSFErrorCode
from gsf.errors import GSFToolError
from gsf.models import TextToSQLResponse
from langchain.agents.middleware.types import ModelResponse
from langchain_core.messages import AIMessage
from pydantic import ValidationError

from aiq_agent.agents.chat_researcher.models import CatalogCandidate
from aiq_agent.agents.chat_researcher.models import CatalogRoutingResponse
from aiq_agent.agents.hybrid_researcher.models import GSFQueryError
from aiq_agent.agents.hybrid_researcher.models import GSFQuerySuccess
from aiq_agent.agents.hybrid_researcher.models import StructuredAnalysisRequest
from aiq_agent.agents.hybrid_researcher.models import StructuredAnalysisResult
from aiq_agent.agents.hybrid_researcher.structured_analysis import StructuredAnalysisError
from aiq_agent.agents.hybrid_researcher.structured_analysis import StructuredAnalysisTimeoutError
from aiq_agent.agents.hybrid_researcher.structured_analysis import StructuredAnalysisWorker
from aiq_agent.agents.hybrid_researcher.structured_analysis import _SequentialGSFMiddleware


class _Provider:
    workdir = "/sandbox/jobs/workflow"

    def __init__(self) -> None:
        self.upload_calls: list[list[tuple[str, bytes]]] = []
        self.execute_calls = []
        self.closed = False
        self.terminated = False

    def upload_files(self, files):
        self.upload_calls.append(files)
        return [SimpleNamespace(path=path, error=None) for path, _content in files]

    def execute(self, command, *, timeout=None):
        self.execute_calls.append((command, timeout))
        return ExecuteResponse(output="derived result", exit_code=0)

    def close(self):
        self.closed = True

    def terminate(self):
        self.terminated = True


def _request() -> StructuredAnalysisRequest:
    return StructuredAnalysisRequest(
        task_id="revenue_analysis",
        objective="Explain revenue change.",
        task_objective="Retrieve the requested period and calculate the change.",
        catalog_context=CatalogRoutingResponse(
            request_id="private-catalog-id",
            coverage=1,
            candidates=[CatalogCandidate(id="private-id", label="Metric", attribute="recognized", term="Revenue")],
        ),
        dependency_results=(),
        database_name="finance",
    )


def _response(question: str = "Revenue by quarter") -> TextToSQLResponse:
    return TextToSQLResponse(
        request_id=question.replace(" ", "-"),
        sql="SELECT quarter, revenue FROM authorized_result",
        rows=[{"quarter": "Q1", "revenue": 100}, {"quarter": "Q2", "revenue": 125}],
        assumptions=["Reported basis"],
    )


async def test_one_useful_gsf_response_returns_compact_provenance(monkeypatch):
    provider = _Provider()
    captured: dict[str, Any] = {}

    async def invoke(request):
        captured["gsf_request"] = request
        return _response()

    class Agent:
        async def ainvoke(self, state, config=None):
            captured["state"] = state
            result = await captured["tools"][0].ainvoke({"question": "Revenue by quarter"})
            captured["projection"] = json.loads(result)
            return {
                "structured_response": {
                    "sufficiency": "sufficient",
                    "content": "Revenue increased from 100 to 125.",
                    "limitations": [],
                }
            }

    def fake_create_agent(**kwargs):
        captured.update(kwargs)
        return Agent()

    monkeypatch.setattr(
        "aiq_agent.agents.hybrid_researcher.structured_analysis.create_agent",
        fake_create_agent,
    )
    worker = StructuredAnalysisWorker(llm=object(), gsf_invoke=invoke, sandbox_factory=lambda _job: provider)
    result = await worker.run(_request())
    assert result.sufficiency == "sufficient"
    assert len(result.gsf_provenance) == 1
    assert isinstance(result.gsf_provenance[0], GSFQuerySuccess)
    assert result.gsf_provenance[0].returned_row_count == 2
    assert captured["gsf_request"].database_name == "finance"
    assert captured["projection"]["rows"] == _response().rows
    assert captured["projection"]["manifest_path"].endswith("/gsf/manifest.json")
    user_context = json.loads(captured["state"]["messages"][0]["content"])
    assert "private-id" not in str(user_context)
    assert "private-catalog-id" not in str(user_context)
    assert [item.name for item in captured["tools"]] == ["query_gsf", "execute_python"]
    assert "within 8000 characters" in captured["system_prompt"]
    assert provider.closed and not provider.terminated


async def test_complete_gsf_payload_stays_worker_local(monkeypatch):
    provider = _Provider()
    captured: dict[str, Any] = {}
    sentinel = "raw-row-value-must-remain-worker-local"
    response = TextToSQLResponse(
        request_id="large-response",
        sql="SELECT private_value FROM authorized_result",
        rows=[{"value": f"{sentinel}-{index}-" + "x" * 250} for index in range(1_000)],
    )

    class Agent:
        async def ainvoke(self, _state, config=None):
            captured["tool_result"] = json.loads(
                await captured["tools"][0].ainvoke({"question": "Return the authorized values"})
            )
            return {
                "structured_response": {
                    "sufficiency": "sufficient",
                    "content": "The requested population contains 1,000 returned rows.",
                    "limitations": [],
                }
            }

    def fake_create_agent(**kwargs):
        captured.update(kwargs)
        return Agent()

    monkeypatch.setattr("aiq_agent.agents.hybrid_researcher.structured_analysis.create_agent", fake_create_agent)
    result = await StructuredAnalysisWorker(
        llm=object(),
        gsf_invoke=lambda _request: asyncio.sleep(0, result=response),
        sandbox_factory=lambda _job: provider,
    ).run(_request())

    uploaded = {path: content for batch in provider.upload_calls for path, content in batch}
    response_path = next(path for path in uploaded if path.endswith("response-1.json"))
    assert sentinel in uploaded[response_path].decode()
    assert sentinel in str(captured["tool_result"]["rows"])
    terminal_json = result.model_dump_json()
    assert sentinel not in terminal_json
    assert '"rows"' not in terminal_json
    assert "authorized_result" not in terminal_json
    assert "manifest" not in terminal_json
    assert result.gsf_provenance[0].returned_row_count == 1_000


async def test_gsf_errors_are_retained_and_do_not_disappear(monkeypatch):
    error = GSFToolError(
        code=GSFErrorCode.UPSTREAM_ERROR,
        retryable=False,
        message="GSF unavailable.",
    )
    captured = {}

    class Agent:
        async def ainvoke(self, _state, config=None):
            captured["tool_result"] = await captured["tools"][0].ainvoke({"question": "Revenue by quarter"})
            return {
                "structured_response": {
                    "sufficiency": "insufficient",
                    "content": "No enterprise rows were available.",
                    "limitations": ["GSF was unavailable."],
                }
            }

    def fake_create_agent(**kwargs):
        captured.update(kwargs)
        return Agent()

    monkeypatch.setattr("aiq_agent.agents.hybrid_researcher.structured_analysis.create_agent", fake_create_agent)
    worker = StructuredAnalysisWorker(
        llm=object(),
        gsf_invoke=lambda _request: asyncio.sleep(0, result=error),
        sandbox_factory=lambda _job: _Provider(),
    )
    result = await worker.run(_request())
    assert isinstance(result.gsf_provenance[0], GSFQueryError)
    assert result.gsf_provenance[0].code == "upstream_error"
    assert json.loads(captured["tool_result"])["code"] == "upstream_error"


async def test_transport_failure_rejects_semantic_rephrasing_without_second_upstream_call(monkeypatch):
    captured = {}
    calls = 0

    async def invoke(_request):
        nonlocal calls
        calls += 1
        return GSFToolError(
            code=GSFErrorCode.TIMEOUT,
            retryable=True,
            message="GSF timed out after internal retries.",
        )

    class Agent:
        async def ainvoke(self, _state, config=None):
            await captured["tools"][0].ainvoke({"question": "Revenue by quarter"})
            blocked = await captured["tools"][0].ainvoke({"question": "Quarterly recognized revenue"})
            assert json.loads(blocked)["code"] == "invalid_request"
            return {
                "structured_response": {
                    "sufficiency": "insufficient",
                    "content": "No enterprise evidence was returned.",
                    "limitations": ["GSF timed out."],
                }
            }

    def fake_create_agent(**kwargs):
        captured.update(kwargs)
        return Agent()

    monkeypatch.setattr("aiq_agent.agents.hybrid_researcher.structured_analysis.create_agent", fake_create_agent)
    result = await StructuredAnalysisWorker(
        llm=object(),
        gsf_invoke=invoke,
        sandbox_factory=lambda _job: _Provider(),
    ).run(_request())
    assert calls == 1
    assert len(result.gsf_provenance) == 2
    assert all(isinstance(item, GSFQueryError) for item in result.gsf_provenance)


async def test_successful_wrong_grain_allows_materially_different_followup(monkeypatch):
    captured = {}
    questions = []

    async def invoke(request):
        questions.append(request.question)
        return _response(request.question)

    class Agent:
        async def ainvoke(self, _state, config=None):
            await captured["tools"][0].ainvoke({"question": "Annual revenue"})
            await captured["tools"][0].ainvoke({"question": "Revenue by fiscal quarter"})
            return {
                "structured_response": {
                    "sufficiency": "sufficient",
                    "content": "The corrected quarterly result supports the analysis.",
                    "limitations": [],
                }
            }

    def fake_create_agent(**kwargs):
        captured.update(kwargs)
        return Agent()

    monkeypatch.setattr("aiq_agent.agents.hybrid_researcher.structured_analysis.create_agent", fake_create_agent)
    result = await StructuredAnalysisWorker(
        llm=object(),
        gsf_invoke=invoke,
        sandbox_factory=lambda _job: _Provider(),
    ).run(_request())
    assert questions == ["Annual revenue", "Revenue by fiscal quarter"]
    assert len(result.gsf_provenance) == 2


async def test_python_requires_success_then_reads_complete_manifest(monkeypatch):
    provider = _Provider()
    captured = {}

    class Agent:
        async def ainvoke(self, _state, config=None):
            with pytest.raises(StructuredAnalysisError, match="until one GSF query succeeds"):
                await captured["tools"][1].ainvoke({"code": "print(1)"})
            await captured["tools"][0].ainvoke({"question": "Revenue by quarter"})
            captured["python"] = json.loads(await captured["tools"][1].ainvoke({"code": "print('derived')"}))
            return {
                "structured_response": {
                    "sufficiency": "sufficient",
                    "content": "Derived comparison complete.",
                    "limitations": [],
                }
            }

    def fake_create_agent(**kwargs):
        captured.update(kwargs)
        return Agent()

    monkeypatch.setattr("aiq_agent.agents.hybrid_researcher.structured_analysis.create_agent", fake_create_agent)
    worker = StructuredAnalysisWorker(
        llm=object(),
        gsf_invoke=lambda _request: asyncio.sleep(0, result=_response()),
        sandbox_factory=lambda _job: provider,
    )
    result = await worker.run(_request())
    uploaded = {path: content for batch in provider.upload_calls for path, content in batch}
    response_path = next(path for path in uploaded if path.endswith("response-1.json"))
    assert json.loads(uploaded[response_path])["rows"] == _response().rows
    assert captured["python"]["output"] == "derived result"
    assert "print('derived')" not in result.model_dump_json()
    assert "derived result" not in result.model_dump_json()
    assert captured["python"]["code_artifact"] not in result.model_dump_json()
    assert captured["python"]["output_artifact"] not in result.model_dump_json()
    assert uploaded[captured["python"]["code_artifact"]] == b"print('derived')"
    assert json.loads(uploaded[captured["python"]["output_artifact"]])["output"] == "derived result"


def test_structured_conclusion_and_provenance_are_bounded_contracts():
    with pytest.raises(ValidationError, match="at most 8000 characters"):
        StructuredAnalysisResult(sufficiency="sufficient", conclusion="x" * 8_001, gsf_provenance=())
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        StructuredAnalysisResult(
            sufficiency="sufficient",
            conclusion="Answer-ready evidence.",
            gsf_provenance=(
                {
                    "status": "success",
                    "question": "Revenue by quarter",
                    "citation_key": "GSF request revenue-1",
                    "returned_row_count": 2,
                    "result_truncated": False,
                    "rows": [{"revenue": 100}],
                },
            ),
        )


async def test_exact_repeated_questions_are_rejected_and_retained(monkeypatch):
    captured = {}
    calls = 0

    async def invoke(_request):
        nonlocal calls
        calls += 1
        return _response()

    class Agent:
        async def ainvoke(self, _state, config=None):
            await captured["tools"][0].ainvoke({"question": "Revenue by quarter"})
            repeated = await captured["tools"][0].ainvoke({"question": "  revenue BY quarter  "})
            assert json.loads(repeated)["code"] == "invalid_request"
            return {
                "structured_response": {
                    "sufficiency": "limited",
                    "content": "The first result was retained.",
                    "limitations": ["Repeated calls are prohibited."],
                }
            }

    def fake_create_agent(**kwargs):
        captured.update(kwargs)
        return Agent()

    monkeypatch.setattr("aiq_agent.agents.hybrid_researcher.structured_analysis.create_agent", fake_create_agent)
    result = await StructuredAnalysisWorker(
        llm=object(),
        gsf_invoke=invoke,
        sandbox_factory=lambda _job: _Provider(),
    ).run(_request())
    assert calls == 1
    assert len(result.gsf_provenance) == 2
    assert isinstance(result.gsf_provenance[1], GSFQueryError)


async def test_multiple_gsf_calls_in_one_model_turn_are_all_rejected():
    middleware = _SequentialGSFMiddleware()
    response = ModelResponse(
        result=[
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "query_gsf", "args": {"question": "One"}, "id": "call-1", "type": "tool_call"},
                    {"name": "query_gsf", "args": {"question": "Two"}, "id": "call-2", "type": "tool_call"},
                ],
            )
        ]
    )
    await middleware.awrap_model_call(SimpleNamespace(), lambda _request: asyncio.sleep(0, result=response))
    handler_called = False

    async def handler(_request):
        nonlocal handler_called
        handler_called = True

    message = await middleware.awrap_tool_call(
        SimpleNamespace(tool_call={"name": "query_gsf", "id": "call-1"}),
        handler,
    )
    assert not handler_called
    assert message.status == "error"
    assert "Multiple query_gsf calls" in message.content


async def test_total_timeout_terminates_sandbox(monkeypatch):
    provider = _Provider()

    class Agent:
        async def ainvoke(self, _state, config=None):
            await asyncio.sleep(1)

    monkeypatch.setattr(
        "aiq_agent.agents.hybrid_researcher.structured_analysis.create_agent",
        lambda **_kwargs: Agent(),
    )
    worker = StructuredAnalysisWorker(
        llm=object(),
        gsf_invoke=lambda _request: asyncio.sleep(0, result=_response()),
        sandbox_factory=lambda _job: provider,
        timeout_seconds=0.01,
    )
    with pytest.raises(StructuredAnalysisTimeoutError, match="0.01-second deadline"):
        await worker.run(_request())
    assert provider.terminated and not provider.closed
