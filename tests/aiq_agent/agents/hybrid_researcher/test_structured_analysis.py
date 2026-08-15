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
        self.events = []
        self.closed = False
        self.terminated = False

    def upload_files(self, files):
        self.events.append(("upload", [path for path, _content in files]))
        self.upload_calls.append(files)
        return [SimpleNamespace(path=path, error=None) for path, _content in files]

    def execute(self, command, *, timeout=None):
        self.events.append(("execute", command))
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
        workflow_run_id="workflow-run-123",
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
    sandbox_names = []

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

    def sandbox_factory(name):
        sandbox_names.append(name)
        return provider

    worker = StructuredAnalysisWorker(llm=object(), gsf_invoke=invoke, sandbox_factory=sandbox_factory)
    result = await worker.run(_request())
    assert result.sufficiency == "sufficient"
    assert len(result.gsf_provenance) == 1
    assert isinstance(result.gsf_provenance[0], GSFQuerySuccess)
    assert result.gsf_provenance[0].returned_row_count == 2
    assert captured["gsf_request"].database_name == "finance"
    assert captured["projection"]["rows"] == _response().rows
    assert "manifest_path" not in captured["projection"]
    user_context = json.loads(captured["state"]["messages"][0]["content"])
    assert "private-id" not in str(user_context)
    assert "private-catalog-id" not in str(user_context)
    assert [item.name for item in captured["tools"]] == ["query_gsf", "execute_python"]
    assert "within 8000 characters" in captured["system_prompt"]
    assert sandbox_names == []
    assert provider.upload_calls == []
    assert not provider.closed and not provider.terminated


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
    sandbox_calls = 0

    def sandbox_factory(_name):
        nonlocal sandbox_calls
        sandbox_calls += 1
        return provider

    result = await StructuredAnalysisWorker(
        llm=object(),
        gsf_invoke=lambda _request: asyncio.sleep(0, result=response),
        sandbox_factory=sandbox_factory,
    ).run(_request())

    assert sandbox_calls == 0
    assert provider.upload_calls == []
    assert sentinel in str(captured["tool_result"]["rows"])
    assert len(captured["tool_result"]["rows"]) == 25
    assert captured["tool_result"]["rows_projection_truncated"] is True
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
    sandbox_names = []

    class Agent:
        async def ainvoke(self, _state, config=None):
            with pytest.raises(StructuredAnalysisError, match="until one GSF query succeeds"):
                await captured["tools"][1].ainvoke({"code": "print(1)"})
            assert sandbox_names == []
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
        sandbox_factory=lambda name: sandbox_names.append(name) or provider,
    )
    result = await worker.run(_request())
    uploaded = {path: content for batch in provider.upload_calls for path, content in batch}
    response_path = next(path for path in uploaded if path.endswith("response-1.json"))
    assert json.loads(uploaded[response_path])["rows"] == _response().rows
    assert provider.events[0][0] == "execute"
    assert provider.events[0][1].startswith("mkdir -p ")
    python_command = next(command for command, _timeout in provider.execute_calls if "python3" in command)
    assert "AIQ_GSF_MANIFEST=" in python_command
    assert captured["python"]["output"] == "derived result"
    assert "print('derived')" not in result.model_dump_json()
    assert "derived result" not in result.model_dump_json()
    assert captured["python"]["code_artifact"] not in result.model_dump_json()
    assert captured["python"]["output_artifact"] not in result.model_dump_json()
    assert uploaded[captured["python"]["code_artifact"]] == b"print('derived')"
    assert json.loads(uploaded[captured["python"]["output_artifact"]])["output"] == "derived result"
    assert provider.terminated and not provider.closed
    assert sandbox_names == ["hybrid-workflow-run-123-revenue_analysis"]


async def test_later_python_call_refreshes_manifest_with_new_gsf_success(monkeypatch):
    provider = _Provider()
    captured = {}

    class Agent:
        async def ainvoke(self, _state, config=None):
            await captured["tools"][0].ainvoke({"question": "Annual revenue"})
            await captured["tools"][1].ainvoke({"code": "print('annual')"})
            await captured["tools"][0].ainvoke({"question": "Quarterly revenue"})
            await captured["tools"][1].ainvoke({"code": "print('quarterly')"})
            return {
                "structured_response": {
                    "sufficiency": "sufficient",
                    "content": "Both complete results were analyzed.",
                    "limitations": [],
                }
            }

    def fake_create_agent(**kwargs):
        captured.update(kwargs)
        return Agent()

    monkeypatch.setattr("aiq_agent.agents.hybrid_researcher.structured_analysis.create_agent", fake_create_agent)
    await StructuredAnalysisWorker(
        llm=object(),
        gsf_invoke=lambda request: asyncio.sleep(0, result=_response(request.question)),
        sandbox_factory=lambda _name: provider,
    ).run(_request())

    manifests = [
        json.loads(content)
        for upload in provider.upload_calls
        for path, content in upload
        if path.endswith("manifest.json")
    ]
    assert [len(manifest["successful_gsf_responses"]) for manifest in manifests] == [1, 2]


async def test_provider_terminates_when_agent_fails_after_python(monkeypatch):
    provider = _Provider()
    captured = {}

    class Agent:
        async def ainvoke(self, _state, config=None):
            await captured["tools"][0].ainvoke({"question": "Revenue by quarter"})
            await captured["tools"][1].ainvoke({"code": "print('derived')"})
            raise RuntimeError("model failed")

    def fake_create_agent(**kwargs):
        captured.update(kwargs)
        return Agent()

    monkeypatch.setattr("aiq_agent.agents.hybrid_researcher.structured_analysis.create_agent", fake_create_agent)
    worker = StructuredAnalysisWorker(
        llm=object(),
        gsf_invoke=lambda _request: asyncio.sleep(0, result=_response()),
        sandbox_factory=lambda _name: provider,
    )
    with pytest.raises(RuntimeError, match="model failed"):
        await worker.run(_request())
    assert provider.terminated and not provider.closed


async def test_provider_terminates_on_timeout_after_python(monkeypatch):
    provider = _Provider()
    captured = {}

    class Agent:
        async def ainvoke(self, _state, config=None):
            await captured["tools"][0].ainvoke({"question": "Revenue by quarter"})
            await captured["tools"][1].ainvoke({"code": "print('derived')"})
            await asyncio.sleep(1)

    def fake_create_agent(**kwargs):
        captured.update(kwargs)
        return Agent()

    monkeypatch.setattr("aiq_agent.agents.hybrid_researcher.structured_analysis.create_agent", fake_create_agent)
    worker = StructuredAnalysisWorker(
        llm=object(),
        gsf_invoke=lambda _request: asyncio.sleep(0, result=_response()),
        sandbox_factory=lambda _name: provider,
        timeout_seconds=0.1,
    )
    with pytest.raises(StructuredAnalysisTimeoutError):
        await worker.run(_request())
    assert provider.terminated and not provider.closed


async def test_provider_terminates_on_cancellation_after_python(monkeypatch):
    provider = _Provider()
    captured = {}
    sandbox_ready = asyncio.Event()

    class Agent:
        async def ainvoke(self, _state, config=None):
            await captured["tools"][0].ainvoke({"question": "Revenue by quarter"})
            await captured["tools"][1].ainvoke({"code": "print('derived')"})
            sandbox_ready.set()
            await asyncio.Event().wait()

    def fake_create_agent(**kwargs):
        captured.update(kwargs)
        return Agent()

    monkeypatch.setattr("aiq_agent.agents.hybrid_researcher.structured_analysis.create_agent", fake_create_agent)
    worker = StructuredAnalysisWorker(
        llm=object(),
        gsf_invoke=lambda _request: asyncio.sleep(0, result=_response()),
        sandbox_factory=lambda _name: provider,
    )
    run_task = asyncio.create_task(worker.run(_request()))
    await asyncio.wait_for(sandbox_ready.wait(), 1)
    run_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run_task
    assert provider.terminated and not provider.closed


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
    sandbox_calls = 0

    def sandbox_factory(_name):
        nonlocal sandbox_calls
        sandbox_calls += 1
        return provider

    worker = StructuredAnalysisWorker(
        llm=object(),
        gsf_invoke=lambda _request: asyncio.sleep(0, result=_response()),
        sandbox_factory=sandbox_factory,
        timeout_seconds=0.01,
    )
    with pytest.raises(StructuredAnalysisTimeoutError, match="0.01-second deadline"):
        await worker.run(_request())
    assert sandbox_calls == 0
    assert not provider.terminated and not provider.closed
