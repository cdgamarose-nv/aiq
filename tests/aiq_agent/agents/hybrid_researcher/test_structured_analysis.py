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
from langchain.agents.middleware.tool_call_limit import ToolCallLimitExceededError
from langchain.agents.middleware.types import ModelResponse
from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages import ToolMessage
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


def _verified_evidence_alignment() -> dict[str, Any]:
    return {
        "sources_and_joins": "verified",
        "grain_and_output_shape": "verified",
        "measures_and_formulas": "verified",
        "filters_and_time_basis": "verified",
        "notes": ["The returned SQL and rows match the synthetic task contract."],
    }


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
                    "evidence_alignment": _verified_evidence_alignment(),
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
    assert '"successful_gsf_responses"' in captured["system_prompt"]
    assert "rows_projection_truncated" in captured["system_prompt"]
    assert "sources and necessary joins" in captured["system_prompt"]
    assert "A plausible\n  row count or polished response is not verification" in captured["system_prompt"]
    assert sandbox_names == []
    assert provider.upload_calls == []
    assert not provider.closed and not provider.terminated


async def test_unresolved_evidence_alignment_downgrades_sufficient_conclusion(monkeypatch):
    captured: dict[str, Any] = {}

    class Agent:
        async def ainvoke(self, _state, config=None):
            await captured["tools"][0].ainvoke({"question": "Revenue by quarter"})
            return {
                "structured_response": {
                    "sufficiency": "sufficient",
                    "content": "The returned rows contain quarterly revenue.",
                    "limitations": [],
                    "evidence_alignment": {
                        **_verified_evidence_alignment(),
                        "measures_and_formulas": "mismatch",
                        "notes": ["The SQL counted records instead of summing the requested revenue measure."],
                    },
                }
            }

    def fake_create_agent(**kwargs):
        captured.update(kwargs)
        return Agent()

    monkeypatch.setattr("aiq_agent.agents.hybrid_researcher.structured_analysis.create_agent", fake_create_agent)
    result = await StructuredAnalysisWorker(
        llm=object(),
        gsf_invoke=lambda _request: asyncio.sleep(0, result=_response()),
        sandbox_factory=lambda _name: _Provider(),
    ).run(_request())

    assert result.sufficiency == "limited"
    assert result.table_evidence is not None
    assert any("measures and formulas" in limitation for limitation in result.limitations)
    assert any("counted records" in limitation for limitation in result.limitations)


async def test_missing_evidence_alignment_recovers_retained_rows(monkeypatch):
    captured: dict[str, Any] = {}

    class Agent:
        async def ainvoke(self, _state, config=None):
            await captured["tools"][0].ainvoke({"question": "Revenue by quarter"})
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

    monkeypatch.setattr("aiq_agent.agents.hybrid_researcher.structured_analysis.create_agent", fake_create_agent)
    result = await StructuredAnalysisWorker(
        llm=object(),
        gsf_invoke=lambda _request: asyncio.sleep(0, result=_response()),
        sandbox_factory=lambda _name: _Provider(),
    ).run(_request())

    assert result.sufficiency == "limited"
    assert result.table_evidence is not None
    assert "did not produce a valid" in result.limitations[0]


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
                    "evidence_alignment": _verified_evidence_alignment(),
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
    assert len(captured["tool_result"]["rows"]) == 100
    assert captured["tool_result"]["rows_projection_truncated"] is True
    terminal_json = result.model_dump_json()
    assert sentinel not in terminal_json
    assert '"rows"' not in terminal_json
    assert "authorized_result" not in terminal_json
    assert "manifest" not in terminal_json
    assert result.gsf_provenance[0].returned_row_count == 1_000


async def test_small_complete_rows_are_preserved_when_model_only_summarizes(monkeypatch):
    captured: dict[str, Any] = {}
    response = TextToSQLResponse(
        sql="SELECT ticker, market_date, volume FROM authorized_result",
        rows=[
            {"ticker": "ETH", "market_date": "06-05-2021", "volume": "87.73K"},
            {"ticker": "BTC", "market_date": "06-05-2021", "volume": "75.20K"},
        ],
    )

    class Agent:
        async def ainvoke(self, _state, config=None):
            await captured["tools"][0].ainvoke({"question": "Return daily volume records"})
            return {
                "structured_response": {
                    "sufficiency": "sufficient",
                    "content": "Two daily volume records were returned.",
                    "limitations": [],
                    "evidence_alignment": _verified_evidence_alignment(),
                }
            }

    def fake_create_agent(**kwargs):
        captured.update(kwargs)
        return Agent()

    monkeypatch.setattr("aiq_agent.agents.hybrid_researcher.structured_analysis.create_agent", fake_create_agent)
    result = await StructuredAnalysisWorker(
        llm=object(),
        gsf_invoke=lambda _request: asyncio.sleep(0, result=response),
        sandbox_factory=lambda _name: _Provider(),
    ).run(_request())
    assert result.table_evidence is not None
    assert result.table_evidence.rows == (
        {"ticker": "ETH", "market_date": "06-05-2021", "volume": "87.73K"},
        {"ticker": "BTC", "market_date": "06-05-2021", "volume": "75.20K"},
    )
    assert "authorized_result" not in result.conclusion


async def test_invalid_terminal_output_recovers_complete_bounded_gsf_rows(monkeypatch):
    captured: dict[str, Any] = {}
    response = TextToSQLResponse(
        sql="SELECT category, item, loss_rate FROM authorized_result",
        rows=[
            {"category": "Leafy", "item": "A", "loss_rate": 4.2},
            {"category": "Leafy", "item": "B", "loss_rate": 3.8},
        ],
    )

    class Agent:
        async def ainvoke(self, _state, config=None):
            await captured["tools"][0].ainvoke({"question": "Top loss-rate items by category"})
            return {"structured_response": None}

    def fake_create_agent(**kwargs):
        captured.update(kwargs)
        return Agent()

    monkeypatch.setattr("aiq_agent.agents.hybrid_researcher.structured_analysis.create_agent", fake_create_agent)
    result = await StructuredAnalysisWorker(
        llm=object(),
        gsf_invoke=lambda _request: asyncio.sleep(0, result=response),
        sandbox_factory=lambda _name: _Provider(),
    ).run(_request())

    assert result.sufficiency == "limited"
    assert result.table_evidence is not None
    assert result.table_evidence.rows == tuple(response.rows)
    assert "did not produce a valid" in result.limitations[0]


async def test_latest_complete_gsf_table_is_preserved_after_corrected_followup(monkeypatch):
    captured: dict[str, Any] = {}
    responses = iter(
        [
            TextToSQLResponse(sql="SELECT item, value FROM detail", rows=[{"item": "A", "value": 1}]),
            TextToSQLResponse(
                sql="SELECT category, total FROM aggregate",
                rows=[{"category": "Leafy", "total": 10}, {"category": "Root", "total": 20}],
            ),
        ]
    )

    class Agent:
        async def ainvoke(self, _state, config=None):
            await captured["tools"][0].ainvoke({"question": "Item detail"})
            await captured["tools"][0].ainvoke({"question": "Category totals"})
            return {
                "structured_response": {
                    "sufficiency": "sufficient",
                    "content": "Two category totals were returned.",
                    "limitations": [],
                    "evidence_alignment": _verified_evidence_alignment(),
                }
            }

    def fake_create_agent(**kwargs):
        captured.update(kwargs)
        return Agent()

    monkeypatch.setattr("aiq_agent.agents.hybrid_researcher.structured_analysis.create_agent", fake_create_agent)
    result = await StructuredAnalysisWorker(
        llm=object(),
        gsf_invoke=lambda _request: asyncio.sleep(0, result=next(responses)),
        sandbox_factory=lambda _name: _Provider(),
    ).run(_request())

    assert result.table_evidence is not None
    assert result.table_evidence.rows == (
        {"category": "Leafy", "total": 10},
        {"category": "Root", "total": 20},
    )
    assert result.table_evidence.citation_key == result.gsf_provenance[-1].citation_key


async def test_invalid_terminal_output_does_not_leak_large_gsf_rows(monkeypatch):
    captured: dict[str, Any] = {}
    sentinel = "large-private-row"
    response = TextToSQLResponse(
        sql="SELECT value FROM authorized_result",
        rows=[{"value": f"{sentinel}-{index}-" + "x" * 250} for index in range(1_000)],
    )

    class Agent:
        async def ainvoke(self, _state, config=None):
            await captured["tools"][0].ainvoke({"question": "Return all values"})
            return {"structured_response": None}

    def fake_create_agent(**kwargs):
        captured.update(kwargs)
        return Agent()

    monkeypatch.setattr("aiq_agent.agents.hybrid_researcher.structured_analysis.create_agent", fake_create_agent)
    result = await StructuredAnalysisWorker(
        llm=object(),
        gsf_invoke=lambda _request: asyncio.sleep(0, result=response),
        sandbox_factory=lambda _name: _Provider(),
    ).run(_request())

    assert result.sufficiency == "limited"
    assert result.table_evidence is None
    assert sentinel not in result.model_dump_json()


async def test_tool_limit_exhaustion_recovers_prior_gsf_evidence(monkeypatch):
    captured: dict[str, Any] = {}

    class Agent:
        async def ainvoke(self, _state, config=None):
            await captured["tools"][0].ainvoke({"question": "Revenue by quarter"})
            raise ToolCallLimitExceededError(
                thread_count=1,
                run_count=4,
                thread_limit=None,
                run_limit=4,
                tool_name="query_gsf",
            )

    def fake_create_agent(**kwargs):
        captured.update(kwargs)
        return Agent()

    monkeypatch.setattr("aiq_agent.agents.hybrid_researcher.structured_analysis.create_agent", fake_create_agent)
    result = await StructuredAnalysisWorker(
        llm=object(),
        gsf_invoke=lambda _request: asyncio.sleep(0, result=_response()),
        sandbox_factory=lambda _name: _Provider(),
    ).run(_request())

    assert result.sufficiency == "limited"
    assert result.table_evidence is not None
    assert result.table_evidence.rows == tuple(_response().rows)


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
                    "evidence_alignment": _verified_evidence_alignment(),
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
                    "evidence_alignment": _verified_evidence_alignment(),
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
                    "evidence_alignment": _verified_evidence_alignment(),
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
                    "evidence_alignment": _verified_evidence_alignment(),
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
    assert captured["python"]["status"] == "success"
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
                    "evidence_alignment": _verified_evidence_alignment(),
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
            await captured["tools"][0].ainvoke({"question": "Revenue by fiscal month"})
            return {
                "structured_response": {
                    "sufficiency": "limited",
                    "content": "The first result was retained.",
                    "limitations": ["Repeated calls are prohibited."],
                    "evidence_alignment": _verified_evidence_alignment(),
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
    assert calls == 2
    assert len(result.gsf_provenance) == 3
    assert isinstance(result.gsf_provenance[1], GSFQueryError)
    assert isinstance(result.gsf_provenance[2], GSFQuerySuccess)


async def test_first_model_step_dispatches_task_objective_to_gsf_without_llm_call():
    middleware = _SequentialGSFMiddleware(initial_question="Final monthly aggregate")
    request = SimpleNamespace(
        messages=[HumanMessage(content="runtime context")],
        tools=[{"name": "query_gsf"}, {"name": "execute_python"}, {"name": "_StructuredConclusion"}],
    )
    handler_called = False

    async def handler(_request):
        nonlocal handler_called
        handler_called = True
        return ModelResponse(result=[AIMessage(content="unexpected")])

    response = await middleware.awrap_model_call(request, handler)
    assert handler_called is False
    assert response.result[0].tool_calls == [
        {
            "name": "query_gsf",
            "args": {"question": "Final monthly aggregate"},
            "id": "call-initial-query-gsf",
            "type": "tool_call",
        }
    ]


async def test_multiple_gsf_calls_in_one_model_turn_keep_first_unseen_call():
    middleware = _SequentialGSFMiddleware()
    first_response = ModelResponse(
        result=[
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "query_gsf", "args": {"question": "One"}, "id": "call-1", "type": "tool_call"},
                ],
            )
        ]
    )
    request = SimpleNamespace(messages=[], tools=[], override=lambda **_kwargs: request)
    await middleware.awrap_model_call(request, lambda _request: asyncio.sleep(0, result=first_response))
    second_response = ModelResponse(
        result=[
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "query_gsf", "args": {"question": "One"}, "id": "call-2", "type": "tool_call"},
                    {"name": "query_gsf", "args": {"question": "Two"}, "id": "call-3", "type": "tool_call"},
                ],
            )
        ]
    )
    filtered = await middleware.awrap_model_call(request, lambda _request: asyncio.sleep(0, result=second_response))
    assert filtered.result[0].tool_calls == [
        {"name": "query_gsf", "args": {"question": "Two"}, "id": "call-3", "type": "tool_call"}
    ]


async def test_data_tool_discards_same_turn_structured_conclusion():
    middleware = _SequentialGSFMiddleware()
    response = ModelResponse(
        result=[
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "query_gsf", "args": {"question": "Revenue"}, "id": "call-1", "type": "tool_call"},
                    {
                        "name": "_StructuredConclusion",
                        "args": {"sufficiency": "sufficient", "content": "Guessed.", "limitations": []},
                        "id": "call-2",
                        "type": "tool_call",
                    },
                ],
            ),
            ToolMessage(
                content='{"sufficiency":"sufficient","content":"Guessed.","limitations":[]}',
                tool_call_id="call-2",
                name="_StructuredConclusion",
            ),
        ],
        structured_response={"sufficiency": "sufficient", "content": "Guessed.", "limitations": []},
    )
    request = SimpleNamespace(messages=[], tools=[], override=lambda **_kwargs: request)
    filtered = await middleware.awrap_model_call(request, lambda _request: asyncio.sleep(0, result=response))
    assert [call["name"] for call in filtered.result[0].tool_calls] == ["query_gsf"]
    assert len(filtered.result) == 1
    assert filtered.structured_response is None


async def test_error_observation_removes_data_tools_from_next_model_call():
    middleware = _SequentialGSFMiddleware()
    request = SimpleNamespace(
        messages=[
            ToolMessage(
                content='{"status":"error","code":"invalid_request"}',
                tool_call_id="call-1",
                name="query_gsf",
            )
        ],
        tools=[{"name": "query_gsf"}, {"name": "execute_python"}, {"name": "_StructuredConclusion"}],
    )
    captured = {}

    def override(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(messages=request.messages, tools=kwargs["tools"])

    request.override = override

    async def handler(updated):
        assert [tool["name"] for tool in updated.tools] == ["_StructuredConclusion"]
        return ModelResponse(result=[AIMessage(content="done")])

    await middleware.awrap_model_call(request, handler)
    assert [tool["name"] for tool in captured["tools"]] == ["_StructuredConclusion"]
    assert captured["tool_choice"] == "_StructuredConclusion"


async def test_complete_but_projected_result_preserves_react_tool_choice():
    middleware = _SequentialGSFMiddleware()
    request = SimpleNamespace(
        messages=[
            ToolMessage(
                content=json.dumps(
                    {
                        "status": "success",
                        "returned_row_count": 76,
                        "rows_projection_truncated": True,
                        "result_truncated": False,
                    }
                ),
                tool_call_id="call-1",
                name="query_gsf",
            )
        ],
        tools=[{"name": "query_gsf"}, {"name": "execute_python"}, {"name": "_StructuredConclusion"}],
    )

    async def handler(updated):
        assert [tool["name"] for tool in updated.tools] == ["query_gsf", "execute_python", "_StructuredConclusion"]
        return ModelResponse(result=[AIMessage(content="done")])

    await middleware.awrap_model_call(request, handler)


async def test_successful_python_forces_structured_conclusion_next():
    middleware = _SequentialGSFMiddleware()
    request = SimpleNamespace(
        messages=[
            ToolMessage(
                content='{"status":"success","exit_code":0,"output":"derived"}',
                tool_call_id="call-1",
                name="execute_python",
            )
        ],
        tools=[{"name": "query_gsf"}, {"name": "execute_python"}, {"name": "_StructuredConclusion"}],
    )
    captured = {}

    def override(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(messages=request.messages, tools=kwargs["tools"])

    request.override = override

    await middleware.awrap_model_call(
        request,
        lambda updated: asyncio.sleep(0, result=ModelResponse(result=[AIMessage(content="done")])),
    )
    assert [tool["name"] for tool in captured["tools"]] == ["_StructuredConclusion"]
    assert captured["tool_choice"] == "_StructuredConclusion"


async def test_failed_python_forces_materially_corrected_python_next():
    middleware = _SequentialGSFMiddleware()
    request = SimpleNamespace(
        messages=[
            ToolMessage(
                content='{"status":"error","code":"execution_failed","exit_code":1,"output":"SyntaxError"}',
                tool_call_id="call-1",
                name="execute_python",
            )
        ],
        tools=[{"name": "query_gsf"}, {"name": "execute_python"}, {"name": "_StructuredConclusion"}],
    )
    captured = {}

    def override(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(messages=request.messages, tools=kwargs["tools"])

    request.override = override

    await middleware.awrap_model_call(
        request,
        lambda updated: asyncio.sleep(0, result=ModelResponse(result=[AIMessage(content="done")])),
    )
    assert [tool["name"] for tool in captured["tools"]] == ["execute_python", "_StructuredConclusion"]
    assert captured["tool_choice"] == "execute_python"


async def test_exact_repeated_python_is_rejected_without_execution(monkeypatch):
    provider = _Provider()
    captured = {}

    class Agent:
        async def ainvoke(self, _state, config=None):
            await captured["tools"][0].ainvoke({"question": "Revenue by quarter"})
            first = json.loads(await captured["tools"][1].ainvoke({"code": "print('derived')"}))
            repeated = json.loads(await captured["tools"][1].ainvoke({"code": "  print('derived')  "}))
            assert first["status"] == "success"
            assert repeated["code"] == "repeated_python"
            return {
                "structured_response": {
                    "sufficiency": "sufficient",
                    "content": "The first calculation established the result.",
                    "limitations": [],
                    "evidence_alignment": _verified_evidence_alignment(),
                }
            }

    def fake_create_agent(**kwargs):
        captured.update(kwargs)
        return Agent()

    monkeypatch.setattr("aiq_agent.agents.hybrid_researcher.structured_analysis.create_agent", fake_create_agent)
    await StructuredAnalysisWorker(
        llm=object(),
        gsf_invoke=lambda _request: asyncio.sleep(0, result=_response()),
        sandbox_factory=lambda _name: provider,
    ).run(_request())
    python_executes = [command for command, _timeout in provider.execute_calls if "python3" in command]
    assert len(python_executes) == 1


async def test_empty_python_output_requires_a_corrected_call(monkeypatch):
    provider = _Provider()
    provider.execute = lambda command, *, timeout=None: (
        provider.execute_calls.append((command, timeout)) or ExecuteResponse(output="  \n", exit_code=0)
    )
    captured = {}

    class Agent:
        async def ainvoke(self, _state, config=None):
            await captured["tools"][0].ainvoke({"question": "Revenue by quarter"})
            result = json.loads(await captured["tools"][1].ainvoke({"code": "import pandas as pd"}))
            assert result["status"] == "error"
            assert result["code"] == "empty_output"
            assert "Print the requested result" in result["message"]
            return {
                "structured_response": {
                    "sufficiency": "limited",
                    "content": "The calculation produced no inspectable output.",
                    "limitations": ["Python did not print an analytical result."],
                    "evidence_alignment": _verified_evidence_alignment(),
                }
            }

    def fake_create_agent(**kwargs):
        captured.update(kwargs)
        return Agent()

    monkeypatch.setattr("aiq_agent.agents.hybrid_researcher.structured_analysis.create_agent", fake_create_agent)
    await StructuredAnalysisWorker(
        llm=object(),
        gsf_invoke=lambda _request: asyncio.sleep(0, result=_response()),
        sandbox_factory=lambda _name: provider,
    ).run(_request())
    assert len([command for command, _timeout in provider.execute_calls if "python3" in command]) == 1


async def test_sufficient_conclusion_is_rejected_when_all_python_attempts_fail(monkeypatch):
    provider = _Provider()
    provider.execute = lambda command, *, timeout=None: (
        provider.execute_calls.append((command, timeout)) or ExecuteResponse(output="", exit_code=0)
    )
    captured = {}

    class Agent:
        async def ainvoke(self, _state, config=None):
            await captured["tools"][0].ainvoke({"question": "Per-customer balances"})
            python_result = json.loads(await captured["tools"][1].ainvoke({"code": "import pandas as pd"}))
            assert python_result["code"] == "empty_output"
            return {
                "structured_response": {
                    "sufficiency": "sufficient",
                    "content": "Python calculated an average of 185.94.",
                    "limitations": [],
                    "evidence_alignment": _verified_evidence_alignment(),
                }
            }

    def fake_create_agent(**kwargs):
        captured.update(kwargs)
        return Agent()

    monkeypatch.setattr("aiq_agent.agents.hybrid_researcher.structured_analysis.create_agent", fake_create_agent)
    result = await StructuredAnalysisWorker(
        llm=object(),
        gsf_invoke=lambda _request: asyncio.sleep(0, result=_response()),
        sandbox_factory=lambda _name: provider,
    ).run(_request())

    assert result.sufficiency == "limited"
    assert "185.94" not in result.conclusion
    assert "required Python work did not succeed" in result.limitations[0]
    assert any("No Python execution completed successfully" in limitation for limitation in result.limitations)


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
