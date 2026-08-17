# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed contracts for the append-only Hybrid Research workflow."""

from __future__ import annotations

import json
import operator
import uuid
from datetime import datetime
from typing import Annotated
from typing import Any
from typing import Literal

from gsf.models import DatabaseName
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import JsonValue
from pydantic import StringConstraints
from pydantic import field_validator
from pydantic import model_validator

from aiq_agent.agents.chat_researcher.models import CatalogRoutingResponse
from aiq_agent.agents.chat_researcher.models import WorkflowClarificationRequired
from aiq_agent.agents.deep_researcher.models import ResearchNotes

TaskKind = Literal["research", "structured_analysis"]
TaskStatus = Literal["succeeded", "failed", "blocked"]
ContinuationAction = Literal["finish", "append", "fail"]
Sufficiency = Literal["sufficient", "limited", "insufficient"]

STRUCTURED_CONCLUSION_MAX_CHARS = 8_000
STRUCTURED_EVIDENCE_MAX_CHARS = 10_000
STRUCTURED_LIMITATION_MAX_CHARS = 2_000
STRUCTURED_MAX_LIMITATIONS = 20
STRUCTURED_TABLE_MAX_COLUMNS = 50
STRUCTURED_TABLE_MAX_ROWS = 100
GSF_QUESTION_MAX_CHARS = 4_096
GSF_PROVENANCE_MAX_COLUMNS = 12
GSF_PROVENANCE_MAX_METADATA_ITEMS = 5
GSF_PROVENANCE_MAX_METADATA_CHARS = 256
GSF_PROVENANCE_MAX_ERROR_CHARS = 1_000
GSF_PROVENANCE_MAX_IDENTIFIER_CHARS = 512

NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
WorkflowRunId = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=128)]
GSFQuestionText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=GSF_QUESTION_MAX_CHARS),
]
StructuredConclusionText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=STRUCTURED_CONCLUSION_MAX_CHARS),
]
StructuredLimitationText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=STRUCTURED_LIMITATION_MAX_CHARS),
]
GSFProvenanceMetadataText = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        max_length=GSF_PROVENANCE_MAX_METADATA_CHARS,
    ),
]
GSFProvenanceErrorText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=GSF_PROVENANCE_MAX_ERROR_CHARS),
]
GSFProvenanceIdentifierText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=GSF_PROVENANCE_MAX_IDENTIFIER_CHARS),
]
NodeId = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        pattern=r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$",
    ),
]


class _StrictModel(BaseModel):
    """Reject fields outside a declared Hybrid Research contract."""

    model_config = ConfigDict(extra="forbid")


class ClarificationDecision(_StrictModel):
    """Small, tolerant decision returned by the Hybrid clarifier."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["ready", "needs_clarification"]
    missing_dimensions: tuple[NonEmptyText, ...] = ()
    clarification_question: NonEmptyText | None = None
    clarified_question: NonEmptyText | None = None
    proposed_defaults: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("missing_dimensions")
    @classmethod
    def deduplicate_dimensions(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        """Preserve the first occurrence of each missing dimension."""
        return tuple(dict.fromkeys(values))

    @model_validator(mode="after")
    def require_selected_route_output(self) -> ClarificationDecision:
        """Require only the value consumed by the selected graph route."""
        if self.status == "ready" and self.clarified_question is None:
            raise ValueError("ready decisions require clarified_question")
        if self.status == "needs_clarification" and self.clarification_question is None:
            raise ValueError("needs_clarification decisions require clarification_question")
        return self


class ClarificationTurn(_StrictModel):
    """One clarification question and the user's reply."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    clarification_question: NonEmptyText
    user_reply: NonEmptyText
    proposed_defaults: dict[str, JsonValue] = Field(default_factory=dict)


class HybridTask(_StrictModel):
    """One coherent worker trajectory in the cumulative task ledger."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: NodeId
    kind: TaskKind
    objective: NonEmptyText
    depends_on: tuple[NodeId, ...] = ()


class HybridTaskPlan(_StrictModel):
    """Initial coarse task graph."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    objective: NonEmptyText
    tasks: tuple[HybridTask, ...] = Field(min_length=1)


class ContinuationDecision(_StrictModel):
    """Append-only decision made after the current task ledger is exhausted."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    action: ContinuationAction
    new_tasks: tuple[HybridTask, ...] = ()
    reasoning: NonEmptyText

    @model_validator(mode="after")
    def validate_action_payload(self) -> ContinuationDecision:
        """Require task additions only for an append decision."""
        if self.action == "append" and not self.new_tasks:
            raise ValueError("append decisions require new_tasks")
        if self.action != "append" and self.new_tasks:
            raise ValueError("new_tasks are only valid for append decisions")
        return self


class ResearchWorkerRequest(_StrictModel):
    """Input accepted by the registered single-query research worker."""

    question: NonEmptyText
    data_sources: list[NonEmptyText] | None = None


class ResearchTaskResult(_StrictModel):
    """Complete structured output from an unstructured research worker."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["research"] = "research"
    notes: ResearchNotes


class GSFResultColumnSummary(_StrictModel):
    """One bounded result-column identity retained after structured analysis."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: GSFProvenanceMetadataText
    data_type: GSFProvenanceMetadataText | None = None


class GSFSemanticContextSummary(_StrictModel):
    """Bounded semantic provenance retained after structured analysis."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    grain: GSFProvenanceMetadataText | None = None
    units: tuple[GSFProvenanceMetadataText, ...] = Field(default=(), max_length=GSF_PROVENANCE_MAX_METADATA_ITEMS)
    filters: tuple[GSFProvenanceMetadataText, ...] = Field(default=(), max_length=GSF_PROVENANCE_MAX_METADATA_ITEMS)
    rules: tuple[GSFProvenanceMetadataText, ...] = Field(default=(), max_length=GSF_PROVENANCE_MAX_METADATA_ITEMS)
    omissions: tuple[GSFProvenanceMetadataText, ...] = Field(
        default=(),
        max_length=GSF_PROVENANCE_MAX_METADATA_ITEMS,
    )


class GSFQuerySuccess(_StrictModel):
    """Compact provenance for one successful GSF query."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["success"] = "success"
    question: GSFQuestionText
    request_id: GSFProvenanceIdentifierText | None = None
    citation_key: GSFProvenanceIdentifierText
    returned_row_count: int = Field(ge=0)
    result_truncated: bool
    columns: tuple[GSFResultColumnSummary, ...] = Field(default=(), max_length=GSF_PROVENANCE_MAX_COLUMNS)
    semantic_context: GSFSemanticContextSummary | None = None
    assumptions: tuple[GSFProvenanceMetadataText, ...] = Field(
        default=(),
        max_length=GSF_PROVENANCE_MAX_METADATA_ITEMS,
    )
    warnings: tuple[GSFProvenanceMetadataText, ...] = Field(
        default=(),
        max_length=GSF_PROVENANCE_MAX_METADATA_ITEMS,
    )
    metadata_truncated: bool = False


class GSFQueryError(_StrictModel):
    """Compact provenance for one failed or rejected GSF query."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["error"] = "error"
    question: GSFQuestionText
    request_id: GSFProvenanceIdentifierText | None = None
    code: NonEmptyText
    retryable: bool
    message: GSFProvenanceErrorText
    metadata_truncated: bool = False


GSFQueryProvenance = Annotated[GSFQuerySuccess | GSFQueryError, Field(discriminator="status")]


class BoundedTableEvidence(_StrictModel):
    """One complete non-truncated GSF result small enough for downstream synthesis."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    citation_key: GSFProvenanceIdentifierText
    columns: tuple[GSFResultColumnSummary, ...] = Field(min_length=1, max_length=STRUCTURED_TABLE_MAX_COLUMNS)
    rows: tuple[dict[str, JsonValue], ...] = Field(min_length=1, max_length=STRUCTURED_TABLE_MAX_ROWS)
    returned_row_count: int = Field(ge=1)

    @model_validator(mode="after")
    def require_complete_row_count(self) -> BoundedTableEvidence:
        """This contract represents all rows from one complete GSF response."""
        if self.returned_row_count != len(self.rows):
            raise ValueError("bounded table evidence must contain every returned row")
        return self


class StructuredAnalysisResult(_StrictModel):
    """Typed conclusion and retained provenance from one structured trajectory."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["structured_analysis"] = "structured_analysis"
    sufficiency: Sufficiency
    conclusion: StructuredConclusionText
    gsf_provenance: tuple[GSFQueryProvenance, ...] = Field(max_length=4)
    limitations: tuple[StructuredLimitationText, ...] = Field(default=(), max_length=STRUCTURED_MAX_LIMITATIONS)
    table_evidence: BoundedTableEvidence | None = None

    @model_validator(mode="after")
    def bound_downstream_evidence(self) -> StructuredAnalysisResult:
        """Bound the combined answer-ready conclusion and optional exact table."""
        table_chars = 0
        if self.table_evidence is not None:
            table_chars = len(
                json.dumps(
                    self.table_evidence.model_dump(mode="json"),
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
        if len(self.conclusion) + table_chars > STRUCTURED_EVIDENCE_MAX_CHARS:
            raise ValueError(
                f"structured conclusion and table evidence exceed {STRUCTURED_EVIDENCE_MAX_CHARS} characters"
            )
        return self


TaskResult = Annotated[
    ResearchTaskResult | StructuredAnalysisResult,
    Field(discriminator="kind"),
]


class TaskRun(_StrictModel):
    """Terminal result of one task execution or dependency transition."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: NodeId
    kind: TaskKind
    status: TaskStatus
    attempts: int = Field(default=0, ge=0)
    result: TaskResult | None = None
    error: NonEmptyText | None = None

    @model_validator(mode="after")
    def validate_outcome(self) -> TaskRun:
        """Keep terminal status and payload shape consistent."""
        if self.status == "succeeded" and self.result is None:
            raise ValueError("A succeeded task run requires a result")
        if self.status != "succeeded" and self.error is None:
            raise ValueError(f"A {self.status} task run requires an error")
        if self.result is not None and self.result.kind != self.kind:
            raise ValueError("result kind must match task kind")
        if self.result is not None and self.status != "succeeded":
            raise ValueError("Failed and blocked task runs cannot include a result")
        if self.error is not None and self.status == "succeeded":
            raise ValueError("Successful task runs cannot include an error")
        return self


class StructuredAnalysisRequest(_StrictModel):
    """Self-contained input for one headless structured-analysis worker."""

    task_id: NodeId
    objective: NonEmptyText
    task_objective: NonEmptyText
    catalog_context: CatalogRoutingResponse
    dependency_results: tuple[TaskRun, ...]
    database_name: DatabaseName | None = None
    workflow_run_id: WorkflowRunId = Field(default_factory=lambda: str(uuid.uuid4()))

    @field_validator("dependency_results")
    @classmethod
    def require_successful_dependencies(cls, runs: tuple[TaskRun, ...]) -> tuple[TaskRun, ...]:
        """Workers only receive complete successful dependency results."""
        if any(run.status != "succeeded" for run in runs):
            raise ValueError("dependency_results must all be successful")
        return runs


class TaskExecutionRequest(_StrictModel):
    """Inputs supplied to one task executor invocation."""

    task: HybridTask
    objective: NonEmptyText
    catalog_context: CatalogRoutingResponse
    dependency_runs: tuple[TaskRun, ...] = ()
    data_sources: list[NonEmptyText] | None = None
    database_name: DatabaseName | None = None
    workflow_run_id: WorkflowRunId = Field(default_factory=lambda: str(uuid.uuid4()))

    @model_validator(mode="after")
    def validate_dependencies(self) -> TaskExecutionRequest:
        """Expose exactly the successful results declared by the task."""
        run_ids = tuple(run.task_id for run in self.dependency_runs)
        if run_ids != self.task.depends_on:
            raise ValueError("dependency_runs must match task.depends_on in declared order")
        if any(run.status != "succeeded" for run in self.dependency_runs):
            raise ValueError("dependency_runs must all be successful")
        return self


class HybridResearchState(_StrictModel):
    """Checkpointed source of truth for one append-only Hybrid Research run."""

    question: NonEmptyText
    catalog_context: CatalogRoutingResponse
    workflow_run_id: WorkflowRunId = Field(default_factory=lambda: str(uuid.uuid4()))
    database_name: DatabaseName | None = None
    data_sources: list[NonEmptyText] | None = None
    skip_clarifier: bool = False
    user_info: dict[str, Any] | None = None
    clarification_reference_datetime: datetime = Field(default_factory=lambda: datetime.now().astimezone())
    clarification_history: tuple[ClarificationTurn, ...] = ()
    proposed_defaults: dict[str, JsonValue] = Field(default_factory=dict)
    clarification_decision: ClarificationDecision | None = None
    clarified_question: NonEmptyText | None = None
    clarification_required: WorkflowClarificationRequired | None = None
    plan: HybridTaskPlan | None = None
    task_runs: Annotated[list[TaskRun], operator.add] = Field(default_factory=list)
    continuation_history: tuple[ContinuationDecision, ...] = ()
    plan_extensions: int = Field(default=0, ge=0)
    final_answer: NonEmptyText | None = None
    terminal_message: NonEmptyText | None = None

    @model_validator(mode="after")
    def validate_workflow_context(self) -> HybridResearchState:
        """Enforce router eligibility and mutually exclusive clarifier outcomes."""
        if not self.catalog_context.candidates:
            raise ValueError("Hybrid Research requires catalog candidates selected by the entry router")
        reference_datetime = self.clarification_reference_datetime
        if reference_datetime.tzinfo is None or reference_datetime.utcoffset() is None:
            raise ValueError("clarification_reference_datetime must be timezone-aware")
        if sum(value is not None for value in (self.clarified_question, self.clarification_required)) > 1:
            raise ValueError("clarifier terminal outcomes are mutually exclusive")
        return self


__all__ = [
    "BoundedTableEvidence",
    "ClarificationDecision",
    "ClarificationTurn",
    "ContinuationAction",
    "ContinuationDecision",
    "GSFQueryError",
    "GSFQueryProvenance",
    "GSFQuerySuccess",
    "GSFResultColumnSummary",
    "GSFSemanticContextSummary",
    "HybridResearchState",
    "HybridTask",
    "HybridTaskPlan",
    "NodeId",
    "NonEmptyText",
    "STRUCTURED_CONCLUSION_MAX_CHARS",
    "STRUCTURED_EVIDENCE_MAX_CHARS",
    "STRUCTURED_TABLE_MAX_COLUMNS",
    "STRUCTURED_TABLE_MAX_ROWS",
    "ResearchTaskResult",
    "ResearchWorkerRequest",
    "StructuredAnalysisRequest",
    "StructuredAnalysisResult",
    "Sufficiency",
    "TaskExecutionRequest",
    "TaskKind",
    "TaskResult",
    "TaskRun",
    "TaskStatus",
    "WorkflowRunId",
]
