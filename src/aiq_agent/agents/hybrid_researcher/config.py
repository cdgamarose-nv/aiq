# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NAT configuration contracts for hierarchical Hybrid Research."""

from __future__ import annotations

from gsf.models import DatabaseName
from pydantic import ConfigDict
from pydantic import Field
from pydantic import field_validator
from pydantic import model_validator

from aiq_agent.agents.deep_researcher.deepagents_runtime import DeepResearchSandboxConfig
from nat.data_models.component_ref import FunctionGroupRef
from nat.data_models.component_ref import FunctionRef
from nat.data_models.component_ref import LLMRef
from nat.data_models.function import FunctionBaseConfig


def validate_analysis_sandbox(config: DeepResearchSandboxConfig) -> DeepResearchSandboxConfig:
    """Require the isolation and packages assumed by structured analysis."""
    if config.network_mode != "blocked":
        raise ValueError("Hybrid structured analysis requires sandbox network='blocked'")
    if config.provider.lower() == "modal" and "pandas" not in config.packages:
        raise ValueError("Hybrid structured analysis requires pandas in Modal sandbox packages")
    return config


class HybridResearchWorkerConfig(FunctionBaseConfig, name="hybrid_research_worker"):
    """Configuration for the focused unstructured research worker."""

    model_config = ConfigDict(extra="forbid")
    researcher_llm: LLMRef
    tools: list[FunctionRef | FunctionGroupRef] = Field(default_factory=list)
    exclude_tools: list[str] = Field(default_factory=lambda: ["gsf__catalog_search", "gsf__text_to_sql"])
    max_source_tool_calls: int = Field(default=6, ge=1)
    max_concurrent_source_tool_calls: int = Field(default=2, ge=1)
    max_source_tool_batch_size: int = Field(default=2, ge=1)
    timeout_seconds: float = Field(default=180, gt=0)
    verbose: bool = False


class StructuredAnalysisWorkerConfig(FunctionBaseConfig, name="structured_analysis_worker"):
    """Configuration for the headless enterprise structured-analysis worker."""

    model_config = ConfigDict(extra="forbid")
    structured_analysis_llm: LLMRef
    sql_tool: FunctionRef
    sandbox: DeepResearchSandboxConfig | FunctionRef
    database_name: DatabaseName | None = None
    sql_max_rows: int = Field(default=1_000, ge=1)
    structured_max_gsf_calls: int = Field(default=4, ge=1, le=4)
    structured_max_python_calls: int = Field(default=4, ge=1, le=4)
    structured_analysis_timeout_seconds: float = Field(default=600, gt=0)
    python_execute_timeout_seconds: int = Field(default=60, ge=1)
    max_code_chars: int = Field(default=40_000, ge=1)
    max_output_chars: int = Field(default=40_000, ge=1)
    model_result_rows: int = Field(default=25, ge=1)
    verbose: bool = False

    @field_validator("sandbox", mode="before")
    @classmethod
    def parse_inline_sandbox(cls, value):
        if isinstance(value, dict):
            return DeepResearchSandboxConfig.model_validate(value)
        return value

    @model_validator(mode="after")
    def validate_inline_sandbox(self) -> StructuredAnalysisWorkerConfig:
        if isinstance(self.sandbox, DeepResearchSandboxConfig):
            validate_analysis_sandbox(self.sandbox)
        return self


class HybridResearchAgentConfig(FunctionBaseConfig, name="hybrid_research_agent"):
    """Configuration for outer Hybrid Research orchestration."""

    model_config = ConfigDict(extra="forbid")
    clarifier_llm: LLMRef | None = None
    planner_llm: LLMRef
    writer_llm: LLMRef
    research_worker: FunctionRef
    structured_analysis_worker: FunctionRef
    database_name: DatabaseName | None = None
    enable_clarifier: bool = True
    max_clarification_turns: int = Field(default=3, ge=1, le=10)
    max_parallel_tasks: int = Field(default=4, ge=1)
    max_total_tasks: int = Field(default=12, ge=1, le=50)
    max_plan_extensions: int = Field(default=2, ge=0, le=10)
    llm_timeout: float = Field(default=90, gt=0)
    planner_timeout_seconds: float = Field(default=90, gt=0)
    continuation_timeout_seconds: float = Field(default=90, gt=0)
    writer_timeout_seconds: float = Field(default=120, gt=0)
    writer_max_input_chars: int = Field(default=200_000, ge=1)
    enable_citation_verification: bool = True
    checkpoint_db: str = Field(default="./checkpoints.db", min_length=1)
    verbose: bool = False

    @model_validator(mode="after")
    def require_clarifier_llm_when_enabled(self) -> HybridResearchAgentConfig:
        if self.enable_clarifier and self.clarifier_llm is None:
            raise ValueError("clarifier_llm is required when enable_clarifier is true")
        return self


__all__ = [
    "HybridResearchAgentConfig",
    "HybridResearchWorkerConfig",
    "StructuredAnalysisWorkerConfig",
    "validate_analysis_sandbox",
]
