# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for hierarchical Hybrid NAT configuration."""

import pytest
from pydantic import ValidationError

from aiq_agent.agents.hybrid_researcher.config import HybridResearchAgentConfig
from aiq_agent.agents.hybrid_researcher.config import StructuredAnalysisWorkerConfig


def test_outer_config_uses_task_and_extension_limits_without_reviewer_fields():
    config = HybridResearchAgentConfig(
        enable_clarifier=False,
        planner_llm="planner",
        writer_llm="writer",
        research_worker="research",
        structured_analysis_worker="structured",
    )
    assert config.max_total_tasks == 12
    assert config.max_parallel_tasks == 4
    assert config.max_plan_extensions == 2
    assert config.continuation_timeout_seconds == 90
    assert "reviewer_llm" not in type(config).model_fields
    assert "analysis_llm" not in type(config).model_fields
    assert "max_plan_revisions" not in type(config).model_fields


def test_structured_worker_defaults_and_sandbox_guards():
    config = StructuredAnalysisWorkerConfig(
        structured_analysis_llm="structured",
        sql_tool="gsf__text_to_sql",
        sandbox={"provider": "modal", "packages": ["pandas"], "network": "blocked"},
    )
    assert config.structured_max_gsf_calls == 4
    assert config.structured_max_python_calls == 4
    assert config.structured_analysis_timeout_seconds == 600
    assert config.python_execute_timeout_seconds == 60
    with pytest.raises(ValidationError, match="network='blocked'"):
        StructuredAnalysisWorkerConfig(
            structured_analysis_llm="structured",
            sql_tool="gsf__text_to_sql",
            sandbox={"provider": "modal", "packages": ["pandas"], "network": "open"},
        )


def test_enabled_clarifier_requires_llm_and_limits_are_positive():
    with pytest.raises(ValidationError, match="clarifier_llm"):
        HybridResearchAgentConfig(
            planner_llm="planner",
            writer_llm="writer",
            research_worker="research",
            structured_analysis_worker="structured",
        )
    with pytest.raises(ValidationError):
        HybridResearchAgentConfig(
            enable_clarifier=False,
            planner_llm="planner",
            writer_llm="writer",
            research_worker="research",
            structured_analysis_worker="structured",
            max_total_tasks=0,
        )
