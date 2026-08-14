# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Structured clarification for Hybrid Research."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from typing import Protocol

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage
from langchain_core.runnables import Runnable
from langchain_core.runnables import RunnableConfig
from pydantic import ValidationError

from aiq_agent.common import load_prompt
from aiq_agent.common import render_prompt_template

from .models import ClarificationDecision
from .models import HybridResearchState

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent / "prompts"


class ClarifierError(RuntimeError):
    """Base error for clarifier failures that stop downstream execution."""


class InvalidClarificationDecisionError(ClarifierError):
    """The clarifier returned an invalid decision."""


class ClarifierConfigurationError(ClarifierError):
    """The configured model cannot produce the required structured output."""


class ClarifierTimeoutError(ClarifierError):
    """Clarification assessment exceeded its configured deadline."""


class ClarificationTurnLimitError(ClarifierError):
    """The request remained ambiguous after the configured number of replies."""

    def __init__(self, max_turns: int, decision: ClarificationDecision) -> None:
        missing = ", ".join(decision.missing_dimensions)
        super().__init__(f"The request is still ambiguous after {max_turns} clarification turn(s): {missing}.")


class Clarifier(Protocol):
    """Async boundary for one clarification decision."""

    async def __call__(self, state: HybridResearchState) -> ClarificationDecision | dict[str, Any]:
        """Assess the original question and completed clarification turns."""


def _strict_message_text(response: Any) -> str:
    content = getattr(response, "content", response)
    if not isinstance(content, str):
        raise InvalidClarificationDecisionError("The clarifier correction did not return one JSON text object.")
    return content.strip()


class StructuredClarifier:
    """Return one schema-validated clarification decision from bounded context."""

    def __init__(
        self,
        llm: BaseChatModel,
        *,
        template: str | None = None,
        timeout: float = 90,
        callbacks: Sequence[BaseCallbackHandler] = (),
    ) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be positive")

        self._llm = llm
        self._policy = render_prompt_template(template or load_prompt(_PROMPTS_DIR, "query_clarifier"))
        self._timeout = timeout
        self._callbacks = tuple(callbacks)
        try:
            self._structured_llm: Runnable = llm.with_structured_output(ClarificationDecision)
        except (NotImplementedError, ValueError) as exc:
            raise ClarifierConfigurationError(
                "The configured clarifier model does not support ClarificationDecision structured output."
            ) from exc

        schema = json.dumps(ClarificationDecision.model_json_schema(), separators=(",", ":"), ensure_ascii=False)
        self._correction = (
            "The structured response was missing or invalid. Return exactly one complete JSON object matching this "
            "JSON Schema. Do not include Markdown fences, prefixes, suffixes, or prose:\n"
            f"{schema}"
        )

    def prompt_context(self, state: HybridResearchState) -> dict[str, Any]:
        """Project state into the decision-useful context allowed by policy."""
        catalog = state.catalog_context
        return {
            "current_datetime": state.clarification_reference_datetime.isoformat(timespec="seconds"),
            "original_question": state.question,
            "clarification_history": [turn.model_dump(mode="json") for turn in state.clarification_history],
            "retained_proposed_defaults": state.proposed_defaults,
            "catalog_context": {
                "truncated": catalog.truncated,
                "uncovered_entities": catalog.uncovered_entities,
                "candidates": [
                    {"label": candidate.label, "attribute": candidate.attribute, "term": candidate.term}
                    for candidate in catalog.candidates
                ],
            },
        }

    def messages(self, state: HybridResearchState) -> list[SystemMessage | HumanMessage]:
        """Build a stable policy message and a compact dynamic-context message."""
        context = json.dumps(self.prompt_context(state), ensure_ascii=False, separators=(",", ":"), default=str)
        return [SystemMessage(content=self._policy), HumanMessage(content=context)]

    def _run_config(self) -> RunnableConfig | None:
        return {"callbacks": list(self._callbacks)} if self._callbacks else None

    @staticmethod
    def _validate(value: Any) -> ClarificationDecision:
        if value is None:
            raise InvalidClarificationDecisionError("The clarifier model returned no structured decision.")
        try:
            return ClarificationDecision.model_validate(value)
        except (TypeError, ValueError, ValidationError) as exc:
            raise InvalidClarificationDecisionError("The clarifier model returned an invalid decision.") from exc

    async def _correct_once(self, messages: list[SystemMessage | HumanMessage]) -> ClarificationDecision:
        logger.warning("Retrying ClarificationDecision once as strict JSON text")
        response = await self._llm.ainvoke(
            [*messages, HumanMessage(content=self._correction)],
            config=self._run_config(),
        )
        try:
            return ClarificationDecision.model_validate_json(_strict_message_text(response))
        except (TypeError, ValueError, ValidationError) as exc:
            raise InvalidClarificationDecisionError(
                "The clarifier model did not return one valid ClarificationDecision JSON object."
            ) from exc

    async def __call__(self, state: HybridResearchState) -> ClarificationDecision:
        """Run one structured assessment with one bounded format correction."""
        messages = self.messages(state)
        try:
            async with asyncio.timeout(self._timeout):
                try:
                    response = await self._structured_llm.ainvoke(messages, config=self._run_config())
                    return self._validate(response)
                except (InvalidClarificationDecisionError, OutputParserException, ValidationError):
                    return await self._correct_once(messages)
        except TimeoutError as exc:
            raise ClarifierTimeoutError("The clarifier exceeded its configured timeout.") from exc


__all__ = [
    "Clarifier",
    "ClarifierConfigurationError",
    "ClarifierError",
    "ClarifierTimeoutError",
    "ClarificationTurnLimitError",
    "InvalidClarificationDecisionError",
    "StructuredClarifier",
]
