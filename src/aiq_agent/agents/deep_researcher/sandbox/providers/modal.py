# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Modal sandbox provider (cloud example)."""

from __future__ import annotations

import logging
import re
import shlex
from typing import TYPE_CHECKING
from typing import Any

from deepagents.backends.protocol import ExecuteResponse
from deepagents.backends.protocol import FileDownloadResponse
from deepagents.backends.protocol import FileUploadResponse
from deepagents.backends.sandbox import BaseSandbox

from ..base import SandboxProvider
from ..capabilities import SandboxCapabilities
from ..logging_utils import log_sandbox_failure
from ..registry import register_sandbox_provider

if TYPE_CHECKING:
    from ..config import SandboxConfig

logger = logging.getLogger(__name__)

_IMPORT_HINT = (
    "The Modal sandbox backend requires the `langchain-modal` and `modal` packages. "
    "Install the updated AIQ dependencies and run `modal setup` before enabling a Modal sandbox."
)


def _validate_modal_sandbox_name(job_id: str) -> str:
    """Validate that ``job_id`` is a legal Modal object name.

    Args:
        job_id: Candidate sandbox name.

    Returns:
        The validated name.

    Raises:
        ValueError: If the name is too long, has illegal characters, or matches a
            reserved Modal app-id shape.
    """
    if len(job_id) > 64 or re.match(r"^[a-zA-Z0-9-_.]+$", job_id) is None or re.match(r"^ap-[a-zA-Z0-9]{22}$", job_id):
        raise ValueError(
            "Deep research job_id must be a valid Modal sandbox name: 64 characters or fewer, using only "
            "alphanumeric characters, dashes, periods, and underscores."
        )
    return job_id


def _is_modal_not_found_error(exc: Exception) -> bool:
    """Return whether ``exc`` is Modal's typed NotFoundError (stale container)."""
    try:
        import modal

        return isinstance(exc, modal.exception.NotFoundError)
    except ImportError:
        return exc.__class__.__name__ == "NotFoundError" and exc.__class__.__module__.startswith("modal")


class _ModalSandbox(BaseSandbox):
    """Deep Agents adapter using Modal's supported filesystem namespace."""

    def __init__(self, sandbox: Any) -> None:
        self._sandbox = sandbox
        self._default_timeout = 30 * 60

    @property
    def id(self) -> str:
        return self._sandbox.object_id

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        effective_timeout = timeout if timeout is not None else self._default_timeout
        process = self._sandbox.exec("bash", "-c", command, timeout=effective_timeout)
        process.wait()
        stdout = process.stdout.read()
        stderr = process.stderr.read()
        output = stdout or ""
        if stderr:
            output += f"\n{stderr}" if output else stderr
        return ExecuteResponse(output=output, exit_code=process.returncode, truncated=False)

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        return [self._upload_file(path, content) for path, content in files]

    def _upload_file(self, path: str, content: bytes) -> FileUploadResponse:
        if not path.startswith("/"):
            return FileUploadResponse(path=path, error="invalid_path")
        try:
            self._sandbox.filesystem.write_bytes(content, path)
        except Exception as exc:  # Modal exposes provider-specific filesystem exception classes
            return FileUploadResponse(path=path, error=_modal_file_error(exc))
        return FileUploadResponse(path=path)

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        return [self._download_file(path) for path in paths]

    def _download_file(self, path: str) -> FileDownloadResponse:
        if not path.startswith("/"):
            return FileDownloadResponse(path=path, error="invalid_path")
        try:
            content = self._sandbox.filesystem.read_bytes(path)
        except Exception as exc:  # Modal exposes provider-specific filesystem exception classes
            return FileDownloadResponse(path=path, error=_modal_file_error(exc))
        return FileDownloadResponse(path=path, content=content)


def _modal_file_error(exc: Exception) -> str:
    """Map Modal filesystem failures to the bounded Deep Agents file contract."""
    try:
        import modal

        if isinstance(exc, modal.exception.SandboxFilesystemPermissionError):
            return "permission_denied"
        if isinstance(exc, modal.exception.SandboxFilesystemIsADirectoryError):
            return "is_directory"
        if isinstance(
            exc,
            (modal.exception.SandboxFilesystemNotFoundError, modal.exception.SandboxFilesystemNotADirectoryError),
        ):
            return "file_not_found"
    except ImportError:
        pass
    raise exc


class ModalSandboxProvider(SandboxProvider):
    """Job-scoped Modal backend.

    Modal enforces network blocking via the ``block_network`` create flag and
    supports deterministic termination, so it declares those capabilities.
    """

    provider_name = "modal"

    def __init__(self, config: SandboxConfig, job_id: str) -> None:
        """Initialize the provider, requiring the Modal SDK and adapter to import."""
        super().__init__(config, job_id)
        try:
            import langchain_modal  # noqa: F401
            import modal  # noqa: F401
        except ImportError as exc:
            raise ImportError(_IMPORT_HINT) from exc

    @classmethod
    def _scoped_name(cls, job_id: str) -> str:
        """Return the validated, job-scoped Modal sandbox name."""
        return _validate_modal_sandbox_name(job_id)

    @property
    def capabilities(self) -> SandboxCapabilities:
        """Declare the guarantees the Modal backend can enforce."""
        return SandboxCapabilities(
            supports_network_policy=True,
            supports_resource_limits=True,
            supports_artifact_download=True,
            supports_cleanup=True,
        )

    def is_recoverable_error(self, exc: Exception) -> bool:
        """Return whether the error is a missing-sandbox condition worth one retry."""
        return _is_modal_not_found_error(exc)

    def _create_session(self) -> BaseSandbox:
        """Create a fresh, job-scoped Modal sandbox.

        Create-first semantics: unlike the legacy backend, this does NOT attach to
        an existing sandbox by name as its primary path (which risked binding a new
        job to a prior job's workspace). It creates fresh; only an
        a typed same-name conflict (this job's own sandbox from earlier in the run,
        since the name is the unique job id) falls back to attach.
        """
        try:
            import modal
        except ImportError as exc:
            raise ImportError(_IMPORT_HINT) from exc

        cfg = self.config
        modal_cfg = cfg.providers.modal
        app = modal.App.lookup(name=modal_cfg.app_name, create_if_missing=True)

        image = modal.Image.from_registry(modal_cfg.image)
        if modal_cfg.python_packages:
            image = image.pip_install(*modal_cfg.python_packages)
        if cfg.workdir:
            image = image.run_commands(f"mkdir -p {shlex.quote(cfg.workdir)}")

        # Opt-in resource caps (None => Modal default). The capability gate has already
        # refused limits on providers that cannot enforce them, so passing them here is safe.
        resource_kwargs: dict[str, object] = {}
        if cfg.resources.cpu is not None:
            resource_kwargs["cpu"] = cfg.resources.cpu
        if cfg.resources.memory_mb is not None:
            resource_kwargs["memory"] = cfg.resources.memory_mb

        try:
            sandbox = modal.Sandbox.create(
                app=app,
                image=image,
                workdir=cfg.workdir,
                name=self.sandbox_name,
                timeout=cfg.timeout,
                idle_timeout=cfg.idle_timeout,
                block_network=cfg.block_network,
                **resource_kwargs,
            )
            logger.info(
                "Modal sandbox CREATED: name=%s image=%s workdir=%s timeout=%ds",
                self.sandbox_name,
                modal_cfg.image,
                cfg.workdir,
                cfg.timeout,
            )
        except (modal.exception.AlreadyExistsError, modal.exception.ConflictError):
            sandbox = modal.Sandbox.from_name(modal_cfg.app_name, self.sandbox_name)
            logger.info("Modal sandbox attached after a same-name create conflict: name=%s", self.sandbox_name)
        return _ModalSandbox(sandbox)

    def _terminate_session(self, session: BaseSandbox | None) -> None:
        """Hard-stop the Modal sandbox wrapped by ``langchain-modal``."""
        if session is None:
            return
        sandbox = getattr(session, "_sandbox", None)
        terminate = getattr(sandbox, "terminate", None)
        if not callable(terminate):
            self._record_cleanup_failure("session_terminate_unavailable")
            logger.error(
                "Modal sandbox termination unavailable: provider=%s sandbox=%s",
                self.provider_name,
                self.sandbox_name,
            )
            return
        try:
            terminate(wait=True)
        except Exception as exc:  # noqa: BLE001 - cleanup must never raise on a terminal path
            self._record_cleanup_failure("session_terminate_failed")
            log_sandbox_failure(
                logger,
                operation="session_terminate",
                reason_code="session_terminate_failed",
                exc=exc,
                provider=self.provider_name,
                sandbox=self.sandbox_name,
            )


register_sandbox_provider("modal", ModalSandboxProvider)
