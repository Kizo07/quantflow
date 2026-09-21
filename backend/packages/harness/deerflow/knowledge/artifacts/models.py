"""Artifact record model for the Quantflow Research Knowledge Plane.

Every significant research output (dataset snapshot, code bundle, notebook,
result file, log, chart, environment lock) becomes an immutable artifact
addressed by the SHA-256 digest of its bytes::

    artifact://sha256/<digest>

A modified file always receives a **new** artifact; bytes are never mutated
in place. This module defines the record shape from the knowledge-base
contract plus the error taxonomy shared by backends and the store.
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any


class ArtifactKind(StrEnum):
    """Logical kind of an artifact, per the knowledge-base contract."""

    DATASET_SNAPSHOT = "dataset_snapshot"
    SOURCE = "source"
    CODE = "code"
    NOTEBOOK = "notebook"
    RESULT = "result"
    LOG = "log"
    CHART = "chart"
    ENVIRONMENT = "environment"


#: Canonical URI scheme prefix for content-addressed artifacts.
ARTIFACT_URI_SCHEME = "artifact://sha256/"


def artifact_uri(digest: str) -> str:
    """Return the canonical ``artifact://`` URI for a hex SHA-256 digest."""
    return f"{ARTIFACT_URI_SCHEME}{digest}"


class ArtifactError(Exception):
    """Base error for all artifact-store failures."""


class ArtifactNotFoundError(ArtifactError):
    """Raised when no artifact exists for a digest/artifact id."""

    def __init__(self, identifier: str) -> None:
        super().__init__(f"artifact not found: {identifier}")
        self.identifier = identifier


class ArtifactIntegrityError(ArtifactError):
    """Raised when stored bytes do not re-hash to the recorded digest."""

    def __init__(self, digest: str, actual: str | None = None) -> None:
        detail = f"expected sha256={digest}"
        if actual is not None:
            detail += f" but recomputed sha256={actual}"
        super().__init__(f"artifact integrity check failed: {detail}")
        self.expected_digest = digest
        self.actual_digest = actual


class ArtifactBackendError(ArtifactError):
    """Raised when the underlying object backend fails an operation."""

    def __init__(self, operation: str, cause: BaseException) -> None:
        super().__init__(f"artifact backend {operation} failed: {cause}")
        self.operation = operation
        self.cause = cause


@dataclass(frozen=True)
class ArtifactRecord:
    """Immutable metadata record for one stored artifact.

    Attributes:
        artifact_id: Stable UUID for this record (distinct from the digest so
            re-registration of identical bytes keeps referential identity).
        sha256: Hex SHA-256 digest of the artifact bytes; the content address.
        kind: Logical artifact kind.
        storage_uri: Backend-specific locator (e.g. ``s3://bucket/key`` or a
            ``file://`` path). The canonical content URI is available via
            :pyattr:`uri`.
        media_type: MIME type of the bytes.
        byte_size: Exact byte length of the artifact.
        created_by_run_id: Run that produced the artifact, if known.
        created_at: UTC timestamp of registration (timezone-aware).
        metadata: Caller-supplied free-form metadata (must be JSON-serializable).
    """

    artifact_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sha256: str = ""
    kind: ArtifactKind = ArtifactKind.RESULT
    storage_uri: str = ""
    media_type: str = "application/octet-stream"
    byte_size: int = 0
    created_by_run_id: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.sha256:
            raise ValueError("sha256 digest must not be empty")
        if len(self.sha256) != 64 or any(c not in "0123456789abcdef" for c in self.sha256.lower()):
            raise ValueError(f"sha256 must be 64 lowercase hex chars, got {self.sha256!r}")
        if self.byte_size < 0:
            raise ValueError(f"byte_size must be >= 0, got {self.byte_size}")
        try:
            uuid.UUID(self.artifact_id)
        except ValueError as exc:
            raise ValueError(f"artifact_id must be a UUID, got {self.artifact_id!r}") from exc

    @property
    def uri(self) -> str:
        """Canonical content-addressed URI (``artifact://sha256/<digest>``)."""
        return artifact_uri(self.sha256)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the record to a JSON-compatible dict."""
        payload: dict[str, Any] = asdict(self)
        payload["kind"] = self.kind.value
        payload["created_at"] = self.created_at.isoformat()
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ArtifactRecord:
        """Deserialize a record produced by :meth:`to_dict`.

        Raises:
            ValueError: If required fields are missing or malformed.
        """
        try:
            kind = ArtifactKind(payload["kind"])
        except (KeyError, ValueError) as exc:
            raise ValueError(f"invalid artifact kind: {payload.get('kind')!r}") from exc
        try:
            created_at = datetime.fromisoformat(str(payload.get("created_at", "")))
        except (ValueError, TypeError) as exc:
            raise ValueError(f"invalid created_at: {payload.get('created_at')!r}") from exc
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=UTC)
        metadata = payload.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError(f"metadata must be an object, got {type(metadata).__name__}")
        return cls(
            artifact_id=str(payload.get("artifact_id", str(uuid.uuid4()))),
            sha256=str(payload["sha256"]),
            kind=kind,
            storage_uri=str(payload.get("storage_uri", "")),
            media_type=str(payload.get("media_type", "application/octet-stream")),
            byte_size=int(payload["byte_size"]),
            created_by_run_id=payload.get("created_by_run_id"),
            created_at=created_at,
            metadata=dict(metadata),
        )
