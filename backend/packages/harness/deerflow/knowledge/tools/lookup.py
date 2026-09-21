"""Agent lookup tools for the Research Knowledge Plane (Phase 2: shared recall).

Implements the KB mid-run read tools — ``knowledge_search``,
``knowledge_get``, ``experiment_get``, ``artifact_manifest`` and
``artifact_open`` — over small storage-boundary protocols that the
integration step binds to PostgreSQL and the object store. This module
performs no I/O of its own; every function takes an explicit backend
except the thin ``@tool`` wrappers, which resolve process-wide bindings
registered via :func:`bind_knowledge_backends` (mirroring the
``set_knowledge_config`` singleton pattern).

Conventions mirrored from the existing knowledge package:

* ``write_api.py`` — eager input validation raising
  :class:`KnowledgeValidationError`, frozen record dataclasses with
  ``to_dict()``, ``Protocol`` storage boundaries with a PG binding sketch
  in the docstring.
* ``search.py`` — oversized ``limit`` values are rejected, never clamped.
* ``agents/memory/tools.py`` — ``@tool`` wrappers take ``runtime`` first,
  resolve user scope from it, and map every failure (including
  ``NotImplementedError`` from unsupported backends) to a JSON-able
  ``{"error": ...}`` payload instead of raising.

Import-cycle contract: importing this module must never pull in
``deerflow.config``. It depends on ``deerflow.knowledge`` record/search
modules, the artifact store value types, ``langchain`` tool plumbing and
``deerflow.runtime``/``deerflow.tools`` runtime helpers only.
"""

import json
import logging
import re
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from langchain.tools import tool

from deerflow.knowledge.artifacts.models import ArtifactNotFoundError, ArtifactRecord
from deerflow.knowledge.artifacts.store import ArtifactStore, parse_artifact_uri
from deerflow.knowledge.search import ExperimentSearchStore
from deerflow.knowledge.write_api import ExperimentRecord, KnowledgeError, KnowledgeValidationError
from deerflow.runtime.user_context import resolve_runtime_user_id
from deerflow.tools.types import Runtime

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_LIMIT",
    "MAX_LIMIT",
    "MAX_IDS_PER_GET",
    "MAX_QUERY_CHARS",
    "MAX_OPEN_BYTES",
    "MAX_OPEN_LINES",
    "MAX_SCAN_BYTES",
    "RETRIEVAL_KINDS",
    "SEARCH_FILTER_FIELDS",
    "RetrievalDocument",
    "RetrievalPage",
    "ArtifactSlice",
    "KnowledgeRetrievalBackend",
    "ExperimentLookupStore",
    "ArtifactReadStore",
    "ArtifactStoreReader",
    "KnowledgeBackends",
    "bind_knowledge_backends",
    "get_knowledge_backends",
    "reset_knowledge_backends",
    "knowledge_search",
    "knowledge_get",
    "experiment_get",
    "artifact_manifest",
    "artifact_open",
    "parse_locator",
    "knowledge_search_tool",
    "knowledge_get_tool",
    "experiment_get_tool",
    "artifact_manifest_tool",
    "artifact_open_tool",
    "get_knowledge_tools",
]

#: Default page size for ``knowledge_search`` when the caller passes no limit.
DEFAULT_LIMIT = 10
#: Hard ceiling for ``knowledge_search`` limits; larger requests are rejected.
MAX_LIMIT = 100
#: Maximum ids accepted by a single ``knowledge_get`` call.
MAX_IDS_PER_GET = 50
#: Maximum query characters accepted by ``knowledge_search``.
MAX_QUERY_CHARS = 2000
#: Maximum bytes returned by a single ``artifact_open`` call.
MAX_OPEN_BYTES = 262_144
#: Maximum text lines returned by a single ``artifact_open`` call.
MAX_OPEN_LINES = 2000
#: Maximum bytes scanned from the start of an artifact while resolving a
#: line-addressed locator. Line addressing past this budget must switch to
#: byte locators (see :func:`artifact_open`).
MAX_SCAN_BYTES = 8_388_608
#: Read chunk size used when streaming artifact bytes for a slice.
_READ_CHUNK_SIZE = 65_536

#: Document kinds addressable through the retrieval backend. ``experiment``
#: documents carry summary text; full experiment rows come from
#: :func:`experiment_get`. ``skill`` is intentionally absent: reusable
#: skills resolve through the bootstrap skill catalog, not mid-run search.
RETRIEVAL_KINDS = frozenset({"finding", "experiment", "failure", "conflict", "assumption", "dossier", "artifact"})

#: Closed vocabulary of structured ``knowledge_search`` filter keys.
#: ``project_id`` scopes to one research project; ``status`` filters the
#: lifecycle state (``validated``/``candidate``/``superseded``/
#: ``rejected``); the remaining keys constrain research scope. Unknown keys
#: are rejected so typos fail loudly instead of silently widening recall.
SEARCH_FILTER_FIELDS = frozenset(
    {
        "project_id",
        "status",
        "asset_class",
        "market",
        "universe",
        "horizon",
        "concept",
        "period_start",
        "period_end",
    }
)

_STATUSES = frozenset({"validated", "candidate", "superseded", "rejected"})
_HASH_RE = re.compile(r"[0-9a-f]{64}")
_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
_LOCATOR_RE = re.compile(r"^(bytes|lines|head|tail):(\d+)(?:-(\d+))?$")


@dataclass(frozen=True)
class RetrievalDocument:
    """One retrieval-backend hit (finding, failure, conflict, dossier, ...).

    Attributes:
        id: Stable document id (``F17``/``E542``/``C31`` style KB ids or
            UUIDs, depending on the backend).
        kind: One of :data:`RETRIEVAL_KINDS`.
        title: Short human-readable title.
        summary: L0/L1 abstract text; full evidence opens via
            :func:`knowledge_get` / :func:`experiment_get` /
            :func:`artifact_open`.
        score: Backend rank score (higher is better); ``None`` when the
            backend does not score (e.g. direct id lookup).
        status: Lifecycle state when known (``validated``/``candidate``/...).
        evidence_refs: Locators the agent may open next (experiment ids,
            artifact URIs, finding ids).
        payload: Backend-specific extra fields (already JSON-safe).
    """

    id: str
    kind: str
    title: str = ""
    summary: str = ""
    score: float | None = None
    status: str | None = None
    evidence_refs: list[str] = field(default_factory=list)
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dict copy of the document."""
        return {
            "id": self.id,
            "kind": self.kind,
            "title": self.title,
            "summary": self.summary,
            "score": self.score,
            "status": self.status,
            "evidence_refs": list(self.evidence_refs),
            "payload": dict(self.payload),
        }


@dataclass(frozen=True)
class RetrievalPage:
    """One page of retrieval results."""

    documents: list[RetrievalDocument] = field(default_factory=list)
    limit: int = DEFAULT_LIMIT
    offset: int = 0
    total: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dict copy of the page."""
        return {
            "documents": [doc.to_dict() for doc in self.documents],
            "count": len(self.documents),
            "limit": self.limit,
            "offset": self.offset,
            "total": self.total,
        }


@dataclass(frozen=True)
class ArtifactSlice:
    """A bounded window of artifact bytes (never the whole multi-GB blob)."""

    data: bytes
    offset: int
    total_size: int

    @property
    def has_more(self) -> bool:
        """True when bytes remain after this window."""
        return self.offset + len(self.data) < self.total_size


@runtime_checkable
class KnowledgeRetrievalBackend(Protocol):
    """Read boundary for hybrid KB retrieval; the integration step binds PG.

    PG binding sketch (Phase 2 KB schema): ``search`` fans out to the
    indexed channels — structured lookup over ``finding``/``experiment``
    scope columns, ``search_document`` TSVECTOR ranking, pgvector cosine
    over ``embedding`` — then fuses candidate ranks (RRF) with the
    research-quality modifiers from the KB "rank fusion" section and
    applies ``LIMIT``/``OFFSET``. ``get_documents`` is a primary-key
    fetch over the same tables (``include_evidence=True`` additionally
    joins ``finding_evidence``/``experiment_artifact`` for evidence rows).
    """

    def search(
        self,
        query: str,
        *,
        kinds: tuple[str, ...],
        filters: Mapping[str, str],
        limit: int,
        offset: int,
    ) -> RetrievalPage:
        """Return one page of fused retrieval hits for ``query``."""
        ...

    def get_documents(self, ids: Sequence[str], *, include_evidence: bool) -> list[RetrievalDocument | None]:
        """Return one entry per requested id, ``None`` for unknown ids (order preserved)."""
        ...


@runtime_checkable
class ExperimentLookupStore(ExperimentSearchStore, Protocol):
    """Experiment read boundary: ``search.py`` queries plus id lookup.

    PG binding sketch: inherits the ``ExperimentSearchStore`` table mapping
    (``experiment`` by ``execution_hash`` / ``experiment_family_hash`` /
    structured ``WHERE``); ``find_by_id`` is a primary-key point lookup on
    ``experiment.id``.
    """

    def find_by_id(self, experiment_id: str) -> ExperimentRecord | None:
        """Return the experiment with this id, or None (primary-key lookup)."""
        ...


@runtime_checkable
class ArtifactReadStore(Protocol):
    """Bounded read boundary for content-addressed artifacts.

    PG/object-store binding sketch: ``get_artifact_record`` reads the
    ``artifact`` table by id or ``artifact://sha256/<digest>`` URI;
    ``read_artifact_slice`` streams a byte window from the object store
    (S3 ``Range`` GET / filesystem seek) without loading the full blob.
    """

    def get_artifact_record(self, ref: str) -> ArtifactRecord:
        """Return metadata for an artifact id, digest, or ``artifact://`` URI.

        Raises:
            KeyError: When no artifact matches ``ref``.
        """
        ...

    def read_artifact_slice(self, ref: str, *, offset: int, length: int) -> ArtifactSlice:
        """Return up to ``length`` bytes starting at ``offset``.

        Raises:
            KeyError: When no artifact matches ``ref``.
        """
        ...


class ArtifactStoreReader:
    """An :class:`ArtifactReadStore` over the existing :class:`ArtifactStore`.

    Resolves ``ref`` as an ``artifact://`` URI or bare SHA-256 digest and
    streams bounded windows through ``ArtifactStore.open`` (seek when the
    backend stream supports it, bounded discard-stream otherwise), so
    locator reads never load a whole multi-GB blob into memory. Integrity
    verification stays enabled: digest mismatches surface when the window
    reaches EOF.
    """

    def __init__(self, store: ArtifactStore) -> None:
        """Create a reader over ``store``."""
        self._store = store

    @property
    def store(self) -> ArtifactStore:
        """The wrapped artifact store."""
        return self._store

    def get_artifact_record(self, ref: str) -> ArtifactRecord:
        """Return the metadata record for ``ref`` (URI or bare digest).

        Raises:
            KeyError: When no artifact matches ``ref`` (translated from
                the store's ``ArtifactNotFoundError`` per the
                :class:`ArtifactReadStore` contract; integrity errors
                propagate unchanged).
        """
        digest = self._digest_for(ref)
        try:
            return self._store.get(digest)
        except ArtifactNotFoundError:
            raise KeyError(ref) from None

    def read_artifact_slice(self, ref: str, *, offset: int, length: int) -> ArtifactSlice:
        """Return up to ``length`` bytes starting at ``offset`` (bounded streaming)."""
        if offset < 0:
            raise KnowledgeValidationError(f"offset must be >= 0, got {offset}.")
        if length < 1:
            raise KnowledgeValidationError(f"length must be >= 1, got {length}.")
        if length > MAX_OPEN_BYTES:
            raise KnowledgeValidationError(f"length {length} exceeds the per-call cap MAX_OPEN_BYTES={MAX_OPEN_BYTES}; narrow the locator.")
        digest = self._digest_for(ref)
        try:
            record = self._store.get(digest)
        except ArtifactNotFoundError:
            raise KeyError(ref) from None
        total = record.byte_size
        if offset >= total:
            return ArtifactSlice(data=b"", offset=offset, total_size=total)
        want = min(length, total - offset)
        try:
            opener = self._store.open(digest)
        except ArtifactNotFoundError:
            raise KeyError(ref) from None
        with opener as stream:
            seek = getattr(stream, "seek", None)
            tell = getattr(stream, "tell", None)
            if callable(seek) and callable(tell):
                try:
                    stream.seek(offset)  # type: ignore[union-attr]
                    if stream.tell() == offset:  # type: ignore[union-attr]
                        return ArtifactSlice(data=self._read_exact(stream, want), offset=offset, total_size=total)
                except (OSError, ValueError):
                    pass  # Fall through to the discard-stream path below.
            discarded = 0
            while discarded < offset:
                chunk = stream.read(min(_READ_CHUNK_SIZE, offset - discarded))
                if not chunk:
                    return ArtifactSlice(data=b"", offset=offset, total_size=total)
                discarded += len(chunk)
            return ArtifactSlice(data=self._read_exact(stream, want), offset=offset, total_size=total)

    @staticmethod
    def _read_exact(stream: Any, want: int) -> bytes:
        """Read up to ``want`` bytes from ``stream`` (short reads allowed at EOF)."""
        buf = bytearray()
        while len(buf) < want:
            chunk = stream.read(min(_READ_CHUNK_SIZE, want - len(buf)))
            if not chunk:
                break
            buf += chunk if isinstance(chunk, (bytes, bytearray)) else bytes(chunk)
        return bytes(buf)

    @staticmethod
    def _digest_for(ref: str) -> str:
        """Normalize ``ref`` to a bare digest, mapping errors to validation errors."""
        if not isinstance(ref, str) or not ref.strip():
            raise KnowledgeValidationError(f"artifact ref must be a non-empty string, got {ref!r}.")
        try:
            return parse_artifact_uri(ref.strip())
        except ValueError as exc:
            raise KnowledgeValidationError(f"invalid artifact ref {ref!r}: {exc}") from None


@dataclass(frozen=True)
class KnowledgeBackends:
    """Process-wide backend bindings for the ``@tool`` wrappers.

    The integration step populates this once real PG/object-store bindings
    exist; until then the wrappers report ``{"error": ...}`` instead of
    raising. Pure functions (:func:`knowledge_search`, ...) always take
    explicit backends and never consult this registry.
    """

    retrieval: KnowledgeRetrievalBackend | None = None
    experiments: ExperimentLookupStore | None = None
    artifacts: ArtifactReadStore | None = None


_knowledge_backends = KnowledgeBackends()


def bind_knowledge_backends(
    *,
    retrieval: KnowledgeRetrievalBackend | None = None,
    experiments: ExperimentLookupStore | None = None,
    artifacts: ArtifactReadStore | None = None,
) -> KnowledgeBackends:
    """Register backend bindings for the ``@tool`` wrappers (integration/tests).

    Only the provided (non-None) bindings are replaced, so callers may bind
    incrementally. Returns the effective bindings.
    """
    global _knowledge_backends
    _knowledge_backends = KnowledgeBackends(
        retrieval=retrieval if retrieval is not None else _knowledge_backends.retrieval,
        experiments=experiments if experiments is not None else _knowledge_backends.experiments,
        artifacts=artifacts if artifacts is not None else _knowledge_backends.artifacts,
    )
    return _knowledge_backends


def get_knowledge_backends() -> KnowledgeBackends:
    """Return the current process-wide backend bindings."""
    return _knowledge_backends


def reset_knowledge_backends() -> None:
    """Clear all backend bindings (tests only)."""
    global _knowledge_backends
    _knowledge_backends = KnowledgeBackends()


def _require_query(query: Any) -> str:
    """Validate a retrieval query string."""
    if not isinstance(query, str):
        raise KnowledgeValidationError(f"query must be a string, got {type(query).__name__}.")
    stripped = query.strip()
    if not stripped:
        raise KnowledgeValidationError("query must be a non-empty string.")
    if len(stripped) > MAX_QUERY_CHARS:
        raise KnowledgeValidationError(f"query exceeds {MAX_QUERY_CHARS} characters ({len(stripped)}).")
    return stripped


def _require_kinds(kinds: Any) -> tuple[str, ...]:
    """Validate an optional kind filter (None/empty means "all kinds")."""
    if kinds is None:
        return tuple(sorted(RETRIEVAL_KINDS))
    if isinstance(kinds, str):
        kinds = [part.strip() for part in kinds.split(",")]
    if not isinstance(kinds, Sequence) or isinstance(kinds, (bytes, bytearray)):
        raise KnowledgeValidationError(f"kinds must be a sequence of kind names, got {type(kinds).__name__}.")
    normalized = [kind.strip().lower() for kind in kinds if isinstance(kind, str) and kind.strip()]
    if not normalized:
        return tuple(sorted(RETRIEVAL_KINDS))
    unknown = [kind for kind in normalized if kind not in RETRIEVAL_KINDS]
    if unknown:
        raise KnowledgeValidationError(f"unknown kinds {unknown}; expected a subset of {sorted(RETRIEVAL_KINDS)}.")
    return tuple(dict.fromkeys(normalized))


def _require_filters(filters: Any) -> dict[str, str]:
    """Validate the structured scope-filter mapping (None becomes {})."""
    if filters is None:
        return {}
    if not isinstance(filters, Mapping):
        raise KnowledgeValidationError(f"filters must be a mapping, got {type(filters).__name__}.")
    unknown = [key for key in filters if key not in SEARCH_FILTER_FIELDS]
    if unknown:
        raise KnowledgeValidationError(f"unknown filter keys {sorted(str(k) for k in unknown)}; expected a subset of {sorted(SEARCH_FILTER_FIELDS)}.")
    normalized: dict[str, str] = {}
    for key, value in filters.items():
        if not isinstance(value, str) or not value.strip():
            raise KnowledgeValidationError(f"filter {key!r} must be a non-empty string, got {value!r}.")
        text = value.strip()
        if key == "project_id":
            try:
                text = str(uuid.UUID(text))
            except ValueError:
                raise KnowledgeValidationError(f"filter 'project_id' must be a valid UUID, got {value!r}.") from None
        elif key == "status" and text.lower() not in _STATUSES:
            raise KnowledgeValidationError(f"filter 'status' must be one of {sorted(_STATUSES)}, got {value!r}.")
        elif key in ("period_start", "period_end") and not _DATE_RE.fullmatch(text):
            raise KnowledgeValidationError(f"filter {key!r} must be YYYY-MM-DD, got {value!r}.")
        normalized[key] = text
    if "period_start" in normalized and "period_end" in normalized and normalized["period_start"] > normalized["period_end"]:
        raise KnowledgeValidationError(f"filter period_start {normalized['period_start']!r} is after period_end {normalized['period_end']!r}.")
    return normalized


def _require_pagination(limit: Any, offset: Any) -> tuple[int, int]:
    """Validate limit/offset (oversized limits are rejected, never clamped)."""
    if not isinstance(limit, int) or isinstance(limit, bool):
        raise KnowledgeValidationError(f"limit must be an int, got {type(limit).__name__}.")
    if not isinstance(offset, int) or isinstance(offset, bool):
        raise KnowledgeValidationError(f"offset must be an int, got {type(offset).__name__}.")
    if limit < 1:
        raise KnowledgeValidationError(f"limit must be >= 1, got {limit}.")
    if limit > MAX_LIMIT:
        raise KnowledgeValidationError(f"limit {limit} exceeds MAX_LIMIT {MAX_LIMIT}.")
    if offset < 0:
        raise KnowledgeValidationError(f"offset must be >= 0, got {offset}.")
    return limit, offset


def _require_ids(ids: Any) -> list[str]:
    """Validate a knowledge_get id list (comma-separated string or sequence)."""
    if isinstance(ids, str):
        ids = [part.strip() for part in ids.split(",")]
    if not isinstance(ids, Sequence) or isinstance(ids, (bytes, bytearray)):
        raise KnowledgeValidationError(f"ids must be a sequence of id strings, got {type(ids).__name__}.")
    normalized = [item.strip() for item in ids if isinstance(item, str) and item.strip()]
    if not normalized:
        raise KnowledgeValidationError("ids must contain at least one non-empty id.")
    if len(normalized) > MAX_IDS_PER_GET:
        raise KnowledgeValidationError(f"ids lists {len(normalized)} entries, exceeding MAX_IDS_PER_GET={MAX_IDS_PER_GET}; split the call.")
    return normalized


def knowledge_search(
    query: str,
    *,
    kinds: Sequence[str] | str | None = None,
    filters: Mapping[str, str] | None = None,
    limit: int = DEFAULT_LIMIT,
    offset: int = 0,
    backend: KnowledgeRetrievalBackend,
) -> dict[str, Any]:
    """Search KB documents by query text plus structured scope filters.

    Args:
        query: Natural-language / lexical query (ticker symbols, factor
            definitions, error strings and research phrases all match).
        kinds: Optional kind filter; ``None``/empty means all
            :data:`RETRIEVAL_KINDS`. Accepts a sequence or comma string.
        filters: Optional structured scope filters (see
            :data:`SEARCH_FILTER_FIELDS`); hard incompatibilities filter
            before ranking, never after.
        limit: Page size (1..:data:`MAX_LIMIT`); oversized values raise.
        offset: Zero-based page offset.
        backend: Retrieval implementation (integration binds PG).

    Returns:
        JSON-safe page dict (``documents``/``count``/``limit``/``offset``/
        ``total`` plus the echoed ``query``/``kinds``/``filters``).

    Raises:
        KnowledgeValidationError: On invalid query, kinds, filters or
            pagination.
    """
    validated_query = _require_query(query)
    validated_kinds = _require_kinds(kinds)
    validated_filters = _require_filters(filters)
    validated_limit, validated_offset = _require_pagination(limit, offset)
    page = backend.search(
        validated_query,
        kinds=validated_kinds,
        filters=validated_filters,
        limit=validated_limit,
        offset=validated_offset,
    )
    payload = page.to_dict()
    payload["query"] = validated_query
    payload["kinds"] = list(validated_kinds)
    payload["filters"] = dict(validated_filters)
    return payload


def knowledge_get(
    ids: Sequence[str] | str,
    *,
    include_evidence: bool = False,
    backend: KnowledgeRetrievalBackend,
) -> dict[str, Any]:
    """Fetch KB documents by id, optionally with their evidence rows.

    Unknown ids do not fail the call: each surfaces as a per-id
    ``{"id": ..., "error": "not_found"}`` entry so a run can cite what
    resolved and report what did not.

    Args:
        ids: One id, a comma-separated string, or a sequence of ids
            (max :data:`MAX_IDS_PER_GET`).
        include_evidence: When true the backend attaches evidence rows
            (finding evidence / experiment links); the citation-lock rule
            (KB: cite only opened evidence) is enforced by the caller/run
            state, not here.
        backend: Retrieval implementation (integration binds PG).

    Returns:
        JSON-safe dict with ``documents`` (resolved hits in request order)
        and ``errors`` (per-id misses in request order).
    """
    if not isinstance(include_evidence, bool):
        raise KnowledgeValidationError(f"include_evidence must be a bool, got {type(include_evidence).__name__}.")
    validated_ids = _require_ids(ids)
    rows = backend.get_documents(validated_ids, include_evidence=include_evidence)
    documents: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for requested, row in zip(validated_ids, list(rows)):
        if row is None:
            errors.append({"id": requested, "error": "not_found"})
        else:
            documents.append(row.to_dict())
    for requested in validated_ids[len(documents) + len(errors) :]:
        errors.append({"id": requested, "error": "not_found"})
    return {"documents": documents, "errors": errors, "include_evidence": include_evidence}


def experiment_get(ref: str, *, store: ExperimentLookupStore) -> dict[str, Any]:
    """Fetch one experiment by id (UUID) or exact execution hash (64-hex).

    The ``ref`` form is auto-detected: a 64-character lowercase hex digest
    takes the exact-rerun lookup path, anything else must be a UUID
    experiment id. Unknown refs raise :class:`ExperimentNotFound`-style
    ``KnowledgeError`` (message ``"Experiment not found: ..."``) so the
    ``@tool`` wrapper can map it to JSON like the memory tools do.

    Args:
        ref: Experiment UUID or execution hash.
        store: Experiment read implementation (integration binds PG).

    Returns:
        JSON-safe dict with the ``experiment`` record, the ``ref`` as
        given, and ``lookup`` (``"id"`` or ``"execution_hash"``).
    """
    if not isinstance(ref, str) or not ref.strip():
        raise KnowledgeValidationError(f"ref must be a non-empty string, got {ref!r}.")
    text = ref.strip()
    if _HASH_RE.fullmatch(text):
        record = store.find_by_execution_hash(text)
        lookup = "execution_hash"
    else:
        try:
            experiment_id = str(uuid.UUID(text))
        except ValueError:
            raise KnowledgeValidationError(f"ref must be an experiment UUID or a 64-char execution hash, got {ref!r}.") from None
        record = store.find_by_id(experiment_id)
        lookup = "id"
    if record is None:
        raise KnowledgeError(f"Experiment not found: {text}")
    return {"experiment": record.to_dict(), "ref": text, "lookup": lookup}


def artifact_manifest(ref: str, *, artifacts: ArtifactReadStore) -> dict[str, Any]:
    """Return the metadata manifest for an artifact (never its bytes).

    Args:
        ref: Artifact id, bare SHA-256 digest, or ``artifact://`` URI.
        artifacts: Bounded artifact reader (integration binds PG + object
            store; :class:`ArtifactStoreReader` adapts the Phase 1 store).

    Returns:
        JSON-safe manifest: canonical ``uri``, ``sha256``, ``kind``,
        ``media_type``, ``byte_size``, ``created_at``, ``metadata``, plus
        ``locator_schemes`` and the ``open_limits`` enforced by
        :func:`artifact_open`.
    """
    record = _resolve_artifact(ref, artifacts)
    return {
        "uri": record.uri,
        "artifact_id": record.artifact_id,
        "sha256": record.sha256,
        "kind": record.kind.value,
        "media_type": record.media_type,
        "byte_size": record.byte_size,
        "created_by_run_id": record.created_by_run_id,
        "created_at": record.created_at.isoformat(),
        "metadata": dict(record.metadata),
        "locator_schemes": [
            "bytes:<offset>-<end> (byte window, inclusive end)",
            "lines:<offset>-<end> (UTF-8 text line window, 0-based, inclusive end)",
            "head:<n> (first n text lines)",
            "tail:<n> (last n text lines)",
        ],
        "open_limits": {
            "max_open_bytes": MAX_OPEN_BYTES,
            "max_open_lines": MAX_OPEN_LINES,
            "max_scan_bytes": MAX_SCAN_BYTES,
        },
    }


def parse_locator(locator: Any) -> dict[str, Any]:
    """Parse an artifact locator into ``{"kind", "offset", "length"}``.

    Accepted forms (mapping or compact string):

    * ``{"kind": "bytes", "offset": 0, "length": 1024}`` — byte window.
    * ``{"kind": "lines", "offset": 0, "length": 50}`` — 0-based line window.
    * ``"bytes:0-1023"`` / ``"lines:10-59"`` — inclusive-end windows.
    * ``"head:20"`` — first 20 lines (``offset`` 0).
    * ``"tail:20"`` — last 20 lines (resolved by scanning, bounded by
      :data:`MAX_SCAN_BYTES`).

    ``offset``/``length`` must be non-negative ints with ``length >= 1``;
    byte windows larger than :data:`MAX_OPEN_BYTES` and line windows
    larger than :data:`MAX_OPEN_LINES` are rejected, never clamped.

    Raises:
        KnowledgeValidationError: On malformed locators or over-budget
            windows.
    """
    if isinstance(locator, str):
        text = locator.strip().lower()
        match = _LOCATOR_RE.fullmatch(text)
        if match is None:
            raise KnowledgeValidationError(f"locator string must match 'bytes:<a>-<b>', 'lines:<a>-<b>', 'head:<n>' or 'tail:<n>', got {locator!r}.")
        scheme, first, second = match.group(1), int(match.group(2)), match.group(3)
        if scheme == "bytes":
            if second is None:
                raise KnowledgeValidationError(f"'bytes:' locators need an inclusive end offset, got {locator!r}.")
            start, end = first, int(second)
            if end < start:
                raise KnowledgeValidationError(f"bytes locator end {end} is before start {start}.")
            return _checked_window("bytes", start, end - start + 1)
        if scheme == "lines":
            if second is None:
                raise KnowledgeValidationError(f"'lines:' locators need an inclusive end line, got {locator!r}.")
            start, end = first, int(second)
            if end < start:
                raise KnowledgeValidationError(f"lines locator end {end} is before start {start}.")
            return _checked_window("lines", start, end - start + 1)
        if second is not None:
            raise KnowledgeValidationError(f"'{scheme}:' locators take a single count, got {locator!r}.")
        return _checked_window(scheme, 0, first)
    if not isinstance(locator, Mapping):
        raise KnowledgeValidationError(f"locator must be a mapping or compact string, got {type(locator).__name__}.")
    kind = locator.get("kind")
    if kind not in ("bytes", "lines", "head", "tail"):
        raise KnowledgeValidationError(f"locator kind must be one of bytes/lines/head/tail, got {kind!r}.")
    offset = locator.get("offset", 0)
    length = locator.get("length")
    if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
        raise KnowledgeValidationError(f"locator offset must be a non-negative int, got {offset!r}.")
    if not isinstance(length, int) or isinstance(length, bool) or length < 1:
        raise KnowledgeValidationError(f"locator length must be an int >= 1, got {length!r}.")
    if kind in ("head", "tail") and offset != 0:
        raise KnowledgeValidationError(f"locator kind {kind!r} requires offset 0, got {offset}.")
    return _checked_window(str(kind), offset, length)


def _checked_window(kind: str, offset: int, length: int) -> dict[str, Any]:
    """Enforce per-call budgets on a parsed locator window."""
    if kind == "bytes" and length > MAX_OPEN_BYTES:
        raise KnowledgeValidationError(f"bytes window of {length} exceeds MAX_OPEN_BYTES={MAX_OPEN_BYTES}; split the read.")
    if kind in ("lines", "head", "tail") and length > MAX_OPEN_LINES:
        raise KnowledgeValidationError(f"line window of {length} exceeds MAX_OPEN_LINES={MAX_OPEN_LINES}; split the read.")
    return {"kind": kind, "offset": offset, "length": length}


def artifact_open(ref: str, locator: Any, *, artifacts: ArtifactReadStore) -> dict[str, Any]:
    """Open a bounded window of an artifact through a locator (no full loads).

    Byte locators stream exactly the requested window; line locators scan
    forward from the start (``tail:`` scans the whole artifact, bounded by
    :data:`MAX_SCAN_BYTES`) and decode UTF-8 incrementally, so a corrupted
    or binary artifact fails with a validation error instead of mojibake.
    Callers needing more than one window issue one call per window.

    Args:
        ref: Artifact id, bare SHA-256 digest, or ``artifact://`` URI.
        locator: Mapping or compact string per :func:`parse_locator`.
        artifacts: Bounded artifact reader.

    Returns:
        JSON-safe dict with ``uri``, ``byte_size``, the normalized
        ``locator``, the ``content`` (text for line locators; UTF-8 text
        with ``encoding: "utf-8"`` — or hex with ``encoding: "hex"`` when
        bytes are not valid UTF-8 — for byte locators), ``bytes_returned``,
        ``has_more`` and the ``open_limits`` in force.

    Raises:
        KnowledgeValidationError: On malformed locators, over-budget
            windows, non-text line reads, or scans past
            :data:`MAX_SCAN_BYTES`.
    """
    window = parse_locator(locator)
    record = _resolve_artifact(ref, artifacts)
    kind = str(window["kind"])
    if kind == "bytes":
        return _open_bytes_window(record, artifacts, int(window["offset"]), int(window["length"]))
    return _open_line_window(record, artifacts, kind, int(window["offset"]), int(window["length"]))


def _resolve_artifact(ref: str, artifacts: ArtifactReadStore) -> ArtifactRecord:
    """Resolve ``ref`` to a record, mapping misses to knowledge errors."""
    if not isinstance(ref, str) or not ref.strip():
        raise KnowledgeValidationError(f"artifact ref must be a non-empty string, got {ref!r}.")
    try:
        return artifacts.get_artifact_record(ref.strip())
    except KeyError:
        raise KnowledgeError(f"Artifact not found: {ref.strip()}") from None


def _open_bytes_window(record: ArtifactRecord, artifacts: ArtifactReadStore, offset: int, length: int) -> dict[str, Any]:
    """Serve a byte-window read as text-or-hex content."""
    try:
        window = artifacts.read_artifact_slice(record.uri, offset=offset, length=length)
    except KeyError:
        raise KnowledgeError(f"Artifact not found: {record.uri}") from None
    try:
        content: str = window.data.decode("utf-8")
        encoding = "utf-8"
    except UnicodeDecodeError:
        content = window.data.hex()
        encoding = "hex"
    return {
        "uri": record.uri,
        "byte_size": window.total_size,
        "locator": {"kind": "bytes", "offset": offset, "length": len(window.data)},
        "encoding": encoding,
        "content": content,
        "bytes_returned": len(window.data),
        "has_more": window.has_more,
        "open_limits": {"max_open_bytes": MAX_OPEN_BYTES, "max_open_lines": MAX_OPEN_LINES, "max_scan_bytes": MAX_SCAN_BYTES},
    }


def _open_line_window(record: ArtifactRecord, artifacts: ArtifactReadStore, kind: str, offset: int, length: int) -> dict[str, Any]:
    """Serve a line-window read by bounded forward scanning from byte 0."""
    if record.byte_size == 0:
        return {
            "uri": record.uri,
            "byte_size": 0,
            "locator": {"kind": kind, "offset": offset, "length": 0},
            "encoding": "utf-8",
            "content": "",
            "lines_returned": 0,
            "has_more": False,
            "open_limits": {"max_open_bytes": MAX_OPEN_BYTES, "max_open_lines": MAX_OPEN_LINES, "max_scan_bytes": MAX_SCAN_BYTES},
        }
    lines: list[str] = []
    pending = bytearray()
    cursor = 0
    complete = False
    try:
        while True:
            if cursor >= MAX_SCAN_BYTES and cursor < record.byte_size:
                raise KnowledgeValidationError(f"line addressing reached the MAX_SCAN_BYTES={MAX_SCAN_BYTES} scan budget; use a bytes: locator for content past this point.")
            try:
                window = artifacts.read_artifact_slice(record.uri, offset=cursor, length=min(_READ_CHUNK_SIZE, MAX_OPEN_BYTES))
            except KeyError:
                raise KnowledgeError(f"Artifact not found: {record.uri}") from None
            if not window.data:
                complete = True
                break
            cursor += len(window.data)
            pending += window.data
            while True:
                newline = pending.find(b"\n")
                if newline == -1:
                    break
                lines.append(pending[:newline].decode("utf-8"))
                del pending[: newline + 1]
                if kind != "tail" and len(lines) >= offset + length:
                    break
            if kind != "tail" and len(lines) >= offset + length:
                break
            if not window.has_more:
                complete = True
                break
    except UnicodeDecodeError as exc:
        raise KnowledgeValidationError(f"artifact {record.uri} is not valid UTF-8 text; use a bytes: locator (decode failed: {exc}).") from None
    if complete and pending:
        try:
            lines.append(bytes(pending).decode("utf-8"))
        except UnicodeDecodeError as exc:
            raise KnowledgeValidationError(f"artifact {record.uri} is not valid UTF-8 text; use a bytes: locator (decode failed: {exc}).") from None
        pending.clear()
    if kind == "tail":
        selected = lines[max(0, len(lines) - length) :]
        selected_offset = max(0, len(lines) - length)
        has_more = False
    else:
        selected_offset = offset
        selected = lines[offset : offset + length]
        # "More" means parsed lines beyond the window, an unparsed
        # remainder in the buffer (early break), or unread bytes.
        has_more = len(lines) > offset + len(selected) or bool(pending) or (not complete and cursor < record.byte_size)
    return {
        "uri": record.uri,
        "byte_size": record.byte_size,
        "locator": {"kind": kind, "offset": selected_offset, "length": len(selected)},
        "encoding": "utf-8",
        "content": "\n".join(selected),
        "lines_returned": len(selected),
        "has_more": has_more,
        "open_limits": {"max_open_bytes": MAX_OPEN_BYTES, "max_open_lines": MAX_OPEN_LINES, "max_scan_bytes": MAX_SCAN_BYTES},
    }


def _resolve_scope(runtime: Runtime | None = None) -> tuple[str | None, str]:
    """Resolve agent_name and user_id for tool handler scope (cf. memory/tools.py)."""
    context = getattr(runtime, "context", None)
    agent_name = None
    if isinstance(context, dict) and context.get("agent_name"):
        agent_name = str(context["agent_name"])
    return agent_name, resolve_runtime_user_id(runtime)


def _scoped_filters(runtime: Runtime | None, filters: dict[str, str] | None) -> dict[str, str]:
    """Merge the run's project scope (when present) under explicit filters."""
    merged = dict(filters or {})
    context = getattr(runtime, "context", None)
    if isinstance(context, dict) and context.get("project_id") and "project_id" not in merged:
        merged["project_id"] = str(context["project_id"])
    return merged


def _tool_error(exc: BaseException, *, log_message: str) -> str:
    """Map a tool failure to a JSON ``{"error": ...}`` payload (cf. memory/tools.py)."""
    if isinstance(exc, NotImplementedError):
        return json.dumps({"error": f"knowledge backend does not support this operation: {exc}"})
    if isinstance(exc, KnowledgeError):
        return json.dumps({"error": str(exc)})
    logger.exception(log_message)
    return json.dumps({"error": str(exc)})


@tool("knowledge_search", parse_docstring=True)
def knowledge_search_tool(
    runtime: Runtime,
    query: str,
    kinds: str | None = None,
    filters_json: str | None = None,
    limit: int = DEFAULT_LIMIT,
    offset: int = 0,
) -> str:
    """Search shared research knowledge by query text plus scope filters.

    Use this during a run to find prior findings, experiments, failures,
    conflicts, assumptions, and dossiers — before designing a backtest and
    again before publishing a conclusion (dup/conflict check).

    Args:
        query: Natural-language or lexical query (tickers, factor names,
            error strings, research phrases).
        kinds: Optional comma-separated kind filter
            (finding,experiment,failure,conflict,assumption,dossier,artifact).
            Omit for all kinds.
        filters_json: Optional JSON object of structured scope filters
            (project_id,status,asset_class,market,universe,horizon,concept,
            period_start,period_end).
        limit: Maximum results to return (default 10, max 100).
        offset: Zero-based page offset (default 0).

    Returns:
        JSON string with "documents" (retrieval hits), "count", paging
        fields, and the echoed query/kinds/filters — or "error".
    """
    _agent_name, _user_id = _resolve_scope(runtime)
    try:
        backend = get_knowledge_backends().retrieval
        if backend is None:
            return json.dumps({"error": "knowledge retrieval backend is not bound"})
        filters: dict[str, str] | None = None
        if filters_json:
            try:
                decoded = json.loads(filters_json)
            except json.JSONDecodeError as exc:
                return json.dumps({"error": f"filters_json is not valid JSON: {exc}"})
            if not isinstance(decoded, dict):
                return json.dumps({"error": "filters_json must decode to a JSON object."})
            filters = decoded
        payload = knowledge_search(
            query,
            kinds=kinds,
            filters=_scoped_filters(runtime, filters),
            limit=limit,
            offset=offset,
            backend=backend,
        )
        return json.dumps(payload, ensure_ascii=False)
    except Exception as exc:
        return _tool_error(exc, log_message="knowledge_search_tool failed")


@tool("knowledge_get", parse_docstring=True)
def knowledge_get_tool(
    runtime: Runtime,
    ids: str,
    include_evidence: bool = False,
) -> str:
    """Fetch KB documents by id, optionally with evidence rows.

    Use this to open specific findings, conflicts, assumptions, or dossiers
    surfaced by knowledge_search or the bootstrap packet. Cite a finding in
    a final conclusion only after opening it here.

    Args:
        ids: One id or comma-separated ids (max 50).
        include_evidence: When true, attach evidence rows to each hit.

    Returns:
        JSON string with "documents" (resolved hits) and "errors"
        (per-id misses) — or "error" for a malformed call.
    """
    _agent_name, _user_id = _resolve_scope(runtime)
    try:
        backend = get_knowledge_backends().retrieval
        if backend is None:
            return json.dumps({"error": "knowledge retrieval backend is not bound"})
        payload = knowledge_get(ids, include_evidence=include_evidence, backend=backend)
        return json.dumps(payload, ensure_ascii=False)
    except Exception as exc:
        return _tool_error(exc, log_message="knowledge_get_tool failed")


@tool("experiment_get", parse_docstring=True)
def experiment_get_tool(runtime: Runtime, ref: str) -> str:
    """Fetch one experiment by id or exact execution hash.

    Use this to open prior experiments from search results or the
    bootstrap packet (methodology, parameters, datasets, code/env
    identity, outcome, metrics).

    Args:
        ref: Experiment UUID id or 64-char execution hash (auto-detected).

    Returns:
        JSON string with the "experiment" record and the "lookup" path
        used ("id" or "execution_hash") — or "error".
    """
    _agent_name, _user_id = _resolve_scope(runtime)
    try:
        store = get_knowledge_backends().experiments
        if store is None:
            return json.dumps({"error": "knowledge experiment store is not bound"})
        payload = experiment_get(ref, store=store)
        return json.dumps(payload, ensure_ascii=False)
    except Exception as exc:
        return _tool_error(exc, log_message="experiment_get_tool failed")


@tool("artifact_manifest", parse_docstring=True)
def artifact_manifest_tool(runtime: Runtime, ref: str) -> str:
    """Return an artifact's metadata manifest (never its bytes).

    Use this to inspect a dataset snapshot, code bundle, result file, or
    log before opening a bounded window with artifact_open.

    Args:
        ref: Artifact id, bare SHA-256 digest, or artifact:// URI.

    Returns:
        JSON string manifest (uri, sha256, kind, media_type, byte_size,
        locator schemes, open limits) — or "error".
    """
    _agent_name, _user_id = _resolve_scope(runtime)
    try:
        artifacts = get_knowledge_backends().artifacts
        if artifacts is None:
            return json.dumps({"error": "knowledge artifact store is not bound"})
        payload = artifact_manifest(ref, artifacts=artifacts)
        return json.dumps(payload, ensure_ascii=False)
    except Exception as exc:
        return _tool_error(exc, log_message="artifact_manifest_tool failed")


@tool("artifact_open", parse_docstring=True)
def artifact_open_tool(runtime: Runtime, ref: str, locator: str) -> str:
    """Open a bounded window of an artifact through a locator.

    Locator forms: "bytes:<a>-<b>" (inclusive byte window),
    "lines:<a>-<b>" (0-based inclusive text lines), "head:<n>",
    "tail:<n>". Reads are capped (256 KiB / 2000 lines per call); the
    full multi-GB blob is never loaded.

    Args:
        ref: Artifact id, bare SHA-256 digest, or artifact:// URI.
        locator: Window selector (see above).

    Returns:
        JSON string with the window "content", "encoding", paging flags,
        and the limits in force — or "error".
    """
    _agent_name, _user_id = _resolve_scope(runtime)
    try:
        artifacts = get_knowledge_backends().artifacts
        if artifacts is None:
            return json.dumps({"error": "knowledge artifact store is not bound"})
        payload = artifact_open(ref, locator, artifacts=artifacts)
        return json.dumps(payload, ensure_ascii=False)
    except Exception as exc:
        return _tool_error(exc, log_message="artifact_open_tool failed")


def get_knowledge_tools() -> list:
    """Return the lookup ``@tool`` wrappers for agent registration.

    Called by the integration step when wiring knowledge tools onto an
    agent (this module registers nothing by itself). The run-start
    ``knowledge_bootstrap`` tool lives in ``bootstrap.py`` next to the
    packet planner.
    """
    return [
        knowledge_search_tool,
        knowledge_get_tool,
        experiment_get_tool,
        artifact_manifest_tool,
        artifact_open_tool,
    ]
