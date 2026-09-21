"""Text-embedding provider interface + deterministic test fake (Phase 2: shared recall).

Implements the KB vector-retrieval leg (``knowledge_base.md`` § "Retrieval
should be hybrid, scoped, and evidence-aware" and § "Proposed canonical
schema"): findings/experiments/dossiers carry an ``embedding VECTOR(768)``
column beside the relational metadata and ``search_document`` TSVECTOR, and
background workers generate vectors asynchronously via a transactional
outbox (``knowledge_base.md`` § "Writes should separate ... outbox event";
``implementation_plan.md`` Phase 6).

Storage boundary: this module performs no I/O of its own. Embedding
*computation* goes through :class:`EmbeddingProvider`; embedding *persistence*
goes through :class:`EmbeddingVectorStore` (bound to PostgreSQL/pgvector by
the integration step — see the Protocol docstrings for the column mapping).
The :func:`backfill_findings_embeddings` helper drives finding-id -> vector
upserts across those two boundaries.

Canonical dimension: :data:`EMBEDDING_DIM` (768). Every vector this package
produces, validates, or stores is 768 floats; providers reporting any other
dimension are rejected at the persistence boundary, never silently
truncated or padded.

Real-model status: the :mod:`deerflow.knowledge.providers_st` module wires
``sentence-transformers/all-mpnet-base-v2`` (768-d) through
:func:`register_provider_factory` (optional ``knowledge-st`` extra; cached
checkpoint required, never downloaded implicitly). This module ships the
provider interface, the deterministic test fake, the batch API, and the
backfill helper. There is deliberately no provider module that pretends to
be a real model: the fake stamps ``test-fake/v1`` on every vector it makes.
"""

from __future__ import annotations

import hashlib
import math
import os
import struct
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from deerflow.knowledge.hashing import normalize_text
from deerflow.knowledge.write_api import KnowledgeError, KnowledgeValidationError

#: Canonical embedding dimension. Must match the ``VECTOR(768)`` column the
#: Phase 2 schema migration defines for finding/experiment embeddings.
EMBEDDING_DIM = 768

#: Model id of the deterministic test fake (see :class:`DeterministicEmbeddingProvider`).
FAKE_MODEL_ID = "test-fake/v1"

#: Domain-separation prefix for fake-vector derivation (mirrors ``hashing`` domains).
FAKE_DOMAIN = "quantflow.embedding.fake/v1"

#: Environment variable naming the embedding model the integration step wires in.
ENV_MODEL = "DEER_FLOW_KNOWLEDGE_EMBEDDING_MODEL"

#: Default/maximum texts per provider call and per backfill batch.
DEFAULT_BATCH_SIZE = 64
MAX_BATCH_SIZE = 512

#: Maximum characters accepted for a single embeddable text (matches the
#: write-API prose ceiling so finding bodies always fit).
MAX_TEXT_CHARS = 100_000

__all__ = [
    "EMBEDDING_DIM",
    "FAKE_MODEL_ID",
    "FAKE_DOMAIN",
    "ENV_MODEL",
    "DEFAULT_BATCH_SIZE",
    "MAX_BATCH_SIZE",
    "MAX_TEXT_CHARS",
    "EmbeddingError",
    "EmbeddingProviderError",
    "EmbeddingProviderUnavailableError",
    "EmbeddingProvider",
    "DeterministicEmbeddingProvider",
    "FindingTextSource",
    "EmbeddingVectorStore",
    "BackfillFailure",
    "BackfillResult",
    "embed_texts",
    "render_embeddable_text",
    "register_provider_factory",
    "load_provider",
    "backfill_findings_embeddings",
]


class EmbeddingError(KnowledgeError):
    """Base class for all embedding-plane errors."""


class EmbeddingProviderError(EmbeddingError):
    """Raised when a provider fails or returns malformed vectors.

    Returned vectors are validated fail-closed: wrong length, non-finite
    floats, or non-numeric entries always raise rather than flowing into
    the vector store.
    """


class EmbeddingProviderUnavailableError(EmbeddingError):
    """Raised when no provider is registered for the requested model id.

    See :func:`load_provider` for the wiring instructions included in the
    error message.
    """


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Computation boundary for text embeddings; integration binds real models here.

    A provider turns research prose into fixed-dimension float vectors. The
    canonical dimension is :data:`EMBEDDING_DIM`; providers reporting any
    other dimension are refused at the persistence boundary.

    Implementations must be deterministic for a fixed ``model_id`` (same
    text -> same vector across processes and machines) so vectors stay
    comparable across backfills and worker restarts. Thread-safety follows
    the store convention: instances shared across threads must synchronize
    internally.
    """

    @property
    def model_id(self) -> str:
        """Stable model identifier recorded beside every stored vector."""
        ...

    @property
    def dimension(self) -> int:
        """Vector dimension this provider emits (canonical: 768)."""
        ...

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed ``texts`` in order; returns one vector per input text.

        Raises:
            KnowledgeValidationError: On invalid input texts.
            EmbeddingProviderError: When the underlying model fails.
        """
        ...

    def embed_one(self, text: str) -> list[float]:
        """Embed a single text (default: one-element :meth:`embed_batch`)."""
        return self.embed_batch([text])[0]


def _require_text(name: str, value: object) -> str:
    """Validate one embeddable text (non-empty, bounded, NFC/whitespace-normalized)."""
    if not isinstance(value, str):
        raise KnowledgeValidationError(f"{name} must be a string, got {type(value).__name__}.")
    if len(value) > MAX_TEXT_CHARS:
        raise KnowledgeValidationError(f"{name} exceeds {MAX_TEXT_CHARS} characters ({len(value)}).")
    normalized = normalize_text(value)
    if not normalized:
        raise KnowledgeValidationError(f"{name} must be a non-empty string.")
    return normalized


def _require_batch_size(batch_size: object) -> int:
    """Validate a batch size (1..MAX_BATCH_SIZE, never silently clamped)."""
    if not isinstance(batch_size, int) or isinstance(batch_size, bool):
        raise KnowledgeValidationError(f"batch_size must be an int, got {type(batch_size).__name__}.")
    if batch_size < 1:
        raise KnowledgeValidationError(f"batch_size must be >= 1, got {batch_size}.")
    if batch_size > MAX_BATCH_SIZE:
        raise KnowledgeValidationError(f"batch_size {batch_size} exceeds MAX_BATCH_SIZE {MAX_BATCH_SIZE}.")
    return batch_size


def _check_vector(vector: object, *, dimension: int, model_id: str, index: int) -> list[float]:
    """Fail-closed validation of one provider-returned vector."""
    where = f"vector {index} from provider {model_id!r}"
    if not isinstance(vector, Sequence) or isinstance(vector, (str, bytes, bytearray)):
        raise EmbeddingProviderError(f"{where} is not a float sequence (got {type(vector).__name__}).")
    values = list(vector)
    if len(values) != dimension:
        raise EmbeddingProviderError(f"{where} has dimension {len(values)}, expected {dimension}.")
    checked: list[float] = []
    for position, entry in enumerate(values):
        if not isinstance(entry, (int, float)) or isinstance(entry, bool):
            raise EmbeddingProviderError(f"{where} entry {position} is not a float (got {type(entry).__name__}).")
        number = float(entry)
        if not math.isfinite(number):
            raise EmbeddingProviderError(f"{where} entry {position} is not finite ({entry!r}).")
        checked.append(number)
    return checked


class DeterministicEmbeddingProvider:
    """Deterministic, dependency-free test fake with stable 768-d vectors.

    Each vector derives from ``SHA256(FAKE_DOMAIN || 0x00 || normalized text)``
    expanded in counter mode (96 digests x 8 uint32 words) and L2-normalized,
    so the same text always yields the same unit vector on any machine with
    any Python version — no ``random`` module, no ``hash()``, no BLAS.

    These vectors carry no semantic content: they exist so retrieval,
    backfill, and pgvector plumbing tests run offline with realistic shapes
    (dimension, dtype, unit norm). They must never be presented as real
    semantic embeddings; stored rows record ``model_id="test-fake/v1"`` so
    test vectors are trivially distinguishable from production ones.
    """

    def __init__(self, *, dimension: int = EMBEDDING_DIM, model_id: str = FAKE_MODEL_ID) -> None:
        """Create the fake.

        Args:
            dimension: Vector dimension (default canonical 768; other values
                exist only so tests can exercise dimension-mismatch guards).
            model_id: Model id stamped on stored rows (default ``test-fake/v1``).

        Raises:
            KnowledgeValidationError: On a non-positive dimension or an
                empty model id.
        """
        if not isinstance(dimension, int) or isinstance(dimension, bool) or dimension < 1:
            raise KnowledgeValidationError(f"dimension must be a positive int, got {dimension!r}.")
        if not isinstance(model_id, str) or not model_id.strip():
            raise KnowledgeValidationError("model_id must be a non-empty string.")
        self._dimension = dimension
        self._model_id = model_id.strip()

    @property
    def model_id(self) -> str:
        """Return the model id stamped on stored rows."""
        return self._model_id

    @property
    def dimension(self) -> int:
        """Return the vector dimension this fake emits."""
        return self._dimension

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed ``texts`` deterministically (see the class docstring)."""
        if isinstance(texts, (str, bytes, bytearray)):
            raise KnowledgeValidationError(f"texts must be a sequence of strings, got {type(texts).__name__}.")
        try:
            items = list(texts)
        except TypeError:
            raise KnowledgeValidationError(f"texts must be a sequence of strings, got {type(texts).__name__}.") from None
        return [self._vector_for_text(_require_text(f"texts[{i}]", item)) for i, item in enumerate(items)]

    def embed_one(self, text: str) -> list[float]:
        """Embed a single text deterministically."""
        return self._vector_for_text(_require_text("text", text))

    def _vector_for_text(self, normalized: str) -> list[float]:
        """Derive the stable unit vector for already-normalized text."""
        seed = hashlib.sha256(FAKE_DOMAIN.encode("utf-8") + b"\x00" + normalized.encode("utf-8")).digest()
        raw: list[float] = []
        counter = 0
        while len(raw) < self._dimension:
            digest = hashlib.sha256(seed + counter.to_bytes(4, "big")).digest()
            for words in struct.iter_unpack(">8I", digest):
                for word in words:
                    raw.append(word / 4294967296.0 * 2.0 - 1.0)
                    if len(raw) == self._dimension:
                        break
                if len(raw) == self._dimension:
                    break
            counter += 1
        norm = math.sqrt(sum(x * x for x in raw))
        if norm == 0.0:  # Unreachable in practice; keep the unit-norm promise total.
            return [1.0 / math.sqrt(self._dimension)] * self._dimension
        return [x / norm for x in raw]


def embed_texts(
    provider: EmbeddingProvider,
    texts: Sequence[str],
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[list[float]]:
    """Embed ``texts`` through ``provider`` in ``batch_size`` chunks.

    This is the shared batch API for one-off embedding (queries, eval
    fixtures, worker micro-batches): it validates inputs eagerly, preserves
    input order across chunks, and fail-closed-validates every returned
    vector against the provider's declared dimension.

    Args:
        provider: Embedding computation boundary.
        texts: Input texts (empty list returns ``[]`` without a provider call).
        batch_size: Texts per provider call (1..MAX_BATCH_SIZE).

    Returns:
        One validated ``list[float]`` vector per input text, in order.

    Raises:
        KnowledgeValidationError: On invalid texts or batch size.
        EmbeddingProviderError: When the provider fails or returns
            malformed vectors.
    """
    size = _require_batch_size(batch_size)
    if isinstance(texts, (str, bytes, bytearray)):
        raise KnowledgeValidationError(f"texts must be a sequence of strings, got {type(texts).__name__}.")
    try:
        items = list(texts)
    except TypeError:
        raise KnowledgeValidationError(f"texts must be a sequence of strings, got {type(texts).__name__}.") from None
    validated = [_require_text(f"texts[{i}]", item) for i, item in enumerate(items)]
    if not validated:
        return []
    dimension = provider.dimension
    model_id = provider.model_id
    vectors: list[list[float]] = []
    for start in range(0, len(validated), size):
        chunk = validated[start : start + size]
        try:
            raw = provider.embed_batch(chunk)
        except (KnowledgeValidationError, EmbeddingProviderError):
            raise
        except Exception as exc:
            raise EmbeddingProviderError(f"provider {model_id!r} failed on batch of {len(chunk)} texts: {exc}") from exc
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)) or len(list(raw)) != len(chunk):
            got = type(raw).__name__ if not isinstance(raw, Sequence) else f"sequence of length {len(list(raw))}"
            raise EmbeddingProviderError(f"provider {model_id!r} returned {got} for a batch of {len(chunk)} texts.")
        for offset, vector in enumerate(raw):
            vectors.append(_check_vector(vector, dimension=dimension, model_id=model_id, index=start + offset))
    return vectors


def render_embeddable_text(*, title: str, body: str, extra: Sequence[str] | None = None) -> str:
    """Render the canonical embeddable text for one knowledge object.

    Format is ``"<title>\\n\\n<body>"`` with optional ``extra`` sections
    appended as ``"\\n\\n<section>"`` (e.g. scope line, method summary).
    Every section is NFC/whitespace-normalized so semantically identical
    inputs embed identically regardless of source formatting.

    Args:
        title: Short label (finding title, experiment hypothesis, ...).
        body: Main prose (finding statement, dossier section, ...).
        extra: Optional additional sections in fixed caller-chosen order.

    Raises:
        KnowledgeValidationError: On empty title/body or invalid sections.
    """
    normalized_title = _require_text("title", title)
    normalized_body = _require_text("body", body)
    sections = [normalized_title, normalized_body]
    if extra is not None:
        if isinstance(extra, (str, bytes, bytearray)):
            raise KnowledgeValidationError(f"extra must be a sequence of strings, got {type(extra).__name__}.")
        try:
            extras = list(extra)
        except TypeError:
            raise KnowledgeValidationError(f"extra must be a sequence of strings, got {type(extra).__name__}.") from None
        for i, section in enumerate(extras):
            sections.append(_require_text(f"extra[{i}]", section))
    return "\n\n".join(sections)


_provider_factories: dict[str, Callable[[], EmbeddingProvider]] = {}


def register_provider_factory(model_id: str, factory: Callable[[], EmbeddingProvider]) -> None:
    """Register how to build the real provider for ``model_id`` (integration wiring point).

    The integration step calls this once at startup with a factory that
    loads the offline embedding model (e.g. a vendored
    sentence-transformers checkpoint resolved from
    ``DEER_FLOW_KNOWLEDGE_EMBEDDING_MODEL``) and returns an
    :class:`EmbeddingProvider` emitting :data:`EMBEDDING_DIM` vectors.
    Re-registering a model id replaces the previous factory.

    Args:
        model_id: Model id the factory serves (non-empty string).
        factory: Zero-argument callable returning a provider.

    Raises:
        KnowledgeValidationError: On an empty model id or a non-callable factory.
    """
    if not isinstance(model_id, str) or not model_id.strip():
        raise KnowledgeValidationError("model_id must be a non-empty string.")
    if not callable(factory):
        raise KnowledgeValidationError(f"factory must be callable, got {type(factory).__name__}.")
    _provider_factories[model_id.strip()] = factory


def load_provider(model_id: str | None = None) -> EmbeddingProvider:
    """Resolve the embedding provider for ``model_id``.

    Resolution order:

    1. Explicit ``model_id`` argument.
    2. ``DEER_FLOW_KNOWLEDGE_EMBEDDING_MODEL`` environment variable.
    3. :data:`FAKE_MODEL_ID` (``test-fake/v1``) **only when the caller passed
       no model id and the env var is unset** — an explicit offline default
       for tests and local dev, never a silent production fallback.

    ``test-fake`` / ``test-fake/v1`` resolves to
    :class:`DeterministicEmbeddingProvider`. Any other id requires a factory
    registered via :func:`register_provider_factory`; otherwise an
    :class:`EmbeddingProviderUnavailableError` is raised explaining the
    wiring (this is the documented real-model wiring point — no downloads
    are ever attempted implicitly).

    Raises:
        EmbeddingProviderUnavailableError: When the id is neither the fake
            nor a registered factory.
        EmbeddingProviderError: When a registered factory fails to build.
    """
    resolved = model_id.strip() if isinstance(model_id, str) and model_id.strip() else None
    if resolved is None:
        env_value = os.getenv(ENV_MODEL)
        if env_value is not None and env_value.strip():
            resolved = env_value.strip()
    if resolved is None:
        resolved = FAKE_MODEL_ID
    if resolved in {"test-fake", FAKE_MODEL_ID}:
        return DeterministicEmbeddingProvider()
    factory = _provider_factories.get(resolved)
    if factory is None:
        raise EmbeddingProviderUnavailableError(
            f"No embedding provider registered for model {resolved!r}. "
            f"Register one with deerflow.knowledge.embeddings.register_provider_factory({resolved!r}, factory) "
            f"at startup (factory must return an EmbeddingProvider emitting {EMBEDDING_DIM}-d vectors), "
            f"or set {ENV_MODEL}={FAKE_MODEL_ID} for the deterministic offline test fake. "
            "No model weights are downloaded implicitly."
        )
    try:
        return factory()
    except (KnowledgeValidationError, EmbeddingError):
        raise
    except Exception as exc:
        raise EmbeddingProviderError(f"provider factory for model {resolved!r} failed: {exc}") from exc


@runtime_checkable
class FindingTextSource(Protocol):
    """Read boundary: canonical embeddable text per finding id.

    PG binding sketch (Phase 2 ``finding`` table): render
    ``title`` + ``statement`` (+ scope line) with
    :func:`render_embeddable_text` for the row; return ``None`` when the
    finding does not exist or has no embeddable content.
    """

    def get_finding_text(self, finding_id: str) -> str | None:
        """Return the embeddable text for ``finding_id``, or None when missing."""
        ...


@runtime_checkable
class EmbeddingVectorStore(Protocol):
    """Write boundary: persist vectors beside their findings (pgvector).

    PG binding sketch (Phase 2 ``finding`` table): ``upsert_embedding``
    runs one parameterized ``UPDATE finding SET embedding = $2::vector,
    embedding_model = $3, embedding_updated_at = now() WHERE id = $1``
    (exact search first; HNSW index added later per the KB). The integration
    step must reject vectors whose length is not :data:`EMBEDDING_DIM`
    (pgvector does this natively for ``VECTOR(768)``) and keep vector
    indexing out of the transaction that proves a finding exists.
    """

    def get_embedding_model(self, finding_id: str) -> str | None:
        """Return the model id of the stored vector, or None when unembedded."""
        ...

    def upsert_embedding(self, finding_id: str, vector: Sequence[float], *, model_id: str) -> None:
        """Persist ``vector`` for ``finding_id``, stamped with ``model_id``."""
        ...


@dataclass(frozen=True)
class BackfillFailure:
    """One finding id that could not be embedded or upserted (collect mode)."""

    finding_id: str
    error: str

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-safe dict copy."""
        return {"finding_id": self.finding_id, "error": self.error}


@dataclass(frozen=True)
class BackfillResult:
    """Outcome of :func:`backfill_findings_embeddings` (all id lists in input order)."""

    total: int = 0
    embedded: int = 0
    upserted: int = 0
    skipped_up_to_date: int = 0
    upserted_ids: tuple[str, ...] = ()
    skipped_ids: tuple[str, ...] = ()
    missing_ids: tuple[str, ...] = ()
    failures: tuple[BackfillFailure, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe dict copy of the result."""
        return {
            "total": self.total,
            "embedded": self.embedded,
            "upserted": self.upserted,
            "skipped_up_to_date": self.skipped_up_to_date,
            "upserted_ids": list(self.upserted_ids),
            "skipped_ids": list(self.skipped_ids),
            "missing_ids": list(self.missing_ids),
            "failures": [failure.to_dict() for failure in self.failures],
        }


def _require_finding_id(value: object) -> str:
    """Validate a finding id (UUID string, canonical lowercase form)."""
    import uuid as _uuid

    if not isinstance(value, str):
        raise KnowledgeValidationError(f"finding_id must be a UUID string, got {type(value).__name__}.")
    try:
        return str(_uuid.UUID(value.strip()))
    except ValueError:
        raise KnowledgeValidationError(f"finding_id must be a valid UUID, got {value!r}.") from None


def backfill_findings_embeddings(
    provider: EmbeddingProvider,
    texts: FindingTextSource,
    store: EmbeddingVectorStore,
    finding_ids: Sequence[str],
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    skip_up_to_date: bool = True,
    on_error: str = "collect",
) -> BackfillResult:
    """Backfill (finding-id -> vector) upserts for Phase 2 indexing workers.

    For each unique finding id, in input order: optionally skip when the
    stored vector already carries this provider's ``model_id``; otherwise
    fetch the canonical text, embed in ``batch_size`` chunks (input order
    preserved), and upsert each validated vector. Re-running with the same
    provider is idempotent (everything already stamped is skipped); switching
    models re-embeds every row, which is the intended model-migration path.

    Args:
        provider: Embedding computation boundary. Its dimension must equal
            :data:`EMBEDDING_DIM` — anything else is refused before any
            store call so a misconfigured model can never poison the
            ``VECTOR(768)`` column.
        texts: Read boundary supplying canonical embeddable text per id.
        store: Write boundary persisting vectors per id.
        finding_ids: Finding UUID strings (duplicates collapse, order kept).
        batch_size: Findings embedded per provider call (1..MAX_BATCH_SIZE).
        skip_up_to_date: When True (default), skip ids whose stored vector
            already carries ``provider.model_id``.
        on_error: ``"collect"`` records per-id failures and continues;
            ``"raise"`` propagates the first provider/store error.

    Returns:
        A :class:`BackfillResult` with counts and per-id outcomes.

    Raises:
        KnowledgeValidationError: On invalid ids, batch size, or ``on_error``.
        EmbeddingProviderError: On dimension mismatch, or on the first
            provider/store failure when ``on_error="raise"``.
    """
    size = _require_batch_size(batch_size)
    if on_error not in {"collect", "raise"}:
        raise KnowledgeValidationError(f"on_error must be 'collect' or 'raise', got {on_error!r}.")
    if isinstance(finding_ids, (str, bytes, bytearray)):
        raise KnowledgeValidationError(f"finding_ids must be a sequence of UUID strings, got {type(finding_ids).__name__}.")
    try:
        raw_ids = list(finding_ids)
    except TypeError:
        raise KnowledgeValidationError(f"finding_ids must be a sequence of UUID strings, got {type(finding_ids).__name__}.") from None
    unique_ids = list(dict.fromkeys(_require_finding_id(item) for item in raw_ids))

    model_id = provider.model_id
    dimension = provider.dimension
    if dimension != EMBEDDING_DIM:
        raise EmbeddingProviderError(f"Refusing backfill: provider {model_id!r} emits {dimension}-d vectors, but the canonical store column is VECTOR({EMBEDDING_DIM}).")

    skipped: list[str] = []
    missing: list[str] = []
    failures: list[BackfillFailure] = []
    pending: list[tuple[str, str]] = []  # (finding_id, text) pairs to embed, in order.

    def _record(finding_id: str, action: str, exc: BaseException) -> None:
        if on_error == "raise":
            if isinstance(exc, (KnowledgeError,)):
                raise exc
            raise EmbeddingProviderError(f"{action} for finding {finding_id} failed: {exc}") from exc
        failures.append(BackfillFailure(finding_id=finding_id, error=f"{exc}"))

    for finding_id in unique_ids:
        if skip_up_to_date:
            try:
                if store.get_embedding_model(finding_id) == model_id:
                    skipped.append(finding_id)
                    continue
            except Exception as exc:
                _record(finding_id, "embedding-model lookup", exc)
                continue
        try:
            text = texts.get_finding_text(finding_id)
        except Exception as exc:
            _record(finding_id, "finding-text lookup", exc)
            continue
        if text is None:
            missing.append(finding_id)
            continue
        try:
            pending.append((finding_id, _require_text(f"finding {finding_id} text", text)))
        except KnowledgeValidationError as exc:
            _record(finding_id, "finding-text validation", exc)

    upserted: list[str] = []
    embedded_count = 0
    for start in range(0, len(pending), size):
        chunk = pending[start : start + size]
        chunk_ids = [finding_id for finding_id, _ in chunk]
        try:
            raw_vectors = provider.embed_batch([text for _, text in chunk])
        except Exception as exc:
            for finding_id in chunk_ids:
                _record(finding_id, "embed_batch", exc)
            continue
        if not isinstance(raw_vectors, Sequence) or isinstance(raw_vectors, (str, bytes, bytearray)) or len(list(raw_vectors)) != len(chunk):
            for finding_id in chunk_ids:
                _record(finding_id, "embed_batch", EmbeddingProviderError(f"provider {model_id!r} returned a malformed batch."))
            continue
        vectors: list[list[float] | None] = []
        for offset, vector in enumerate(raw_vectors):
            try:
                vectors.append(_check_vector(vector, dimension=EMBEDDING_DIM, model_id=model_id, index=start + offset))
            except EmbeddingProviderError as exc:
                _record(chunk_ids[offset], "vector validation", exc)
                vectors.append(None)
        embedded_count += sum(1 for vector in vectors if vector is not None)
        for finding_id, vector in zip(chunk_ids, vectors, strict=True):
            if vector is None:
                continue
            try:
                store.upsert_embedding(finding_id, vector, model_id=model_id)
            except Exception as exc:
                _record(finding_id, "upsert_embedding", exc)
                continue
            upserted.append(finding_id)

    return BackfillResult(
        total=len(unique_ids),
        embedded=embedded_count,
        upserted=len(upserted),
        skipped_up_to_date=len(skipped),
        upserted_ids=tuple(upserted),
        skipped_ids=tuple(skipped),
        missing_ids=tuple(missing),
        failures=tuple(failures),
    )
