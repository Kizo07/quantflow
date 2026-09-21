"""Real offline embedding provider backed by sentence-transformers (Phase 2: shared recall).

Wiring point implementation for :func:`deerflow.knowledge.embeddings.register_provider_factory`.
Importing this module never imports ``torch``/``sentence-transformers``; the
heavy dependency loads only when :class:`SentenceTransformerEmbeddingProvider`
is constructed, and construction fails closed with a clear error when the
package is missing.

The provider emits canonical :data:`EMBEDDING_DIM` (768) L2-unit vectors from
``sentence-transformers/all-mpnet-base-v2`` (default id :data:`ST_MODEL_ID`),
resolved from the local HuggingFace cache (no implicit downloads at retrieval
time — the checkpoint must already be cached; ``HF_HUB_OFFLINE=1`` is honored).
"""

from __future__ import annotations

import threading

from deerflow.knowledge.embeddings import (
    EMBEDDING_DIM,
    EmbeddingProvider,
    EmbeddingProviderError,
    _check_vector,
    _require_text,
    register_provider_factory,
)
from deerflow.knowledge.write_api import KnowledgeValidationError

#: Default model id: MPNet base fine-tuned for sentence similarity (768-d).
ST_MODEL_ID = "sentence-transformers/all-mpnet-base-v2"

#: Internal encode chunk size (bounds peak GPU/CPU memory on large backfills).
_ST_BATCH_SIZE = 32

__all__ = [
    "ST_MODEL_ID",
    "SentenceTransformerEmbeddingProvider",
    "register_st_provider",
]


class SentenceTransformerEmbeddingProvider:
    """Real semantic embedding provider (MPNet 768-d, L2-normalized).

    Args:
        model_id: Sentence-transformers model id or local path. Must emit
            :data:`EMBEDDING_DIM` vectors; anything else fails closed at
            first encode.

    Raises:
        KnowledgeValidationError: On an empty model id.
        EmbeddingProviderError: When ``sentence-transformers`` is not
            installed or the checkpoint cannot be loaded.
    """

    def __init__(self, model_id: str = ST_MODEL_ID) -> None:
        if not isinstance(model_id, str) or not model_id.strip():
            raise KnowledgeValidationError("model_id must be a non-empty string.")
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise EmbeddingProviderError(f"sentence-transformers is not installed in this environment; install it (pip install sentence-transformers) and cache the {model_id.strip()!r} checkpoint before wiring this provider.") from exc
        self._model_id = model_id.strip()
        self._lock = threading.Lock()
        try:
            self._model = SentenceTransformer(self._model_id)
        except Exception as exc:
            raise EmbeddingProviderError(f"Failed to load sentence-transformers checkpoint {self._model_id!r}: {exc}") from exc
        self._model.eval()
        dimension = int(self._model.get_sentence_embedding_dimension())
        if dimension != EMBEDDING_DIM:
            raise EmbeddingProviderError(f"Checkpoint {self._model_id!r} emits {dimension}-d vectors, expected canonical {EMBEDDING_DIM}; refusing to pad/truncate.")

    @property
    def model_id(self) -> str:
        """Return the checkpoint id stamped on stored rows."""
        return self._model_id

    @property
    def dimension(self) -> int:
        """Return the canonical embedding dimension (768)."""
        return EMBEDDING_DIM

    def embed_batch(self, texts: object) -> list[list[float]]:
        """Embed ``texts`` in order; returns one unit vector per input text."""
        if isinstance(texts, (str, bytes, bytearray)):
            raise KnowledgeValidationError(f"texts must be a sequence of strings, got {type(texts).__name__}.")
        try:
            items = list(texts)  # type: ignore[arg-type]
        except TypeError:
            raise KnowledgeValidationError(f"texts must be a sequence of strings, got {type(texts).__name__}.") from None
        normalized = [_require_text(f"texts[{i}]", item) for i, item in enumerate(items)]
        vectors: list[list[float]] = []
        with self._lock:
            for start in range(0, len(normalized), _ST_BATCH_SIZE):
                chunk = normalized[start : start + _ST_BATCH_SIZE]
                try:
                    encoded = self._model.encode(chunk, normalize_embeddings=True, show_progress_bar=False)
                except Exception as exc:
                    raise EmbeddingProviderError(f"provider {self._model_id!r} failed to encode batch: {exc}") from exc
                for offset, row in enumerate(encoded):
                    vectors.append(
                        _check_vector(
                            [float(v) for v in row],
                            dimension=EMBEDDING_DIM,
                            model_id=self._model_id,
                            index=start + offset,
                        )
                    )
        return vectors

    def embed_one(self, text: str) -> list[float]:
        """Embed a single text."""
        return self.embed_batch([text])[0]


def register_st_provider(model_id: str = ST_MODEL_ID) -> EmbeddingProvider:
    """Register the sentence-transformers factory for ``model_id`` and return a provider.

    Convenience for startup wiring and eval runners::

        from deerflow.knowledge.providers_st import register_st_provider
        provider = register_st_provider()  # also registers the factory
    """
    if not isinstance(model_id, str) or not model_id.strip():
        raise KnowledgeValidationError("model_id must be a non-empty string.")
    resolved = model_id.strip()
    register_provider_factory(resolved, lambda: SentenceTransformerEmbeddingProvider(resolved))
    from deerflow.knowledge.embeddings import load_provider

    return load_provider(resolved)
