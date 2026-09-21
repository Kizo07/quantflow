"""Tests for the sentence-transformers embedding provider (real-model wiring).

Cheap tests (validation, registration shape) always run. The checkpoint
smoke test is ``live``: it loads ``all-mpnet-base-v2`` from the local
HuggingFace cache and runs only with ``DEER_FLOW_RUN_LIVE_TESTS=1`` outside
CI, per the ``test_client_live.py`` convention.
"""

from __future__ import annotations

import os

import pytest

from deerflow.knowledge.embeddings import (
    EMBEDDING_DIM,
    load_provider,
)
from deerflow.knowledge.providers_st import (
    ST_MODEL_ID,
    SentenceTransformerEmbeddingProvider,
    register_st_provider,
)
from deerflow.knowledge.write_api import KnowledgeValidationError

_LIVE_OPT_IN = "DEER_FLOW_RUN_LIVE_TESTS"

live_required = pytest.mark.skipif(
    os.environ.get("CI") or os.environ.get(_LIVE_OPT_IN) != "1",
    reason="live checkpoint test; set DEER_FLOW_RUN_LIVE_TESTS=1 to run",
)


def test_st_model_id_is_mpnet() -> None:
    """Default checkpoint id is the 768-d MPNet sentence model."""
    assert ST_MODEL_ID == "sentence-transformers/all-mpnet-base-v2"


def test_empty_model_id_rejected_without_importing_torch() -> None:
    """Model-id validation runs before the heavy dependency import."""
    with pytest.raises(KnowledgeValidationError):
        SentenceTransformerEmbeddingProvider("   ")
    with pytest.raises(KnowledgeValidationError):
        register_st_provider("")


def test_provider_protocol_shape() -> None:
    """Provider exposes the protocol surface (properties + embed methods)."""
    assert isinstance(SentenceTransformerEmbeddingProvider.model_id, property)
    assert isinstance(SentenceTransformerEmbeddingProvider.dimension, property)
    assert callable(SentenceTransformerEmbeddingProvider.embed_batch)
    assert callable(SentenceTransformerEmbeddingProvider.embed_one)


@pytest.mark.live
@live_required
def test_mpnet_checkpoint_smoke() -> None:
    """Load the cached checkpoint, embed two texts, check shapes + unit norm."""
    pytest.importorskip("sentence_transformers")
    provider = register_st_provider()
    assert provider.model_id == ST_MODEL_ID
    assert provider.dimension == EMBEDDING_DIM == 768
    # Factory registration sticks: load_provider resolves the same id.
    assert load_provider(ST_MODEL_ID).model_id == ST_MODEL_ID
    vectors = provider.embed_batch(["momentum factor returns", "momentum factor returns ".strip()])
    assert len(vectors) == 2
    for vector in vectors:
        assert len(vector) == 768
        norm = sum(v * v for v in vector) ** 0.5
        assert norm == pytest.approx(1.0, abs=1e-4)
    # Same text -> same vector (deterministic for a fixed checkpoint).
    assert vectors[0] == vectors[1]
