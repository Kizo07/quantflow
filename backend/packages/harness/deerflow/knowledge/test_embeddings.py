"""Standalone tests for the embedding provider interface + backfill helper (Phase 2).

Runs with no database, no model weights, and no integration wiring: the
deterministic fake supplies vectors and in-memory fakes implement the
``FindingTextSource`` / ``EmbeddingVectorStore`` boundaries from
``embeddings.py``.

Run from anywhere (the bootstrap below locates the harness package)::

    python -m pytest backend/packages/harness/deerflow/knowledge/test_embeddings.py -q
"""

import math
import sys
from pathlib import Path

_HARNESS_DIR = Path(__file__).resolve().parents[2]
if str(_HARNESS_DIR) not in sys.path:
    sys.path.insert(0, str(_HARNESS_DIR))

import pytest  # noqa: E402

from deerflow.knowledge import embeddings as E  # noqa: E402
from deerflow.knowledge.write_api import KnowledgeValidationError  # noqa: E402

FINDING_1 = "11111111-1111-1111-1111-111111111111"
FINDING_2 = "22222222-2222-2222-2222-222222222222"
FINDING_3 = "33333333-3333-3333-3333-333333333333"


def make_provider(**kwargs):
    return E.DeterministicEmbeddingProvider(**kwargs)


class FakeTexts:
    """In-memory FindingTextSource."""

    def __init__(self, mapping):
        self.mapping = dict(mapping)
        self.calls = []

    def get_finding_text(self, finding_id):
        self.calls.append(finding_id)
        return self.mapping.get(finding_id)


class FakeVectorStore:
    """In-memory EmbeddingVectorStore."""

    def __init__(self):
        self.models = {}
        self.vectors = {}
        self.upserts = []

    def get_embedding_model(self, finding_id):
        return self.models.get(finding_id)

    def upsert_embedding(self, finding_id, vector, *, model_id):
        self.models[finding_id] = model_id
        self.vectors[finding_id] = list(vector)
        self.upserts.append(finding_id)


def norm(vector):
    return math.sqrt(sum(x * x for x in vector))


def cosine(a, b):
    return sum(x * y for x, y in zip(a, b, strict=True)) / (norm(a) * norm(b))


def test_canonical_dimension_is_768():
    assert E.EMBEDDING_DIM == 768
    assert make_provider().dimension == 768


def test_fake_vectors_are_unit_norm_768d_finite():
    vector = make_provider().embed_one("Cross-sectional momentum persists net of costs.")
    assert len(vector) == 768
    assert all(isinstance(x, float) and math.isfinite(x) for x in vector)
    assert norm(vector) == pytest.approx(1.0)


def test_fake_is_deterministic_across_instances():
    text = "Some research prose about turnover."
    assert make_provider().embed_one(text) == make_provider().embed_one(text)
    assert make_provider().embed_batch([text, text])[0] == make_provider().embed_batch([text, text])[1]


def test_fake_normalizes_unicode_and_whitespace():
    base = make_provider().embed_one("h\u00e9llo   momentum")
    assert make_provider().embed_one("he\u0301llo momentum") == base  # decomposed e + combining acute
    assert make_provider().embed_one("  h\u00e9llo momentum\n") == base


def test_fake_is_case_sensitive_and_distinct_per_text():
    a = make_provider().embed_one("AAPL momentum")
    b = make_provider().embed_one("aapl momentum")
    assert a != b
    c = make_provider().embed_one("Unrelated prose about bond convexity.")
    assert cosine(a, c) < 0.99


def test_embed_one_matches_embed_batch():
    provider = make_provider()
    texts = ["first finding", "second finding"]
    assert provider.embed_one(texts[0]) == provider.embed_batch(texts)[0]


def test_embed_one_rejects_bad_text():
    provider = make_provider()
    for bad in ("", "   ", 123, None, "x" * (E.MAX_TEXT_CHARS + 1)):
        with pytest.raises(KnowledgeValidationError):
            provider.embed_one(bad)
    with pytest.raises(KnowledgeValidationError):
        provider.embed_batch("not-a-sequence")
    with pytest.raises(KnowledgeValidationError):
        provider.embed_batch(["ok", "  "])


def test_embed_texts_chunks_and_preserves_order():
    provider = make_provider()
    texts = [f"finding text number {i}" for i in range(5)]
    assert E.embed_texts(provider, texts, batch_size=2) == provider.embed_batch(texts)
    assert E.embed_texts(provider, []) == []
    with pytest.raises(KnowledgeValidationError):
        E.embed_texts(provider, texts, batch_size=0)
    with pytest.raises(KnowledgeValidationError):
        E.embed_texts(provider, texts, batch_size=E.MAX_BATCH_SIZE + 1)


def test_embed_texts_rejects_misbehaving_provider():
    class WrongDim:
        model_id = "rogue"
        dimension = 768

        def embed_batch(self, texts):
            return [[0.1, 0.2] for _ in texts]

    class NonFinite:
        model_id = "rogue"
        dimension = 2

        def embed_batch(self, texts):
            return [[float("nan"), 0.0] for _ in texts]

    class Exploding:
        model_id = "rogue"
        dimension = 768

        def embed_batch(self, texts):
            raise RuntimeError("boom")

    with pytest.raises(E.EmbeddingProviderError):
        E.embed_texts(WrongDim(), ["a"])
    with pytest.raises(E.EmbeddingProviderError):
        E.embed_texts(NonFinite(), ["a"])
    with pytest.raises(E.EmbeddingProviderError):
        E.embed_texts(Exploding(), ["a"])


def test_render_embeddable_text_is_canonical():
    rendered = E.render_embeddable_text(title="  Title ", body="line1\nline2", extra=[" scope: equities "])
    assert rendered == "Title\n\nline1 line2\n\nscope: equities"
    with pytest.raises(KnowledgeValidationError):
        E.render_embeddable_text(title="t", body="  ")
    with pytest.raises(KnowledgeValidationError):
        E.render_embeddable_text(title="t", body="b", extra="not-a-sequence")


def test_load_provider_fake_and_registry(monkeypatch):
    provider = E.load_provider("test-fake/v1")
    assert isinstance(provider, E.DeterministicEmbeddingProvider)
    monkeypatch.delenv(E.ENV_MODEL, raising=False)
    assert isinstance(E.load_provider(None), E.DeterministicEmbeddingProvider)
    monkeypatch.setenv(E.ENV_MODEL, "test-fake/v1")
    assert isinstance(E.load_provider(None), E.DeterministicEmbeddingProvider)
    with pytest.raises(E.EmbeddingProviderUnavailableError):
        E.load_provider("nonexistent-model-xyz")
    sentinel = make_provider()
    E.register_provider_factory("custom-test-model", lambda: sentinel)
    try:
        assert E.load_provider("custom-test-model") is sentinel
    finally:
        E._provider_factories.pop("custom-test-model", None)
    with pytest.raises(KnowledgeValidationError):
        E.register_provider_factory("  ", lambda: sentinel)


def test_backfill_upserts_and_is_idempotent():
    provider = make_provider()
    texts = FakeTexts({FINDING_1: "first finding text", FINDING_2: "second finding text"})
    store = FakeVectorStore()
    result = E.backfill_findings_embeddings(provider, texts, store, [FINDING_1, FINDING_2])
    assert (result.total, result.embedded, result.upserted, result.skipped_up_to_date) == (2, 2, 2, 0)
    assert result.failures == () and result.missing_ids == ()
    assert store.models == {FINDING_1: E.FAKE_MODEL_ID, FINDING_2: E.FAKE_MODEL_ID}
    assert all(len(v) == 768 for v in store.vectors.values())
    assert store.vectors[FINDING_1] == provider.embed_one("first finding text")

    rerun = E.backfill_findings_embeddings(provider, texts, store, [FINDING_1, FINDING_2])
    assert (rerun.upserted, rerun.skipped_up_to_date) == (0, 2)
    assert rerun.skipped_ids == (FINDING_1, FINDING_2)


def test_backfill_missing_duplicates_and_bad_ids():
    provider = make_provider()
    texts = FakeTexts({FINDING_1: "only finding"})
    store = FakeVectorStore()
    result = E.backfill_findings_embeddings(provider, texts, store, [FINDING_1, FINDING_1, FINDING_3])
    assert result.total == 2  # duplicates collapse
    assert result.upserted_ids == (FINDING_1,)
    assert result.missing_ids == (FINDING_3,)
    with pytest.raises(KnowledgeValidationError):
        E.backfill_findings_embeddings(provider, texts, store, ["not-a-uuid"])
    with pytest.raises(KnowledgeValidationError):
        E.backfill_findings_embeddings(provider, texts, store, [FINDING_1], on_error="ignore")


def test_backfill_refuses_dimension_mismatch_without_store_calls():
    provider = make_provider(dimension=128)
    texts = FakeTexts({FINDING_1: "text"})
    store = FakeVectorStore()
    with pytest.raises(E.EmbeddingProviderError):
        E.backfill_findings_embeddings(provider, texts, store, [FINDING_1])
    assert store.upserts == [] and texts.calls == []


def test_backfill_collect_vs_raise_on_store_failure():
    class FailingStore(FakeVectorStore):
        def upsert_embedding(self, finding_id, vector, *, model_id):
            raise RuntimeError("pgvector down")

    provider = make_provider()
    texts = FakeTexts({FINDING_1: "text one", FINDING_2: "text two"})
    collected = E.backfill_findings_embeddings(provider, texts, FailingStore(), [FINDING_1, FINDING_2])
    assert collected.upserted == 0
    assert [f.finding_id for f in collected.failures] == [FINDING_1, FINDING_2]
    assert collected.to_dict()["failures"][0]["error"] == "pgvector down"
    with pytest.raises(E.EmbeddingProviderError):
        E.backfill_findings_embeddings(provider, texts, FailingStore(), [FINDING_1], on_error="raise")
