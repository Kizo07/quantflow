"""Cross-implementation hash vectors: the interop contract.

``hash_vectors.json`` (same bytes as
``alpha_engine/tests/hash_vectors.json``) pins the exact digests every
conformant implementation must produce. Generated from this module's
implementation — the reference — so this suite passing here is the
self-consistency half; the alpha_engine suite asserting the same file is
the parity half. A silent drift on either side fails loudly.
"""

import json
from pathlib import Path

from deerflow.knowledge.hashing import (
    canonical_json,
    execution_hash,
    experiment_family_hash,
    make_idempotency_key,
)

FUNCS = {
    "canonical_json": canonical_json,
    "execution_hash": execution_hash,
    "experiment_family_hash": experiment_family_hash,
    "make_idempotency_key": make_idempotency_key,
}


def _load_vectors():
    doc = json.loads((Path(__file__).parent / "hash_vectors.json").read_text())
    assert doc["vectors"], "no vectors in hash_vectors.json"
    return doc["vectors"]


def test_all_hash_vectors_match_reference():
    for item in _load_vectors():
        func = FUNCS[item["func"]]
        assert func(**item["args"]) == item["expected"], item["name"]


def test_nfc_pair_agrees_and_case_pair_differs():
    by_name = {v["name"]: v["expected"] for v in _load_vectors()}
    assert by_name["family_nfc_composed"] == by_name["family_nfc_decomposed"]
    assert by_name["family_case_differs"] != by_name["family_basic"]
    assert by_name["family_whitespace_collapsed"] == by_name["family_basic"]
    assert by_name["family_bool_vs_int"] != by_name["family_int_one"]
