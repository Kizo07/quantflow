"""Canonical hashing for experiments: family identity vs exact execution identity.

The knowledge base distinguishes two signatures (KB: "Experiments should be
first-class objects"):

* ``experiment_family_hash`` — identifies *conceptually equivalent* research
  designs so replications group together. It covers only the normalized
  ``design`` mapping (hypothesis + methodology). Two runs with the same
  design but different random seeds, code revisions, or dataset vintages
  share a family hash.
* ``execution_hash`` — additionally incorporates exact code, data,
  parameters, and environment. Identical execution hashes mean the exact
  computational configuration has already run (deduplication key).

Normalization rules (structural canonicalization before hashing):

1. Mappings serialize with keys sorted lexicographically; nesting depth is
   preserved exactly.
2. Mapping keys must be strings; anything else raises ``TypeError``.
3. ``None`` values are *significant*: ``{"a": None}`` and ``{}`` hash
   differently, because an explicit null is a stated choice, not an omission.
4. Sequences (``list``/``tuple``) preserve element order; ``tuple`` and
   ``list`` with equal items are equivalent.
5. Strings are NFC-normalized (so composed and decomposed Unicode forms are
   equivalent) but remain **case-sensitive** and whitespace-sensitive:
   ``"AAPL"`` and ``"aapl"`` hash differently. Callers that want
   case-insensitive grouping must lowercase before hashing.
6. ``bool`` is distinct from ``int`` (``True`` != ``1``) and ``int`` is
   distinct from ``float`` (``1`` != ``1.0``): JSON renders them differently
   and conflating them would merge semantically different parameters.
7. Floats must be finite; NaN/±Infinity raise ``ValueError`` (they have no
   canonical JSON form). ``-0.0`` normalizes to ``0.0``.
8. ``bytes``, ``set``, and arbitrary objects are rejected with ``TypeError``;
   callers must convert to JSON-native types (e.g. hex-encode digests,
   sort sets into lists) so the conversion is explicit and reviewable.

Serialization is canonical JSON: ``sort_keys=True``,
``separators=(",", ":")``, ``ensure_ascii=True``, UTF-8 bytes. Each hash
domain is separated by prefixing the digest input with
``"<domain>\\x00"`` so a family hash can never collide with an execution
hash or idempotency key by construction.

Equivalence/collision contract:

* Equal normalized structures always produce equal hashes (deterministic
  across processes, machines, and key insertion orders).
* Any normalized difference (added/removed/changed key, reordered list,
  changed scalar) produces a different hash with SHA-256 collision
  resistance; near-duplicate *linking* is a retrieval-plane concern, never
  implied by these hashes.
* Consumers must treat hash equality as the *only* exact-identity signal;
  embedding/lexical similarity must not widen it.
"""

import hashlib
import json
import math
import unicodedata
from collections.abc import Mapping
from typing import Any

FAMILY_DOMAIN = "quantflow.experiment_family/v1"
EXECUTION_DOMAIN = "quantflow.execution/v1"
IDEMPOTENCY_DOMAIN = "quantflow.idempotency/v1"

__all__ = [
    "FAMILY_DOMAIN",
    "EXECUTION_DOMAIN",
    "IDEMPOTENCY_DOMAIN",
    "normalize",
    "normalize_text",
    "canonical_json",
    "canonical_bytes",
    "sha256_hex",
    "experiment_family_hash",
    "execution_hash",
    "make_idempotency_key",
]


def normalize(value: Any) -> Any:
    """Normalize a value to a canonical JSON-native structure.

    Args:
        value: A JSON-native value (None/bool/int/float/str/list/tuple/dict).

    Returns:
        An equivalent structure using only None/bool/int/float/str/list/dict
        with NFC-normalized strings, tuples converted to lists, and -0.0
        converted to 0.0. Dict key order is irrelevant (sorting happens at
        serialization).

    Raises:
        TypeError: For non-JSON-native types (bytes, set, custom objects) or
            non-string mapping keys.
        ValueError: For non-finite floats (NaN, +Infinity, -Infinity).
    """
    if value is None or isinstance(value, (bool, int, str)):
        if isinstance(value, str):
            return unicodedata.normalize("NFC", value)
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"Cannot canonicalize non-finite float: {value!r}")
        if value == 0.0:
            return 0.0
        return value
    if isinstance(value, (list, tuple)):
        return [normalize(item) for item in value]
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"Cannot canonicalize mapping with non-string key: {key!r}")
            normalized[unicodedata.normalize("NFC", key)] = normalize(item)
        return normalized
    raise TypeError(f"Cannot canonicalize value of type {type(value).__name__}: {value!r}")


def normalize_text(text: str) -> str:
    """Normalize free-text hypothesis/statement input for identity purposes.

    Strips leading/trailing whitespace and collapses internal runs of
    whitespace to single spaces (NFC-normalized). Comparison stays
    case-sensitive by design; see the module docstring.
    """
    if not isinstance(text, str):
        raise TypeError(f"normalize_text expects str, got {type(text).__name__}")
    return " ".join(unicodedata.normalize("NFC", text).split())


def canonical_json(value: Any) -> str:
    """Serialize a value to canonical JSON (sorted keys, compact separators, ASCII)."""
    return json.dumps(normalize(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def canonical_bytes(value: Any) -> bytes:
    """Return the UTF-8 bytes of the canonical JSON serialization."""
    return canonical_json(value).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    """Return the lowercase hex SHA-256 digest of raw bytes."""
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError(f"sha256_hex expects bytes, got {type(data).__name__}")
    return hashlib.sha256(bytes(data)).hexdigest()


def _domain_hash(domain: str, payload: Any) -> str:
    """Hash a payload under a domain-separation prefix."""
    return sha256_hex(domain.encode("utf-8") + b"\x00" + canonical_bytes(payload))


def _require_design(design: Any, name: str) -> Mapping[str, Any]:
    """Validate that an identity input is a mapping."""
    if not isinstance(design, Mapping):
        raise TypeError(f"{name} must be a mapping, got {type(design).__name__}")
    return design


def experiment_family_hash(design: Mapping[str, Any]) -> str:
    """Compute the family hash identifying conceptually equivalent designs.

    Args:
        design: Normalized research design, conventionally
            ``{"hypothesis": <normalized text>, "methodology": {...}}`` covering
            universe, frequency, horizon, sample period, protocol, portfolio
            construction, cost/slippage models, neutralization, and rebalance
            rule — everything that makes two runs "the same experiment", and
            nothing run-specific (no seeds, code revisions, dataset vintages,
            timestamps, or run ids).

    Returns:
        64-character lowercase hex digest under ``FAMILY_DOMAIN``.
    """
    _require_design(design, "design")
    return _domain_hash(FAMILY_DOMAIN, design)


def execution_hash(
    *,
    design: Mapping[str, Any],
    code: Mapping[str, Any],
    data: Mapping[str, Any],
    parameters: Mapping[str, Any],
    environment: Mapping[str, Any],
) -> str:
    """Compute the execution hash identifying one exact computation.

    Args:
        design: Same design mapping used for the family hash.
        code: Exact code identity (artifact sha256, git commit, subdirectory).
        data: Exact data identity (dataset version ids, vintages, roles).
        parameters: Full parameter mapping including random seeds.
        environment: Runtime identity (image/lock hash, dependency versions).

    Returns:
        64-character lowercase hex digest under ``EXECUTION_DOMAIN``.
    """
    _require_design(design, "design")
    _require_design(code, "code")
    _require_design(data, "data")
    _require_design(parameters, "parameters")
    _require_design(environment, "environment")
    envelope = {
        "design": design,
        "code": code,
        "data": data,
        "parameters": parameters,
        "environment": environment,
    }
    return _domain_hash(EXECUTION_DOMAIN, envelope)


def make_idempotency_key(run_id: str, sequence: int, payload: Any) -> str:
    """Build a deterministic idempotency key per the KB write protocol.

    ``SHA256(run_id || local_event_sequence || canonical_payload)`` under
    ``IDEMPOTENCY_DOMAIN``. Retried writes carrying the same key replay the
    original record; a different payload under the same key is a conflict.

    Args:
        run_id: Client run identifier (non-empty string).
        sequence: Monotonic local event sequence within the run (non-negative int).
        payload: JSON-native payload the key binds to.

    Returns:
        64-character lowercase hex digest.
    """
    if not isinstance(run_id, str) or not run_id.strip():
        raise TypeError("run_id must be a non-empty string")
    if not isinstance(sequence, int) or isinstance(sequence, bool) or sequence < 0:
        raise TypeError("sequence must be a non-negative int")
    prefix = run_id.encode("utf-8") + b"\x00" + str(sequence).encode("ascii") + b"\x00"
    return sha256_hex(IDEMPOTENCY_DOMAIN.encode("utf-8") + b"\x00" + prefix + canonical_bytes(payload))
