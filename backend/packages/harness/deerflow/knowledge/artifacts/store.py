"""SHA-256 content-addressed artifact store.

:class:`ArtifactStore` layers immutability, content addressing, metadata
records, and read-path integrity verification on top of an
:class:`~deerflow.knowledge.artifacts.backends.ObjectBackend`.

Storage layout under an optional ``key_prefix``::

    <prefix>/sha256/<aa>/<bb>/<digest>           artifact bytes
    <prefix>/sha256/<aa>/<bb>/<digest>.record.json  metadata record sidecar

The two-hex-character fan-out keeps any single directory listing bounded on
filesystem-style backends. Records are stored as JSON sidecars so the store
is self-contained (usable before/without the PostgreSQL ``artifact`` table,
which remains the canonical metadata index once Phase 1 migrations land).
"""

from __future__ import annotations

import hashlib
import io
import json
import os
from collections.abc import Iterator
from typing import Any, BinaryIO

from .backends import DEFAULT_CHUNK_SIZE, ObjectBackend
from .models import ArtifactIntegrityError, ArtifactKind, ArtifactNotFoundError, ArtifactRecord, artifact_uri

#: Suffix for the JSON metadata sidecar stored next to each blob.
RECORD_SUFFIX = ".record.json"


def hash_stream(stream: BinaryIO, chunk_size: int = DEFAULT_CHUNK_SIZE) -> tuple[str, int]:
    """Hash a binary stream with SHA-256; return ``(hex_digest, byte_count)``.

    The stream is consumed from its current position to EOF.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    digest = hashlib.sha256()
    total = 0
    while True:
        chunk = stream.read(chunk_size)
        if not chunk:
            return digest.hexdigest(), total
        if isinstance(chunk, str):
            chunk = chunk.encode("utf-8")
        digest.update(chunk)
        total += len(chunk)


def hash_bytes(data: bytes) -> tuple[str, int]:
    """Hash an in-memory payload; return ``(hex_digest, byte_count)``."""
    return hashlib.sha256(data).hexdigest(), len(data)


def blob_key_for_digest(digest: str, key_prefix: str = "") -> str:
    """Return the object key for artifact bytes with hex digest ``digest``."""
    _require_digest(digest)
    key = f"sha256/{digest[0:2]}/{digest[2:4]}/{digest}"
    return f"{key_prefix.rstrip('/')}/{key}" if key_prefix else key


def record_key_for_digest(digest: str, key_prefix: str = "") -> str:
    """Return the object key for the metadata sidecar of ``digest``."""
    return blob_key_for_digest(digest, key_prefix) + RECORD_SUFFIX


def _require_digest(digest: str) -> str:
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest.lower()):
        raise ValueError(f"invalid sha256 digest: {digest!r}")
    return digest.lower()


def parse_artifact_uri(uri: str) -> str:
    """Extract and validate the hex digest from an ``artifact://`` URI or bare digest."""
    text = uri.strip()
    if text.startswith("artifact://sha256/"):
        text = text[len("artifact://sha256/") :]
    elif "://" in text:
        raise ValueError(f"unsupported artifact URI scheme: {uri!r}")
    return _require_digest(text)


class VerifyingReader(io.RawIOBase):
    """Binary stream wrapper that re-hashes bytes while reading.

    The digest is finalized when EOF is reached; if the recomputed hash does
    not match ``expected_digest``, :class:`ArtifactIntegrityError` is raised
    on the read that hits EOF (or on :meth:`verify`). Partial reads without
    EOF never raise, so callers that need a hard guarantee must consume the
    stream fully or call :meth:`verify` after a full read.
    """

    def __init__(self, stream: BinaryIO, expected_digest: str) -> None:
        self._stream = stream
        self._expected = _require_digest(expected_digest)
        self._hasher = hashlib.sha256()
        self._eof = False
        self._actual: str | None = None

    def readable(self) -> bool:
        return True

    def _read_chunk(self, size: int) -> bytes | None:
        """Read the next chunk from the wrapped stream; None at EOF (hashed)."""
        chunk = self._stream.read(size)
        if not chunk:
            return None
        if isinstance(chunk, str):
            chunk = chunk.encode("utf-8")
        data = bytes(chunk)
        self._hasher.update(data)
        return data

    def read(self, size: int = -1) -> bytes:
        """Read up to ``size`` bytes (all remaining when negative/omitted).

        Unlike raw streams, this loops until ``size`` bytes are collected or
        EOF is reached, so a single ``read()`` always triggers finalization
        (and the mismatch error) for a fully consumed stream.
        """
        if self._eof:
            return b""
        if size == 0:
            return b""
        if size is None or size < 0:
            chunks: list[bytes] = []
            while True:
                chunk = self._read_chunk(DEFAULT_CHUNK_SIZE)
                if chunk is None:
                    self._finalize()
                    return b"".join(chunks)
                chunks.append(chunk)
        buf = bytearray()
        while len(buf) < size:
            chunk = self._read_chunk(size - len(buf))
            if chunk is None:
                self._finalize()
                break
            buf += chunk
        return bytes(buf)

    def readinto(self, buffer: Any) -> int:
        data = self.read(len(buffer))
        count = len(data)
        buffer[:count] = data
        return count

    def _finalize(self) -> None:
        if self._eof:
            return
        self._eof = True
        self._actual = self._hasher.hexdigest()
        if self._actual != self._expected:
            raise ArtifactIntegrityError(self._expected, self._actual)

    def verify(self) -> str:
        """Consume the remainder of the stream and return the verified digest.

        Raises:
            ArtifactIntegrityError: If the recomputed hash mismatches.
        """
        while self.read(DEFAULT_CHUNK_SIZE):
            pass
        assert self._actual is not None  # finalized by EOF read above
        return self._actual

    @property
    def verified(self) -> bool:
        """True once EOF was reached and the digest matched."""
        return self._eof and self._actual == self._expected

    def close(self) -> None:
        try:
            close = getattr(self._stream, "close", None)
            if callable(close):
                close()
        finally:
            super().close()

    def __enter__(self) -> VerifyingReader:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def __iter__(self) -> Iterator[bytes]:
        while True:
            chunk = self.read(DEFAULT_CHUNK_SIZE)
            if not chunk:
                return
            yield chunk


class ArtifactStore:
    """Content-addressed store for immutable research artifacts.

    The store is idempotent: uploading identical bytes twice yields the same
    digest and reuses the stored object (the first record wins; a second
    ``put`` with the same bytes returns the existing record).
    """

    def __init__(self, backend: ObjectBackend, *, key_prefix: str = "") -> None:
        """Create a store over ``backend`` with an optional key prefix."""
        self._backend = backend
        self._prefix = key_prefix.strip("/")

    @property
    def backend(self) -> ObjectBackend:
        """Underlying object backend."""
        return self._backend

    @property
    def key_prefix(self) -> str:
        """Key prefix applied to every object key."""
        return self._prefix

    # -- layout helpers -------------------------------------------------

    def blob_key(self, digest: str) -> str:
        """Object key for the bytes of ``digest``."""
        return blob_key_for_digest(digest, self._prefix)

    def record_key(self, digest: str) -> str:
        """Object key for the metadata sidecar of ``digest``."""
        return record_key_for_digest(digest, self._prefix)

    # -- writes ---------------------------------------------------------

    def put_bytes(
        self,
        data: bytes,
        *,
        kind: ArtifactKind | str = ArtifactKind.RESULT,
        media_type: str = "application/octet-stream",
        created_by_run_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactRecord:
        """Store an in-memory payload; return its artifact record."""
        if isinstance(data, bytearray):
            data = bytes(data)
        if not isinstance(data, (bytes, memoryview)):
            raise TypeError(f"data must be bytes-like, got {type(data).__name__}")
        raw = bytes(data)
        digest, size = hash_bytes(raw)
        return self._store_verified(digest, size, io.BytesIO(raw), kind=kind, media_type=media_type, created_by_run_id=created_by_run_id, metadata=metadata)

    def put_stream(
        self,
        stream: BinaryIO,
        *,
        kind: ArtifactKind | str = ArtifactKind.RESULT,
        media_type: str = "application/octet-stream",
        created_by_run_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        content_length: int | None = None,
    ) -> ArtifactRecord:
        """Store a binary stream without loading it fully into memory.

        The stream is hashed in one pass while spooling to a temp file so
        memory stays bounded for arbitrarily large uploads; the hashed byte
        count is authoritative (the ``content_length`` hint is accepted for
        API symmetry and cross-checked against it).
        """
        kind_value = self._coerce_kind(kind)
        digest, size, spool_path = self._spool_and_hash(stream, chunk_size)
        if content_length is not None and content_length != size:
            try:
                os.unlink(spool_path)
            except OSError:
                pass
            raise ArtifactIntegrityError(digest, f"<size mismatch: hint={content_length} hashed={size}>")
        try:
            return self._store_from(scratch_path=spool_path, digest=digest, size=size, kind=kind_value, media_type=media_type, created_by_run_id=created_by_run_id, metadata=metadata, content_length=size)
        finally:
            try:
                os.unlink(spool_path)
            except OSError:
                pass

    def put_file(
        self,
        path: str | os.PathLike[str],
        *,
        kind: ArtifactKind | str = ArtifactKind.RESULT,
        media_type: str = "application/octet-stream",
        created_by_run_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactRecord:
        """Store a file from disk; return its artifact record."""
        kind_value = self._coerce_kind(kind)
        file_path = os.fspath(path)
        hasher = hashlib.sha256()
        size = 0
        with open(file_path, "rb") as handle:  # noqa: PTH123 — caller-supplied path API
            while True:
                chunk = handle.read(DEFAULT_CHUNK_SIZE)
                if not chunk:
                    break
                hasher.update(chunk)
                size += len(chunk)
        digest = hasher.hexdigest()
        return self._store_from(scratch_path=file_path, digest=digest, size=size, kind=kind_value, media_type=media_type, created_by_run_id=created_by_run_id, metadata=metadata, content_length=size)

    def _spool_and_hash(self, stream: BinaryIO, chunk_size: int) -> tuple[str, int, str]:
        import tempfile  # noqa: PLC0415 — local import keeps module import light

        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        fd, spool_path = tempfile.mkstemp(prefix="artifact-spool-")
        digest = hashlib.sha256()
        size = 0
        try:
            with os.fdopen(fd, "wb") as tmp:
                while True:
                    chunk = stream.read(chunk_size)
                    if not chunk:
                        break
                    if isinstance(chunk, str):
                        chunk = chunk.encode("utf-8")
                    digest.update(chunk)
                    tmp.write(chunk)
                    size += len(chunk)
        except BaseException:
            try:
                os.unlink(spool_path)
            except OSError:
                pass
            raise
        return digest.hexdigest(), size, spool_path

    def _store_from(
        self,
        *,
        scratch_path: str,
        digest: str,
        size: int,
        kind: ArtifactKind,
        media_type: str,
        created_by_run_id: str | None,
        metadata: dict[str, Any] | None,
        content_length: int,
    ) -> ArtifactRecord:
        existing = self._try_get_record(digest)
        if existing is not None and self._backend.exists(self.blob_key(digest)):
            return existing
        key = self.blob_key(digest)
        if not self._backend.exists(key):
            with open(scratch_path, "rb") as handle:  # noqa: PTH123 — store-managed scratch path
                self._backend.put_object(key, handle, content_type=media_type, content_length=content_length)
        # Post-write size check guards against truncated backend writes.
        stored_size = self._backend.stat(key)
        if stored_size != size:
            raise ArtifactIntegrityError(digest, f"<size mismatch: stored={stored_size} computed={size}>")
        record = ArtifactRecord(
            sha256=digest,
            kind=kind,
            storage_uri=self._backend.storage_uri(key),
            media_type=media_type,
            byte_size=size,
            created_by_run_id=created_by_run_id,
            metadata=dict(metadata or {}),
        )
        self._put_record(record)
        return record

    def _store_verified(
        self,
        digest: str,
        size: int,
        stream: BinaryIO,
        *,
        kind: ArtifactKind | str,
        media_type: str,
        created_by_run_id: str | None,
        metadata: dict[str, Any] | None,
    ) -> ArtifactRecord:
        kind_value = self._coerce_kind(kind)
        existing = self._try_get_record(digest)
        key = self.blob_key(digest)
        if existing is not None and self._backend.exists(key):
            return existing
        if not self._backend.exists(key):
            self._backend.put_object(key, stream, content_type=media_type, content_length=size)
        stored_size = self._backend.stat(key)
        if stored_size != size:
            raise ArtifactIntegrityError(digest, f"<size mismatch: stored={stored_size} computed={size}>")
        record = ArtifactRecord(
            sha256=digest,
            kind=kind_value,
            storage_uri=self._backend.storage_uri(key),
            media_type=media_type,
            byte_size=size,
            created_by_run_id=created_by_run_id,
            metadata=dict(metadata or {}),
        )
        self._put_record(record)
        return record

    @staticmethod
    def _coerce_kind(kind: ArtifactKind | str) -> ArtifactKind:
        if isinstance(kind, ArtifactKind):
            return kind
        try:
            return ArtifactKind(str(kind))
        except ValueError as exc:
            valid = ", ".join(item.value for item in ArtifactKind)
            raise ValueError(f"invalid artifact kind {kind!r}; expected one of: {valid}") from exc

    def _put_record(self, record: ArtifactRecord) -> None:
        payload = json.dumps(record.to_dict(), sort_keys=True, ensure_ascii=False).encode("utf-8")
        self._backend.put_object(self.record_key(record.sha256), io.BytesIO(payload), content_type="application/json", content_length=len(payload))

    # -- reads ----------------------------------------------------------

    def exists(self, digest_or_uri: str) -> bool:
        """Return True when the blob for ``digest_or_uri`` is stored."""
        digest = parse_artifact_uri(digest_or_uri)
        return self._backend.exists(self.blob_key(digest))

    def get(self, digest_or_uri: str) -> ArtifactRecord:
        """Return the metadata record for ``digest_or_uri``.

        Raises:
            ArtifactNotFoundError: If no blob (or no record) is stored.
        """
        digest = parse_artifact_uri(digest_or_uri)
        if not self._backend.exists(self.blob_key(digest)):
            raise ArtifactNotFoundError(artifact_uri(digest))
        record = self._try_get_record(digest)
        if record is None:
            raise ArtifactNotFoundError(artifact_uri(digest))
        return record

    def _try_get_record(self, digest: str) -> ArtifactRecord | None:
        try:
            raw = self._backend.get_object(self.record_key(digest))
        except KeyError:
            return None
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict):
            return None
        try:
            return ArtifactRecord.from_dict(payload)
        except ValueError:
            return None

    def get_bytes(self, digest_or_uri: str, *, verify: bool = True) -> bytes:
        """Return the full bytes for ``digest_or_uri``, re-hashing on read.

        Raises:
            ArtifactNotFoundError: If the blob is missing.
            ArtifactIntegrityError: If ``verify`` is true and the recomputed
                digest differs from the requested one.
        """
        digest = parse_artifact_uri(digest_or_uri)
        key = self.blob_key(digest)
        try:
            data = self._backend.get_object(key)
        except KeyError:
            raise ArtifactNotFoundError(artifact_uri(digest)) from None
        if verify:
            actual = hashlib.sha256(data).hexdigest()
            if actual != digest:
                raise ArtifactIntegrityError(digest, actual)
        return data

    def open(self, digest_or_uri: str, *, verify: bool = True) -> BinaryIO:
        """Open a streaming reader for ``digest_or_uri`` (caller closes it).

        With ``verify=True`` (default) the stream re-hashes bytes while
        reading and raises :class:`ArtifactIntegrityError` at EOF on
        mismatch; consume the stream fully (or call ``verify()`` on the
        returned reader) to enforce the guarantee.
        """
        digest = parse_artifact_uri(digest_or_uri)
        try:
            stream = self._backend.open_stream(self.blob_key(digest))
        except KeyError:
            raise ArtifactNotFoundError(artifact_uri(digest)) from None
        if not verify:
            return stream
        return VerifyingReader(stream, digest)  # type: ignore[return-value]

    def download_to(self, digest_or_uri: str, dest: str | os.PathLike[str], *, verify: bool = True, chunk_size: int = DEFAULT_CHUNK_SIZE) -> ArtifactRecord:
        """Stream an artifact to a local file; return its record.

        The destination is written atomically (temp file + rename) and the
        digest is verified during the copy when ``verify`` is true.
        """
        import tempfile  # noqa: PLC0415 — local import keeps module import light

        record = self.get(digest_or_uri)
        dest_path = os.fspath(dest)
        parent = os.path.dirname(os.path.abspath(dest_path))
        os.makedirs(parent, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=parent, prefix=".artifact-download-", suffix=".part")
        try:
            with os.fdopen(fd, "wb") as out:
                with self.open(record.sha256, verify=verify) as stream:
                    while True:
                        chunk = stream.read(chunk_size)
                        if not chunk:
                            break
                        out.write(chunk)
            os.replace(tmp_name, dest_path)
        except BaseException:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise
        return record

    def verify(self, digest_or_uri: str) -> bool:
        """Re-hash the stored blob and return True when it matches.

        Raises:
            ArtifactNotFoundError: If the blob is missing.
            ArtifactIntegrityError: If the recomputed digest differs.
        """
        digest = parse_artifact_uri(digest_or_uri)
        try:
            stream = self._backend.open_stream(self.blob_key(digest))
        except KeyError:
            raise ArtifactNotFoundError(artifact_uri(digest)) from None
        hasher = hashlib.sha256()
        try:
            while True:
                chunk = stream.read(DEFAULT_CHUNK_SIZE)
                if not chunk:
                    break
                hasher.update(chunk)
        finally:
            close = getattr(stream, "close", None)
            if callable(close):
                close()
        actual = hasher.hexdigest()
        if actual != digest:
            raise ArtifactIntegrityError(digest, actual)
        return True
