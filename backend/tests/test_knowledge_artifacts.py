"""Tests for the content-addressed artifact store (Knowledge Plane).

All tests run against the local-filesystem backend and injected fake S3
clients, so no MinIO/S3 service (and neither ``boto3`` nor ``minio``) is
required.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, BinaryIO

import pytest

from deerflow.knowledge.artifacts import (
    ARTIFACT_URI_SCHEME,
    ArtifactBackendError,
    ArtifactIntegrityError,
    ArtifactKind,
    ArtifactNotFoundError,
    ArtifactRecord,
    ArtifactStore,
    Boto3S3Backend,
    LocalFilesystemBackend,
    MinioBackend,
    artifact_uri,
    blob_key_for_digest,
    create_backend,
    hash_bytes,
    hash_stream,
    parse_artifact_uri,
    record_key_for_digest,
)
from deerflow.knowledge.artifacts.store import VerifyingReader

DIGEST_HELLO = hashlib.sha256(b"hello knowledge plane").hexdigest()


def make_store(tmp_path: Path, **kwargs: Any) -> ArtifactStore:
    return ArtifactStore(LocalFilesystemBackend(tmp_path / "objects"), **kwargs)


# -- hashing helpers ----------------------------------------------------


def test_hash_bytes_matches_hashlib() -> None:
    digest, size = hash_bytes(b"abc")
    assert digest == hashlib.sha256(b"abc").hexdigest()
    assert size == 3


def test_hash_bytes_empty_payload() -> None:
    digest, size = hash_bytes(b"")
    assert digest == hashlib.sha256(b"").hexdigest()
    assert size == 0


def test_hash_stream_chunks_and_counts() -> None:
    payload = os.urandom(100_000)
    digest, size = hash_stream(io.BytesIO(payload), chunk_size=4096)
    assert digest == hashlib.sha256(payload).hexdigest()
    assert size == len(payload)


def test_hash_stream_rejects_non_positive_chunk() -> None:
    with pytest.raises(ValueError):
        hash_stream(io.BytesIO(b"x"), chunk_size=0)


# -- key layout + URIs ----------------------------------------------------


def test_blob_key_fanout_and_prefix() -> None:
    digest = "aa" + "bb" + "c" * 60
    assert blob_key_for_digest(digest) == f"sha256/aa/bb/{digest}"
    assert blob_key_for_digest(digest, "prod") == f"prod/sha256/aa/bb/{digest}"
    assert record_key_for_digest(digest).endswith(".record.json")


def test_blob_key_rejects_bad_digest() -> None:
    with pytest.raises(ValueError):
        blob_key_for_digest("not-a-digest")


def test_parse_artifact_uri_accepts_uri_and_bare_digest() -> None:
    assert parse_artifact_uri(f"artifact://sha256/{DIGEST_HELLO}") == DIGEST_HELLO
    assert parse_artifact_uri(DIGEST_HELLO) == DIGEST_HELLO


def test_parse_artifact_uri_rejects_other_schemes() -> None:
    with pytest.raises(ValueError):
        parse_artifact_uri("s3://bucket/key")


def test_artifact_uri_scheme_constant() -> None:
    assert artifact_uri(DIGEST_HELLO) == f"{ARTIFACT_URI_SCHEME}{DIGEST_HELLO}"


# -- records ---------------------------------------------------------------


def test_record_roundtrip_to_dict() -> None:
    record = ArtifactRecord(sha256=DIGEST_HELLO, kind=ArtifactKind.CODE, storage_uri="s3://b/k", media_type="text/x-python", byte_size=7, created_by_run_id="run-1", metadata={"repo": "x"})
    clone = ArtifactRecord.from_dict(json.loads(json.dumps(record.to_dict())))
    assert clone == record
    assert clone.uri == f"artifact://sha256/{DIGEST_HELLO}"


def test_record_rejects_bad_digest_and_size() -> None:
    with pytest.raises(ValueError):
        ArtifactRecord(sha256="zz", byte_size=1)
    with pytest.raises(ValueError):
        ArtifactRecord(sha256=DIGEST_HELLO, byte_size=-1)


def test_record_from_dict_rejects_bad_kind_and_metadata() -> None:
    with pytest.raises(ValueError):
        ArtifactRecord.from_dict({"sha256": DIGEST_HELLO, "byte_size": 1, "kind": "nope"})
    with pytest.raises(ValueError):
        ArtifactRecord.from_dict({"sha256": DIGEST_HELLO, "byte_size": 1, "kind": "result", "metadata": [1]})


# -- store: writes -----------------------------------------------------------


def test_put_bytes_roundtrip_record_fields(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    record = store.put_bytes(b"hello knowledge plane", kind="result", media_type="text/plain", created_by_run_id="run-42", metadata={"seed": 7})
    assert record.sha256 == DIGEST_HELLO
    assert record.kind is ArtifactKind.RESULT
    assert record.media_type == "text/plain"
    assert record.byte_size == len(b"hello knowledge plane")
    assert record.created_by_run_id == "run-42"
    assert record.metadata == {"seed": 7}
    assert record.storage_uri.startswith("file://")
    assert record.uri == f"artifact://sha256/{DIGEST_HELLO}"
    assert store.exists(record.sha256)
    assert store.exists(record.uri)


def test_put_bytes_is_content_addressed_and_idempotent(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    first = store.put_bytes(b"same bytes", kind=ArtifactKind.LOG)
    second = store.put_bytes(b"same bytes", kind=ArtifactKind.LOG)
    assert first.sha256 == second.sha256
    assert first.artifact_id == second.artifact_id  # first record wins
    other = store.put_bytes(b"different bytes", kind=ArtifactKind.LOG)
    assert other.sha256 != first.sha256


def test_put_bytes_rejects_text_input(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    with pytest.raises(TypeError):
        store.put_bytes("not bytes")  # type: ignore[arg-type]


def test_put_bytes_rejects_unknown_kind(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    with pytest.raises(ValueError, match="invalid artifact kind"):
        store.put_bytes(b"x", kind="mystery")


def test_put_stream_large_and_non_seekable(tmp_path: Path) -> None:
    payload = os.urandom(2 * 1024 * 1024 + 123)

    class NonSeekable(io.RawIOBase):
        def __init__(self, data: bytes) -> None:
            self._view = memoryview(data)
            self._pos = 0

        def readable(self) -> bool:
            return True

        def read(self, size: int = -1) -> bytes:
            if self._pos >= len(self._view):
                return b""
            end = len(self._view) if size is None or size < 0 else min(len(self._view), self._pos + size)
            chunk = bytes(self._view[self._pos : end])
            self._pos = end
            return chunk

    store = make_store(tmp_path)
    record = store.put_stream(NonSeekable(payload), kind="dataset_snapshot", media_type="application/octet-stream", chunk_size=65536)  # type: ignore[arg-type]
    assert record.sha256 == hashlib.sha256(payload).hexdigest()
    assert record.byte_size == len(payload)
    assert store.get_bytes(record.sha256) == payload


def test_put_stream_content_length_hint_mismatch(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    with pytest.raises(ArtifactIntegrityError):
        store.put_stream(io.BytesIO(b"12345"), content_length=999)


def test_put_file_roundtrip(tmp_path: Path) -> None:
    src = tmp_path / "input.bin"
    src.write_bytes(b"file payload \x00\xff" * 1000)
    store = make_store(tmp_path)
    record = store.put_file(src, kind=ArtifactKind.NOTEBOOK, media_type="application/json")
    assert record.byte_size == src.stat().st_size
    assert store.get_bytes(record.uri) == src.read_bytes()


def test_put_file_missing_raises(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    with pytest.raises(FileNotFoundError):
        store.put_file(tmp_path / "absent.bin")


# -- store: reads --------------------------------------------------------------


def test_get_returns_record_and_missing_raises(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    record = store.put_bytes(b"find me", kind="chart", media_type="image/png")
    assert store.get(record.uri) == record
    assert not store.exists(hashlib.sha256(b"absent").hexdigest())
    with pytest.raises(ArtifactNotFoundError):
        store.get(hashlib.sha256(b"absent").hexdigest())
    with pytest.raises(ArtifactNotFoundError):
        store.get_bytes(hashlib.sha256(b"absent").hexdigest())


def test_get_bytes_rehashes_and_detects_tampering(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    record = store.put_bytes(b"pristine", kind="result")
    assert store.get_bytes(record.sha256) == b"pristine"
    # Tamper with the blob behind the store's back.
    backend = store.backend
    assert isinstance(backend, LocalFilesystemBackend)
    blob_path = backend.root / Path(store.blob_key(record.sha256))
    blob_path.write_bytes(b"tampered!")
    with pytest.raises(ArtifactIntegrityError) as excinfo:
        store.get_bytes(record.sha256)
    assert excinfo.value.expected_digest == record.sha256
    # Unverified read still returns raw bytes for forensics.
    assert store.get_bytes(record.sha256, verify=False) == b"tampered!"


def test_open_streams_with_verification(tmp_path: Path) -> None:
    payload = os.urandom(500_000)
    store = make_store(tmp_path)
    record = store.put_bytes(payload, kind="source")
    with store.open(record.uri) as stream:
        chunks = []
        while True:
            chunk = stream.read(65536)
            if not chunk:
                break
            chunks.append(chunk)
    assert b"".join(chunks) == payload


def test_open_missing_raises_not_found(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    with pytest.raises(ArtifactNotFoundError):
        store.open(DIGEST_HELLO)


def test_open_without_verify_returns_raw_stream(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    record = store.put_bytes(b"raw", kind="log")
    with store.open(record.sha256, verify=False) as stream:
        assert stream.read() == b"raw"


def test_verify_true_and_tampered_raises(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    record = store.put_bytes(b"verify me", kind="environment")
    assert store.verify(record.uri) is True
    backend = store.backend
    assert isinstance(backend, LocalFilesystemBackend)
    (backend.root / Path(store.blob_key(record.sha256))).write_bytes(b"bad")
    with pytest.raises(ArtifactIntegrityError):
        store.verify(record.sha256)
    with pytest.raises(ArtifactNotFoundError):
        store.verify(hashlib.sha256(b"nope").hexdigest())


def test_download_to_writes_atomically_and_returns_record(tmp_path: Path) -> None:
    payload = os.urandom(300_000)
    store = make_store(tmp_path)
    record = store.put_bytes(payload, kind="result")
    dest = tmp_path / "nested" / "out.bin"
    returned = store.download_to(record.uri, dest)
    assert returned == record
    assert dest.read_bytes() == payload
    # No staging files leak into the destination directory.
    assert [p.name for p in (tmp_path / "nested").iterdir()] == ["out.bin"]


def test_records_survive_new_store_instance(tmp_path: Path) -> None:
    root = tmp_path / "objects"
    first = ArtifactStore(LocalFilesystemBackend(root))
    record = first.put_bytes(b"persistent", kind="code", metadata={"v": 1})
    second = ArtifactStore(LocalFilesystemBackend(root))
    assert second.get(record.sha256) == record
    assert second.get_bytes(record.sha256) == b"persistent"


def test_key_prefix_isolation(tmp_path: Path) -> None:
    root = tmp_path / "objects"
    backend = LocalFilesystemBackend(root)
    prod = ArtifactStore(backend, key_prefix="prod")
    dev = ArtifactStore(backend, key_prefix="dev")
    record = prod.put_bytes(b"shared bytes", kind="result")
    assert record.storage_uri.replace("\\", "/").endswith(f"prod/sha256/{record.sha256[0:2]}/{record.sha256[2:4]}/{record.sha256}")
    assert not dev.exists(record.sha256)
    dev_record = dev.put_bytes(b"shared bytes", kind="result")
    assert dev_record.sha256 == record.sha256
    assert dev_record.storage_uri != record.storage_uri


# -- VerifyingReader -------------------------------------------------------------


def test_verifying_reader_accepts_matching_stream() -> None:
    payload = b"streamed truth" * 1000
    reader = VerifyingReader(io.BytesIO(payload), hashlib.sha256(payload).hexdigest())
    assert reader.verify() == hashlib.sha256(payload).hexdigest()
    assert reader.verified


def test_verifying_reader_raises_at_eof_on_mismatch() -> None:
    reader = VerifyingReader(io.BytesIO(b"corrupt"), hashlib.sha256(b"original").hexdigest())
    assert reader.read(3) == b"cor"  # partial reads never raise
    with pytest.raises(ArtifactIntegrityError):
        reader.read()  # EOF read finalizes and raises


def test_store_open_detects_tampering_on_full_read(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    record = store.put_bytes(b"good bytes here", kind="result")
    backend = store.backend
    assert isinstance(backend, LocalFilesystemBackend)
    (backend.root / Path(store.blob_key(record.sha256))).write_bytes(b"evil bytes here!")
    with store.open(record.sha256) as stream:
        assert isinstance(stream, VerifyingReader)
        with pytest.raises(ArtifactIntegrityError):
            stream.verify()


# -- local backend ------------------------------------------------------------------


def test_local_backend_crud_and_stat(tmp_path: Path) -> None:
    backend = LocalFilesystemBackend(tmp_path / "b")
    uri = backend.put_object("a/b/c.bin", io.BytesIO(b"data"), content_type="text/plain")
    assert uri.startswith("file://")
    assert backend.exists("a/b/c.bin")
    assert backend.get_object("a/b/c.bin") == b"data"
    assert backend.stat("a/b/c.bin") == 4
    with backend.open_stream("a/b/c.bin") as stream:
        assert stream.read() == b"data"
    backend.delete("a/b/c.bin")
    assert not backend.exists("a/b/c.bin")
    backend.delete("a/b/c.bin")  # missing delete is a no-op
    with pytest.raises(KeyError):
        backend.get_object("a/b/c.bin")
    with pytest.raises(KeyError):
        backend.stat("a/b/c.bin")
    with pytest.raises(KeyError):
        backend.open_stream("a/b/c.bin")


def test_local_backend_rejects_path_traversal(tmp_path: Path) -> None:
    backend = LocalFilesystemBackend(tmp_path / "b")
    with pytest.raises(ValueError):
        backend.put_object("../escape.bin", io.BytesIO(b"x"))
    with pytest.raises(ValueError):
        backend.get_object("/abs.bin")


def test_local_backend_atomic_write_leaves_no_part_files(tmp_path: Path) -> None:
    backend = LocalFilesystemBackend(tmp_path / "b")

    class Exploding(io.BytesIO):
        def read(self, size: int = -1) -> bytes:
            raise RuntimeError("boom")

    with pytest.raises(ArtifactBackendError):
        backend.put_object("x.bin", Exploding(b"data"))
    leftovers = [p for p in (tmp_path / "b").rglob("*") if p.is_file()]
    assert leftovers == []


# -- fake S3 clients -------------------------------------------------------------------


class FakeNoSuchKey(Exception):
    """Boto3-style missing-key error (name drives detection)."""


class FakeBoto3Client:
    """Minimal dict-backed stand-in for a boto3 S3 client."""

    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], bytes] = {}
        self.buckets: set[str] = set()

    def head_bucket(self, Bucket: str) -> dict[str, Any]:
        if Bucket not in self.buckets:
            raise FakeNoSuchKey(f"NoSuchBucket: {Bucket}")
        return {}

    def create_bucket(self, Bucket: str) -> dict[str, Any]:
        self.buckets.add(Bucket)
        return {}

    def put_object(self, Bucket: str, Key: str, Body: Any, **kwargs: Any) -> dict[str, Any]:
        del kwargs
        data = Body.read() if hasattr(Body, "read") else bytes(Body)
        self.objects[(Bucket, Key)] = bytes(data)
        return {"ETag": '"fake"'}

    def get_object(self, Bucket: str, Key: str) -> dict[str, Any]:
        try:
            return {"Body": io.BytesIO(self.objects[(Bucket, Key)])}
        except KeyError:
            raise FakeNoSuchKey(f"NoSuchKey: {Key}") from None

    def head_object(self, Bucket: str, Key: str) -> dict[str, Any]:
        try:
            return {"ContentLength": len(self.objects[(Bucket, Key)])}
        except KeyError:
            raise FakeNoSuchKey(f"NoSuchKey: {Key}") from None

    def delete_object(self, Bucket: str, Key: str) -> dict[str, Any]:
        self.objects.pop((Bucket, Key), None)
        return {}


class FakeS3Error(Exception):
    """Minio-style error carrying the service message."""


class FakeMinioClient:
    """Minimal dict-backed stand-in for a minio.Minio client."""

    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], bytes] = {}
        self.buckets: set[str] = set()

    def bucket_exists(self, bucket: str) -> bool:
        return bucket in self.buckets

    def make_bucket(self, bucket: str) -> None:
        self.buckets.add(bucket)

    def put_object(self, bucket: str, key: str, body: BinaryIO, length: int, content_type: str = "application/octet-stream") -> SimpleNamespace:
        del content_type
        data = body.read()
        assert len(data) == length
        self.objects[(bucket, key)] = bytes(data)
        return SimpleNamespace(bucket_name=bucket, object_name=key)

    def get_object(self, bucket: str, key: str) -> BinaryIO:
        try:
            return io.BytesIO(self.objects[(bucket, key)])
        except KeyError:
            raise FakeS3Error(f"NoSuchKey: {key}") from None

    def stat_object(self, bucket: str, key: str) -> SimpleNamespace:
        try:
            return SimpleNamespace(size=len(self.objects[(bucket, key)]))
        except KeyError:
            raise FakeS3Error(f"NoSuchKey: {key}") from None

    def remove_object(self, bucket: str, key: str) -> None:
        self.objects.pop((bucket, key), None)


def test_boto3_backend_with_injected_client() -> None:
    backend = Boto3S3Backend("evidence", client=FakeBoto3Client())
    uri = backend.put_object("k1", io.BytesIO(b"payload"), content_type="text/plain")
    assert uri == "s3://evidence/k1"
    assert backend.exists("k1")
    assert backend.get_object("k1") == b"payload"
    assert backend.stat("k1") == 7
    with backend.open_stream("k1") as stream:
        assert stream.read() == b"payload"
    backend.delete("k1")
    assert not backend.exists("k1")
    with pytest.raises(KeyError):
        backend.get_object("k1")
    with pytest.raises(KeyError):
        backend.stat("k1")


def test_boto3_backend_reads_bytes_body() -> None:
    class BytesBodyClient(FakeBoto3Client):
        def get_object(self, Bucket: str, Key: str) -> dict[str, Any]:
            return {"Body": b"raw-bytes"}

    backend = Boto3S3Backend("evidence", client=BytesBodyClient(), ensure_bucket=False)
    assert backend.get_object("anything") == b"raw-bytes"
    with backend.open_stream("anything") as stream:
        assert stream.read() == b"raw-bytes"


def test_boto3_backend_requires_bucket() -> None:
    with pytest.raises(ValueError):
        Boto3S3Backend("", client=FakeBoto3Client())


def test_minio_backend_with_injected_client() -> None:
    backend = MinioBackend("evidence", client=FakeMinioClient())
    uri = backend.put_object("k1", io.BytesIO(b"payload"))
    assert uri == "s3://evidence/k1"
    assert backend.exists("k1")
    assert backend.get_object("k1") == b"payload"
    assert backend.stat("k1") == 7
    with backend.open_stream("k1") as stream:
        assert stream.read() == b"payload"
    backend.delete("k1")
    assert not backend.exists("k1")
    with pytest.raises(KeyError):
        backend.get_object("k1")
    with pytest.raises(KeyError):
        backend.stat("k1")


def test_minio_backend_spools_non_seekable_stream() -> None:
    class NonSeekable(io.RawIOBase):
        def __init__(self, data: bytes) -> None:
            self._stream = io.BytesIO(data)

        def readable(self) -> bool:
            return True

        def read(self, size: int = -1) -> bytes:
            return self._stream.read(size)

        def seek(self, *args: Any) -> int:
            raise OSError("non-seekable")

        def tell(self) -> int:
            raise OSError("non-seekable")

    backend = MinioBackend("evidence", client=FakeMinioClient())
    backend.put_object("k2", NonSeekable(b"spooled"))
    assert backend.get_object("k2") == b"spooled"


def test_minio_backend_requires_bucket() -> None:
    with pytest.raises(ValueError):
        MinioBackend("", client=FakeMinioClient())


def test_store_works_over_boto3_backend() -> None:
    store = ArtifactStore(Boto3S3Backend("evidence", client=FakeBoto3Client()))
    record = store.put_bytes(b"s3 bytes", kind="result", media_type="text/plain")
    assert record.storage_uri == f"s3://evidence/{store.blob_key(record.sha256)}"
    assert store.get(record.sha256) == record
    assert store.get_bytes(record.sha256) == b"s3 bytes"
    assert store.verify(record.uri) is True


def test_store_works_over_minio_backend() -> None:
    store = ArtifactStore(MinioBackend("evidence", client=FakeMinioClient()))
    record = store.put_stream(io.BytesIO(b"minio bytes"), kind="log")
    assert store.get_bytes(record.sha256) == b"minio bytes"


# -- backend factory ----------------------------------------------------------------------


def test_create_backend_local_path_and_file_uri(tmp_path: Path) -> None:
    backend = create_backend(tmp_path / "a")
    assert isinstance(backend, LocalFilesystemBackend)
    backend2 = create_backend((tmp_path / "b").as_uri())
    assert isinstance(backend2, LocalFilesystemBackend)


def test_create_backend_s3_and_minio_with_injected_clients() -> None:
    s3 = create_backend("s3://evidence", client=FakeBoto3Client())
    assert isinstance(s3, Boto3S3Backend)
    assert s3.bucket == "evidence"
    mio = create_backend("minio://research", client=FakeMinioClient())
    assert isinstance(mio, MinioBackend)
    assert mio.bucket == "research"


def test_create_backend_rejects_bad_specs() -> None:
    with pytest.raises(ValueError):
        create_backend("ftp://host/path")
    with pytest.raises(ValueError):
        create_backend("s3://")
    with pytest.raises(ValueError):
        create_backend("minio://")


def _optional_spec_missing(name: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(name) is None


def test_s3_drivers_without_client_require_optional_sdk() -> None:
    if _optional_spec_missing("boto3"):
        with pytest.raises(ImportError, match="boto3"):
            Boto3S3Backend("evidence", ensure_bucket=False)
    if _optional_spec_missing("minio"):
        with pytest.raises(ImportError, match="minio"):
            MinioBackend("evidence", endpoint="127.0.0.1:9000", access_key="a", secret_key="b", ensure_bucket=False)


def test_create_backend_s3_without_client_or_sdk_raises() -> None:
    if _optional_spec_missing("boto3") and _optional_spec_missing("minio"):
        with pytest.raises(ImportError):
            create_backend("s3://evidence", ensure_bucket=False)
    else:
        pytest.skip("optional S3 SDK installed; injected-client paths covered above")
