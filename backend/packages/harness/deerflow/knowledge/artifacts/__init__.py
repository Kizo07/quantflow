"""SHA-256 content-addressed artifact store (Knowledge Plane).

Artifacts are the immutable evidence layer: every dataset snapshot, code
bundle, notebook, result file, log, chart, or environment lock is stored
under the SHA-256 digest of its bytes and addressed as
``artifact://sha256/<digest>``.

Quick start::

    from deerflow.knowledge.artifacts import ArtifactStore, LocalFilesystemBackend

    store = ArtifactStore(LocalFilesystemBackend("/var/lib/quantflow/artifacts"))
    record = store.put_bytes(b"backtest results", kind="result", media_type="text/plain")
    assert store.exists(record.sha256)
    payload = store.get_bytes(record.uri)  # re-hashed on read

Against MinIO/S3 (``boto3`` preferred, ``minio`` fallback)::

    from deerflow.knowledge.artifacts import ArtifactStore, create_backend

    backend = create_backend(
        "s3://quantflow-evidence",
        endpoint_url="http://127.0.0.1:9000",
        aws_access_key_id="minioadmin",
        aws_secret_access_key="minioadmin",
    )
    store = ArtifactStore(backend, key_prefix="prod")
"""

from .backends import Boto3S3Backend, LocalFilesystemBackend, MinioBackend, ObjectBackend, create_backend
from .models import ARTIFACT_URI_SCHEME, ArtifactBackendError, ArtifactError, ArtifactIntegrityError, ArtifactKind, ArtifactNotFoundError, ArtifactRecord, artifact_uri
from .store import RECORD_SUFFIX, ArtifactStore, VerifyingReader, blob_key_for_digest, hash_bytes, hash_stream, parse_artifact_uri, record_key_for_digest

__all__ = [
    "ARTIFACT_URI_SCHEME",
    "RECORD_SUFFIX",
    "ArtifactBackendError",
    "ArtifactError",
    "ArtifactIntegrityError",
    "ArtifactKind",
    "ArtifactNotFoundError",
    "ArtifactRecord",
    "ArtifactStore",
    "Boto3S3Backend",
    "LocalFilesystemBackend",
    "MinioBackend",
    "ObjectBackend",
    "VerifyingReader",
    "artifact_uri",
    "blob_key_for_digest",
    "create_backend",
    "hash_bytes",
    "hash_stream",
    "parse_artifact_uri",
    "record_key_for_digest",
]
