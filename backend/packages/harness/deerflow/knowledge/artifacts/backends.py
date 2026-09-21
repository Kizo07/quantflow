"""Object backends for the artifact store.

A backend is a thin binary-object layer (put/get/stream/exists/stat/delete)
over either a local directory or any S3-compatible endpoint (MinIO, AWS S3).
The artifact store layers content addressing, records, and integrity checks
on top of this interface.

Backend selection rule: ``boto3``-based S3 when available, else the
``minio`` client, else the local-filesystem backend. Both third-party SDKs
are optional lazy imports so the package (and its tests) work without them;
pre-constructed clients can also be injected for tests or custom wiring.
"""

from __future__ import annotations

import io
import os
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, BinaryIO
from urllib.parse import urlparse

from .models import ArtifactBackendError

#: Default chunk size for streamed copies (8 MiB).
DEFAULT_CHUNK_SIZE = 8 * 1024 * 1024


class ObjectBackend(ABC):
    """Abstract binary-object store keyed by opaque string keys."""

    @abstractmethod
    def put_object(self, key: str, stream: BinaryIO, *, content_type: str = "application/octet-stream", content_length: int | None = None) -> str:
        """Store bytes from ``stream`` under ``key``; return the storage URI."""

    @abstractmethod
    def get_object(self, key: str) -> bytes:
        """Return the full bytes stored under ``key``.

        Raises:
            KeyError: If ``key`` does not exist.
        """

    @abstractmethod
    def open_stream(self, key: str) -> BinaryIO:
        """Return a readable binary stream for ``key`` (caller closes it).

        Raises:
            KeyError: If ``key`` does not exist.
        """

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Return True when ``key`` exists."""

    @abstractmethod
    def stat(self, key: str) -> int:
        """Return the byte size stored under ``key``.

        Raises:
            KeyError: If ``key`` does not exist.
        """

    @abstractmethod
    def delete(self, key: str) -> None:
        """Delete ``key`` if present; missing keys are a no-op."""

    @abstractmethod
    def storage_uri(self, key: str) -> str:
        """Return the backend-specific URI locating ``key``."""


def _copy_stream(src: BinaryIO, dst: BinaryIO, chunk_size: int = DEFAULT_CHUNK_SIZE) -> int:
    """Copy ``src`` to ``dst`` in chunks; return the number of bytes copied."""
    total = 0
    while True:
        chunk = src.read(chunk_size)
        if not chunk:
            return total
        dst.write(chunk)
        total += len(chunk)


class LocalFilesystemBackend(ObjectBackend):
    """Local-directory backend for tests, development, and single-node use.

    Keys map to paths under ``root`` (``..`` segments are rejected). Writes
    are atomic: bytes land in a temp file in the destination directory and
    are moved into place with :func:`os.replace`.
    """

    def __init__(self, root: str | os.PathLike[str]) -> None:
        """Create a backend rooted at ``root`` (created when missing)."""
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    @property
    def root(self) -> Path:
        """Root directory backing this backend."""
        return self._root

    def _resolve(self, key: str) -> Path:
        if not key or key.startswith("/") or ".." in Path(key).parts:
            raise ValueError(f"invalid object key: {key!r}")
        return self._root.joinpath(*Path(key).parts)

    def put_object(self, key: str, stream: BinaryIO, *, content_type: str = "application/octet-stream", content_length: int | None = None) -> str:
        del content_type, content_length
        dest = self._resolve(key)
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_name = tempfile.mkstemp(dir=str(dest.parent), prefix=".upload-", suffix=".part")
            try:
                with os.fdopen(fd, "wb") as tmp:
                    _copy_stream(stream, tmp)
                os.replace(tmp_name, dest)
            except BaseException:
                try:
                    os.unlink(tmp_name)
                except OSError:
                    pass
                raise
        except (ArtifactBackendError, ValueError):
            raise
        except Exception as exc:
            raise ArtifactBackendError("put_object", exc) from exc
        return self.storage_uri(key)

    def get_object(self, key: str) -> bytes:
        path = self._resolve(key)
        try:
            return path.read_bytes()
        except FileNotFoundError:
            raise KeyError(key) from None
        except Exception as exc:
            raise ArtifactBackendError("get_object", exc) from exc

    def open_stream(self, key: str) -> BinaryIO:
        path = self._resolve(key)
        try:
            return path.open("rb")  # noqa: PTH123 — backend-owned path by design
        except FileNotFoundError:
            raise KeyError(key) from None
        except Exception as exc:
            raise ArtifactBackendError("open_stream", exc) from exc

    def exists(self, key: str) -> bool:
        return self._resolve(key).is_file()

    def stat(self, key: str) -> int:
        path = self._resolve(key)
        try:
            return path.stat().st_size
        except FileNotFoundError:
            raise KeyError(key) from None
        except Exception as exc:
            raise ArtifactBackendError("stat", exc) from exc

    def delete(self, key: str) -> None:
        try:
            self._resolve(key).unlink(missing_ok=True)
        except Exception as exc:
            raise ArtifactBackendError("delete", exc) from exc

    def storage_uri(self, key: str) -> str:
        return self._resolve(key).as_uri()


def _wrap_backend_errors(operation: str, func: Any, *args: Any, **kwargs: Any) -> Any:
    """Run an SDK call, translating failures to ArtifactBackendError/KeyError."""
    try:
        return func(*args, **kwargs)
    except KeyError:
        raise
    except ArtifactBackendError:
        raise
    except Exception as exc:
        if _looks_like_missing(exc):
            raise KeyError(str(exc)) from exc
        raise ArtifactBackendError(operation, exc) from exc


def _looks_like_missing(exc: BaseException) -> bool:
    """Best-effort missing-key detection across boto3/minio error shapes."""
    name = type(exc).__name__
    if name in {"NoSuchKey", "NoSuchBucket", "NotFound", "NoSuchObject"}:
        return True
    code = ""
    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        error = response.get("Error", {})
        if isinstance(error, dict):
            code = str(error.get("Code", ""))
    message = f"{name} {code} {exc}"
    return "NoSuchKey" in message or "NoSuchBucket" in message or "Not Found" in message or "404" in message


class Boto3S3Backend(ObjectBackend):
    """S3-compatible backend built on ``boto3`` (preferred S3 driver).

    Works against AWS S3 and S3-compatible endpoints such as MinIO by
    passing ``endpoint_url``. A pre-configured client may be injected via
    ``client`` (used by tests); otherwise ``boto3`` is imported lazily and
    a client is constructed from the remaining arguments.
    """

    def __init__(
        self,
        bucket: str,
        *,
        client: Any | None = None,
        endpoint_url: str | None = None,
        region_name: str | None = None,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        addressing_style: str = "auto",
        ensure_bucket: bool = True,
    ) -> None:
        """Create an S3 backend for ``bucket``.

        Args:
            bucket: Target bucket name (must already exist unless
                ``ensure_bucket`` is true and the credentials allow creation).
            client: Optional pre-configured boto3-compatible client. When
                omitted, ``boto3`` must be installed.
            endpoint_url: Custom S3 endpoint (e.g. MinIO ``http://host:9000``).
            region_name: AWS region for signature scoping.
            aws_access_key_id: Access key (falls back to the default chain).
            aws_secret_access_key: Secret key (falls back to the default chain).
            addressing_style: ``"auto"``, ``"virtual"``, or ``"path"``. MinIO
                behind some proxies requires ``"path"``.
            ensure_bucket: Create (or head) the bucket on construction.

        Raises:
            ImportError: If no client is injected and ``boto3`` is missing.
        """
        if not bucket:
            raise ValueError("bucket must not be empty")
        self._bucket = bucket
        if client is None:
            client = self._build_client(
                endpoint_url=endpoint_url,
                region_name=region_name,
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                addressing_style=addressing_style,
            )
        self._client = client
        if ensure_bucket:
            self._ensure_bucket()

    @staticmethod
    def _build_client(
        *,
        endpoint_url: str | None,
        region_name: str | None,
        aws_access_key_id: str | None,
        aws_secret_access_key: str | None,
        addressing_style: str,
    ) -> Any:
        try:
            import boto3  # noqa: PLC0415 — optional dependency, lazy by design
            from botocore.config import Config  # noqa: PLC0415 — optional dependency
        except ImportError as exc:
            raise ImportError("boto3 is required for Boto3S3Backend; install boto3 or use LocalFilesystemBackend/MinioBackend") from exc
        config_kwargs: dict[str, Any] = {}
        if addressing_style != "auto":
            if addressing_style not in {"virtual", "path"}:
                raise ValueError(f"addressing_style must be auto|virtual|path, got {addressing_style!r}")
            config_kwargs["s3"] = {"addressing_style": addressing_style}
        session_kwargs: dict[str, Any] = {}
        if aws_access_key_id is not None:
            session_kwargs["aws_access_key_id"] = aws_access_key_id
        if aws_secret_access_key is not None:
            session_kwargs["aws_secret_access_key"] = aws_secret_access_key
        if region_name is not None:
            session_kwargs["region_name"] = region_name
        session = boto3.session.Session(**session_kwargs)
        client_kwargs: dict[str, Any] = {"service_name": "s3"}
        if endpoint_url is not None:
            client_kwargs["endpoint_url"] = endpoint_url
        if config_kwargs:
            client_kwargs["config"] = Config(**config_kwargs)
        return session.client(**client_kwargs)

    def _ensure_bucket(self) -> None:
        try:
            self._client.head_bucket(Bucket=self._bucket)
        except Exception as exc:
            if not _looks_like_missing(exc):
                raise ArtifactBackendError("head_bucket", exc) from exc
            _wrap_backend_errors("create_bucket", self._client.create_bucket, Bucket=self._bucket)

    @property
    def bucket(self) -> str:
        """Bucket backing this backend."""
        return self._bucket

    def put_object(self, key: str, stream: BinaryIO, *, content_type: str = "application/octet-stream", content_length: int | None = None) -> str:
        body: BinaryIO | bytes = stream
        if content_length is None:
            data = stream.read()
            if isinstance(data, str):
                data = data.encode("utf-8")
            body = data
            content_length = len(data)
        extra: dict[str, Any] = {"ContentType": content_type}
        if content_length is not None:
            extra["ContentLength"] = content_length
        _wrap_backend_errors("put_object", self._client.put_object, Bucket=self._bucket, Key=key, Body=body, **extra)
        return self.storage_uri(key)

    def get_object(self, key: str) -> bytes:
        response = _wrap_backend_errors("get_object", self._client.get_object, Bucket=self._bucket, Key=key)
        body = response["Body"]
        if isinstance(body, (bytes, bytearray)):
            return bytes(body)
        try:
            data = body.read()
        finally:
            close = getattr(body, "close", None)
            if callable(close):
                close()
        if isinstance(data, str):
            data = data.encode("utf-8")
        return bytes(data)

    def open_stream(self, key: str) -> BinaryIO:
        response = _wrap_backend_errors("get_object", self._client.get_object, Bucket=self._bucket, Key=key)
        body = response["Body"]
        if isinstance(body, (bytes, bytearray)):
            return io.BytesIO(bytes(body))
        return body

    def exists(self, key: str) -> bool:
        try:
            self._client.head_object(Bucket=self._bucket, Key=key)
        except Exception as exc:
            if _looks_like_missing(exc):
                return False
            raise ArtifactBackendError("head_object", exc) from exc
        return True

    def stat(self, key: str) -> int:
        try:
            response = self._client.head_object(Bucket=self._bucket, Key=key)
        except Exception as exc:
            if _looks_like_missing(exc):
                raise KeyError(key) from exc
            raise ArtifactBackendError("head_object", exc) from exc
        try:
            return int(response["ContentLength"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ArtifactBackendError("head_object", exc) from exc

    def delete(self, key: str) -> None:
        _wrap_backend_errors("delete_object", self._client.delete_object, Bucket=self._bucket, Key=key)

    def storage_uri(self, key: str) -> str:
        return f"s3://{self._bucket}/{key}"


class MinioBackend(ObjectBackend):
    """S3-compatible backend built on the ``minio`` client (fallback S3 driver).

    Used when ``boto3`` is unavailable. A pre-configured client may be
    injected via ``client``; otherwise ``minio`` is imported lazily.
    """

    def __init__(
        self,
        bucket: str,
        *,
        client: Any | None = None,
        endpoint: str | None = None,
        access_key: str | None = None,
        secret_key: str | None = None,
        secure: bool = True,
        region: str | None = None,
        ensure_bucket: bool = True,
    ) -> None:
        """Create a MinIO backend for ``bucket``.

        Args:
            bucket: Target bucket name.
            client: Optional pre-configured ``Minio``-compatible client.
            endpoint: Server ``host:port`` (required when ``client`` is None).
            access_key: Access key (required when ``client`` is None).
            secret_key: Secret key (required when ``client`` is None).
            secure: Use HTTPS when building the client.
            region: Optional region hint.
            ensure_bucket: Create the bucket when missing.

        Raises:
            ImportError: If no client is injected and ``minio`` is missing.
        """
        if not bucket:
            raise ValueError("bucket must not be empty")
        self._bucket = bucket
        if client is None:
            client = self._build_client(endpoint=endpoint, access_key=access_key, secret_key=secret_key, secure=secure, region=region)
        self._client = client
        if ensure_bucket:
            self._ensure_bucket()

    @staticmethod
    def _build_client(*, endpoint: str | None, access_key: str | None, secret_key: str | None, secure: bool, region: str | None) -> Any:
        try:
            from minio import Minio  # noqa: PLC0415 — optional dependency, lazy by design
        except ImportError as exc:
            raise ImportError("minio is required for MinioBackend; install minio or use LocalFilesystemBackend/Boto3S3Backend") from exc
        if not endpoint or not access_key or not secret_key:
            raise ValueError("endpoint, access_key, and secret_key are required to build a minio client")
        kwargs: dict[str, Any] = {}
        if region is not None:
            kwargs["region"] = region
        return Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=secure, **kwargs)

    def _ensure_bucket(self) -> None:
        found = _wrap_backend_errors("bucket_exists", self._client.bucket_exists, self._bucket)
        if not found:
            _wrap_backend_errors("make_bucket", self._client.make_bucket, self._bucket)

    @property
    def bucket(self) -> str:
        """Bucket backing this backend."""
        return self._bucket

    def put_object(self, key: str, stream: BinaryIO, *, content_type: str = "application/octet-stream", content_length: int | None = None) -> str:
        length = content_length
        body = stream
        if length is None:
            length = _stream_length(stream)
        tmp_path: str | None = None
        if length is None:
            fd, tmp_path = tempfile.mkstemp(prefix="minio-spool-")
            try:
                with os.fdopen(fd, "wb") as tmp:
                    length = _copy_stream(stream, tmp)
                body = open(tmp_path, "rb")  # noqa: PTH123, SIM115 — closed in finally below
                _wrap_backend_errors("put_object", self._client.put_object, self._bucket, key, body, length, content_type=content_type)
            finally:
                close = getattr(body, "close", None)
                if callable(close) and body is not stream:
                    close()
                os.unlink(tmp_path)
            return self.storage_uri(key)
        _wrap_backend_errors("put_object", self._client.put_object, self._bucket, key, body, length, content_type=content_type)
        return self.storage_uri(key)

    def get_object(self, key: str) -> bytes:
        stream = self.open_stream(key)
        try:
            data = stream.read()
        finally:
            stream.close()
        if isinstance(data, str):
            data = data.encode("utf-8")
        return bytes(data)

    def open_stream(self, key: str) -> BinaryIO:
        try:
            response = self._client.get_object(self._bucket, key)
        except Exception as exc:
            if _looks_like_missing(exc):
                raise KeyError(key) from exc
            raise ArtifactBackendError("get_object", exc) from exc
        if isinstance(response, (bytes, bytearray)):
            return io.BytesIO(bytes(response))
        return response

    def exists(self, key: str) -> bool:
        try:
            self._client.stat_object(self._bucket, key)
        except Exception as exc:
            if _looks_like_missing(exc):
                return False
            raise ArtifactBackendError("stat_object", exc) from exc
        return True

    def stat(self, key: str) -> int:
        try:
            info = self._client.stat_object(self._bucket, key)
        except Exception as exc:
            if _looks_like_missing(exc):
                raise KeyError(key) from exc
            raise ArtifactBackendError("stat_object", exc) from exc
        size = getattr(info, "size", None)
        if size is None and isinstance(info, dict):
            size = info.get("size", info.get("ContentLength"))
        try:
            return int(size)  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise ArtifactBackendError("stat_object", exc) from exc

    def delete(self, key: str) -> None:
        _wrap_backend_errors("remove_object", self._client.remove_object, self._bucket, key)

    def storage_uri(self, key: str) -> str:
        return f"s3://{self._bucket}/{key}"


def _stream_length(stream: BinaryIO) -> int | None:
    """Best-effort content length for a seekable stream; None when unknown."""
    tell = getattr(stream, "tell", None)
    seek = getattr(stream, "seek", None)
    if not callable(tell) or not callable(seek):
        return None
    try:
        position = tell()
        seek(0, os.SEEK_END)
        end = tell()
        seek(position, os.SEEK_SET)
    except Exception:
        return None
    if end < position:
        return None
    return end - position


def create_backend(spec: str | os.PathLike[str], **kwargs: Any) -> ObjectBackend:
    """Create a backend from a URI-style spec.

    Supported specs:

    - ``file:///path`` or a bare filesystem path → :class:`LocalFilesystemBackend`
    - ``s3://bucket`` or ``s3://bucket/prefix`` → :class:`Boto3S3Backend`
      (falls back to :class:`MinioBackend` when ``boto3`` is unavailable but
      ``minio`` plus ``endpoint``/credentials kwargs are provided)
    - ``minio://bucket`` → :class:`MinioBackend`

    Extra keyword arguments are forwarded to the backend constructor
    (``client`` injection is honored for tests).
    """
    text = os.fspath(spec)
    parsed = urlparse(text)
    scheme = parsed.scheme.lower()
    if scheme in {"", "file"}:
        path = parsed.path if scheme == "file" else text
        if not path:
            raise ValueError(f"invalid local backend spec: {text!r}")
        return LocalFilesystemBackend(path, **kwargs)
    if scheme == "s3":
        bucket = parsed.netloc or parsed.path.lstrip("/").split("/", 1)[0]
        if not bucket:
            raise ValueError(f"invalid s3 backend spec: {text!r}")
        try:
            return Boto3S3Backend(bucket, **kwargs)
        except ImportError:
            if "endpoint" in kwargs or "client" in kwargs:
                return MinioBackend(bucket, **kwargs)
            raise
    if scheme == "minio":
        bucket = parsed.netloc or parsed.path.lstrip("/").split("/", 1)[0]
        if not bucket:
            raise ValueError(f"invalid minio backend spec: {text!r}")
        return MinioBackend(bucket, **kwargs)
    raise ValueError(f"unsupported backend spec: {text!r}")
