# Knowledge Plane object storage (MinIO)

S3-compatible object storage backing the content-addressed artifact store
(`backend/packages/harness/deerflow/knowledge/artifacts/`). MinIO is used for
local development; production points the same `create_backend("s3://…")` API
at real S3 with no code changes.

## Files

- `docker-compose.minio.yml` — standalone MinIO service + one-shot bucket
  provisioning (`minio-init`, idempotent).
- `minio.env.example` — environment template (copy to `minio.env`, gitignored
  by convention; never commit real credentials).

## Run

From the repo root:

```bash
cp deploy/knowledge/minio.env.example deploy/knowledge/minio.env
# edit deploy/knowledge/minio.env — at minimum MINIO_ROOT_PASSWORD
docker compose -f deploy/knowledge/docker-compose.minio.yml up -d
docker compose -f deploy/knowledge/docker-compose.minio.yml ps
```

- S3 API: `http://127.0.0.1:9000` (override with `MINIO_PORT`)
- Console: `http://127.0.0.1:9001` (override with `MINIO_CONSOLE_PORT`)
- Default bucket: `quantflow-evidence` (override with `MINIO_BUCKET`)

Ports bind to loopback by default per repo convention; set `BIND_HOST` to
expose them elsewhere deliberately.

## Use from the artifact store

```python
from deerflow.knowledge.artifacts import ArtifactStore, create_backend

backend = create_backend(
    "s3://quantflow-evidence",
    endpoint_url="http://127.0.0.1:9000",
    aws_access_key_id="minioadmin",
    aws_secret_access_key="minioadmin",
    region_name="us-east-1",
)
store = ArtifactStore(backend)
record = store.put_bytes(b"backtest results", kind="result")
print(record.uri)  # artifact://sha256/<digest>
```

Backend selection: `boto3` is preferred when installed, otherwise the `minio`
client. Both are optional — tests and local development use
`LocalFilesystemBackend` with no services running.

## Teardown

```bash
docker compose -f deploy/knowledge/docker-compose.minio.yml down      # keep data
docker compose -f deploy/knowledge/docker-compose.minio.yml down -v  # wipe data
```
