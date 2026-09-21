"""Configuration for the Quantflow Research Knowledge Plane.

This module mirrors the conventions of
``deerflow.config.skills_config`` (pydantic ``BaseModel`` with documented
``Field`` entries plus resolver methods with environment-variable overrides)
and the singleton accessors of ``deerflow.config.memory_config``
(``get_knowledge_config`` / ``set_knowledge_config`` /
``load_knowledge_config_from_dict``).

It intentionally does **not** import ``deerflow.config`` (whose package
``__init__`` pulls the full application config graph). The Knowledge API must
stay importable standalone so its unit tests run without a Gateway config,
and so the integration step can wire it into ``AppConfig`` later without
creating an import cycle.

Resolution order for every resolvable setting:
    1. Explicit model field value
    2. ``DEER_FLOW_KNOWLEDGE_*`` environment variable
    3. Built-in default

Environment variables:
    DEER_FLOW_KNOWLEDGE_ENABLED       "1/true/yes/on" or "0/false/no/off"
    DEER_FLOW_KNOWLEDGE_DSN            PostgreSQL DSN for canonical state
    DEER_FLOW_KNOWLEDGE_S3_ENDPOINT    S3/MinIO endpoint URL (immutable evidence)
    DEER_FLOW_KNOWLEDGE_S3_BUCKET      Bucket for evidence artifacts
    DEER_FLOW_KNOWLEDGE_S3_REGION      Bucket region
"""

import os

from pydantic import BaseModel, Field

ENV_ENABLED = "DEER_FLOW_KNOWLEDGE_ENABLED"
ENV_DSN = "DEER_FLOW_KNOWLEDGE_DSN"
ENV_S3_ENDPOINT = "DEER_FLOW_KNOWLEDGE_S3_ENDPOINT"
ENV_S3_BUCKET = "DEER_FLOW_KNOWLEDGE_S3_BUCKET"
ENV_S3_REGION = "DEER_FLOW_KNOWLEDGE_S3_REGION"

_TRUE_TOKENS = frozenset({"1", "true", "yes", "on"})
_FALSE_TOKENS = frozenset({"0", "false", "no", "off"})

_DEFAULT_BUCKET = "quantflow-knowledge"
_DEFAULT_REGION = "us-east-1"


def _env_str(name: str) -> str | None:
    """Return a stripped env value, treating missing/blank as unset."""
    value = os.getenv(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def _env_flag(name: str) -> bool | None:
    """Parse a boolean env var; return None when unset.

    Raises:
        ValueError: If the variable is set to an unrecognized token.
    """
    raw = _env_str(name)
    if raw is None:
        return None
    token = raw.lower()
    if token in _TRUE_TOKENS:
        return True
    if token in _FALSE_TOKENS:
        return False
    raise ValueError(f"Environment variable {name}={raw!r} is not a recognized boolean (expected one of {sorted(_TRUE_TOKENS | _FALSE_TOKENS)}).")


class KnowledgeConfig(BaseModel):
    """Configuration for the Research Knowledge Plane (Phase 1 scope).

    Covers the synchronous evidence-commit path (experiments, failures,
    assumptions) and experiment search. Retrieval-plane settings (FTS /
    pgvector / rank fusion) belong to a later phase and are deliberately
    absent here.
    """

    enabled: bool = Field(
        default=True,
        description="Master switch for the Knowledge API. Overridden at read time by DEER_FLOW_KNOWLEDGE_ENABLED (see is_enabled()).",
    )
    database_dsn: str | None = Field(
        default=None,
        description="PostgreSQL DSN for canonical knowledge state. Falls back to DEER_FLOW_KNOWLEDGE_DSN. None means the PG binding is not configured (integration step wires AppConfig.database).",
    )
    object_store_endpoint: str | None = Field(
        default=None,
        description="S3-compatible endpoint URL (MinIO for local dev, real S3 in prod) holding immutable evidence artifacts. Falls back to DEER_FLOW_KNOWLEDGE_S3_ENDPOINT.",
    )
    object_store_bucket: str = Field(
        default=_DEFAULT_BUCKET,
        description="Bucket for evidence artifacts. Falls back to DEER_FLOW_KNOWLEDGE_S3_BUCKET when the field holds the default.",
    )
    object_store_region: str = Field(
        default=_DEFAULT_REGION,
        description="Region of the evidence bucket. Falls back to DEER_FLOW_KNOWLEDGE_S3_REGION when the field holds the default.",
    )
    default_page_size: int = Field(
        default=20,
        ge=1,
        le=200,
        description="Default page size for experiment_search when the caller passes no limit.",
    )
    max_page_size: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Hard ceiling for experiment_search limits; larger requests are rejected, not silently clamped.",
    )
    max_payload_bytes: int = Field(
        default=1_000_000,
        ge=1024,
        description="Maximum canonical-JSON bytes accepted for a single write payload (methodology + parameters + code + environment + datasets). Guards the write path against accidental multi-MB blobs.",
    )

    def is_enabled(self) -> bool:
        """Return the effective enabled flag (env override wins).

        Resolution order:
            1. ``DEER_FLOW_KNOWLEDGE_ENABLED`` when set
            2. The ``enabled`` field
        """
        override = _env_flag(ENV_ENABLED)
        if override is not None:
            return override
        return self.enabled

    def get_database_dsn(self) -> str | None:
        """Return the effective PostgreSQL DSN (explicit field, then env, then None)."""
        if self.database_dsn:
            return self.database_dsn
        return _env_str(ENV_DSN)

    def get_object_store_endpoint(self) -> str | None:
        """Return the effective object-store endpoint (explicit field, then env, then None)."""
        if self.object_store_endpoint:
            return self.object_store_endpoint
        return _env_str(ENV_S3_ENDPOINT)

    def get_object_store_bucket(self) -> str:
        """Return the effective evidence bucket (env override wins over the field default)."""
        env_value = _env_str(ENV_S3_BUCKET)
        if env_value is not None:
            return env_value
        return self.object_store_bucket

    def get_object_store_region(self) -> str:
        """Return the effective evidence-bucket region (env override wins over the field default)."""
        env_value = _env_str(ENV_S3_REGION)
        if env_value is not None:
            return env_value
        return self.object_store_region


_knowledge_config: KnowledgeConfig = KnowledgeConfig()


def get_knowledge_config() -> KnowledgeConfig:
    """Return the current process-wide knowledge configuration singleton."""
    return _knowledge_config


def set_knowledge_config(config: KnowledgeConfig) -> None:
    """Replace the process-wide knowledge configuration singleton (tests / bootstrap)."""
    global _knowledge_config
    _knowledge_config = config


def load_knowledge_config_from_dict(config_dict: dict) -> None:
    """Load knowledge configuration from a plain dict (e.g. the ``knowledge:`` YAML section).

    Unknown keys are ignored by pydantic's default extra policy so forward
    additions to the schema never break older readers; validation errors on
    known keys propagate to the caller.
    """
    global _knowledge_config
    _knowledge_config = KnowledgeConfig(**(config_dict or {}))
