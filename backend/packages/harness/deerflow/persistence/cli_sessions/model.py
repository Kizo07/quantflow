from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import Boolean, DateTime, Index, Integer, String, Text, false
from sqlalchemy.orm import Mapped, mapped_column

from deerflow.persistence.base import Base


class CliSessionRow(Base):
    """One headless CLI run tracked by the gateway."""

    __tablename__ = "cli_sessions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    session_ref: Mapped[str | None] = mapped_column(String(128), nullable=True)
    resumed_from: Mapped[str | None] = mapped_column(String(64), nullable=True)
    status: Mapped[str] = mapped_column(String(32), index=True)
    mode: Mapped[str] = mapped_column(String(16), default="coding")
    model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    provider: Mapped[str | None] = mapped_column(String(16), nullable=True)
    preset: Mapped[str | None] = mapped_column(String(64), nullable=True)
    effort: Mapped[str | None] = mapped_column(String(16), nullable=True)
    contributor_mode: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    approval_mode: Mapped[str | None] = mapped_column(String(16), nullable=True)
    sandbox_network: Mapped[str | None] = mapped_column(String(32), nullable=True)
    trust_workspace: Mapped[bool] = mapped_column(Boolean, default=False, server_default=false())
    disable_approval: Mapped[bool] = mapped_column(Boolean, default=False, server_default=false())
    disable_sandbox: Mapped[bool] = mapped_column(Boolean, default=False, server_default=false())
    yolo: Mapped[bool] = mapped_column(Boolean, default=False, server_default=false())
    workspace: Mapped[str | None] = mapped_column(String(512), nullable=True)
    prompt: Mapped[str | None] = mapped_column(Text, nullable=True)
    pid: Mapped[int | None] = mapped_column(Integer, nullable=True)
    tmux_session: Mapped[str | None] = mapped_column(String(64), nullable=True)
    log_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_by: Mapped[str] = mapped_column(String(64), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )

    __table_args__ = (Index("ix_cli_sessions_owner_created", "created_by", "created_at"),)


class CliPresetRow(Base):
    """Named launch defaults for the CLI harness.

    Only the Muse 1.3 preset row may persist contributor mode, an
    explicit reasoning effort, or yolo; the service layer rejects those
    fields for every other model (fail-closed).
    """

    __tablename__ = "cli_presets"

    name: Mapped[str] = mapped_column(String(64), primary_key=True)
    model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    provider: Mapped[str | None] = mapped_column(String(16), nullable=True)
    preset: Mapped[str | None] = mapped_column(String(64), nullable=True)
    effort: Mapped[str | None] = mapped_column(String(16), nullable=True)
    contributor_mode: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    yolo: Mapped[bool] = mapped_column(Boolean, default=False, server_default=false())
    approval_mode: Mapped[str | None] = mapped_column(String(16), nullable=True)
    updated_by: Mapped[str | None] = mapped_column(String(64), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )


class CliAuditRow(Base):
    """One audited control action on the CLI harness.

    Written on every launch and control action (launch, resume,
    export, inspect, stop, preset change): who acted, what ran
    (command string with secrets redacted), which preset was used,
    and the outcome.
    """

    __tablename__ = "cli_audit"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    actor: Mapped[str] = mapped_column(String(64), index=True)
    action: Mapped[str] = mapped_column(String(32), index=True)
    session_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    command: Mapped[str | None] = mapped_column(Text, nullable=True)
    preset: Mapped[str | None] = mapped_column(String(64), nullable=True)
    preset_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    result: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
