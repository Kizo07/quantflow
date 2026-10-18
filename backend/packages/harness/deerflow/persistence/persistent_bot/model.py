"""ORM model for the persistent-bot singleton.

Exactly one row can exist: the primary key is the integer ``1`` and the
``ck_persistent_bot_singleton`` check constraint rejects any other id, so
uniqueness is enforced by the database rather than by application checks.
``thread_id`` and ``heartbeat_task_id`` are additionally unique so no
second bot record could ever claim the same thread or heartbeat task.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import Boolean, CheckConstraint, DateTime, Integer, String, false
from sqlalchemy.orm import Mapped, mapped_column

from deerflow.persistence.base import Base

SINGLETON_ID = 1


class PersistentBotRow(Base):
    __tablename__ = "persistent_bot"
    __table_args__ = (CheckConstraint("id = 1", name="ck_persistent_bot_singleton"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True)
    agent_name: Mapped[str] = mapped_column(String(128))
    thread_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    heartbeat_task_id: Mapped[str | None] = mapped_column(String(64), unique=True, nullable=True)
    # Explicit user off-switch. When true, startup seed/repair is a no-op:
    # the thread and heartbeat task are never recreated or resumed.
    user_paused: Mapped[bool] = mapped_column(Boolean, default=False, server_default=false())
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )
