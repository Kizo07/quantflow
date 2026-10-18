"""SQL repository for the persistent-bot singleton row."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import text, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from deerflow.persistence.persistent_bot.model import SINGLETON_ID, PersistentBotRow
from deerflow.utils.time import coerce_iso

logger = logging.getLogger(__name__)


class PersistentBotRepository:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._sf = session_factory

    @staticmethod
    def _row_to_dict(row: PersistentBotRow) -> dict[str, Any]:
        data = row.to_dict()
        for key in ("created_at", "updated_at"):
            if data.get(key) is not None:
                data[key] = coerce_iso(data[key])
        return data

    async def get(self) -> dict[str, Any] | None:
        async with self._sf() as session:
            row = await session.get(PersistentBotRow, SINGLETON_ID)
            return self._row_to_dict(row) if row is not None else None

    async def ensure(
        self,
        *,
        user_id: str,
        agent_name: str,
        thread_id: str,
    ) -> tuple[dict[str, Any], bool]:
        """Return the singleton row, creating it from these values if absent.

        Returns ``(row, created)``. Concurrent callers converge on one row:
        SQLite takes the writer up front (``BEGIN IMMEDIATE``), and on
        Postgres the losing insert hits the primary-key constraint and
        re-reads the winner. An existing row is returned unchanged — this
        method never overwrites the recorded thread.
        """
        now = datetime.now(UTC)
        async with self._sf() as session:
            if session.get_bind().dialect.name == "sqlite":
                await session.execute(text("BEGIN IMMEDIATE"))
            row = await session.get(PersistentBotRow, SINGLETON_ID)
            if row is not None:
                await session.commit()
                return self._row_to_dict(row), False
            row = PersistentBotRow(
                id=SINGLETON_ID,
                user_id=user_id,
                agent_name=agent_name,
                thread_id=thread_id,
                heartbeat_task_id=None,
                user_paused=False,
                created_at=now,
                updated_at=now,
            )
            session.add(row)
            try:
                await session.commit()
            except IntegrityError:
                await session.rollback()
                existing = await self.get()
                if existing is None:
                    raise
                return existing, False
            await session.refresh(row)
            return self._row_to_dict(row), True

    async def _mutate(self, **values: Any) -> dict[str, Any] | None:
        async with self._sf() as session:
            if session.get_bind().dialect.name == "sqlite":
                await session.execute(update(PersistentBotRow).where(PersistentBotRow.id == SINGLETON_ID).values(updated_at=PersistentBotRow.updated_at))
            row = await session.get(PersistentBotRow, SINGLETON_ID, with_for_update=True, populate_existing=True)
            if row is None:
                await session.rollback()
                return None
            for key, value in values.items():
                setattr(row, key, value)
            row.updated_at = datetime.now(UTC)
            await session.commit()
            await session.refresh(row)
            return self._row_to_dict(row)

    async def set_thread(self, thread_id: str) -> dict[str, Any] | None:
        return await self._mutate(thread_id=thread_id)

    async def set_heartbeat_task(self, heartbeat_task_id: str | None) -> dict[str, Any] | None:
        return await self._mutate(heartbeat_task_id=heartbeat_task_id)

    async def set_identity(self, *, user_id: str, agent_name: str) -> dict[str, Any] | None:
        return await self._mutate(user_id=user_id, agent_name=agent_name)

    async def set_user_paused(self, paused: bool) -> dict[str, Any] | None:
        return await self._mutate(user_paused=paused)
