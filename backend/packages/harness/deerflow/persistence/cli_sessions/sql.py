from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import async_sessionmaker

from deerflow.persistence.cli_sessions.model import CliAuditRow, CliPresetRow, CliSessionRow

_TERMINAL_STATUSES = frozenset({"completed", "failed", "stopped", "interrupted"})

_NULLABLE_WRITES = frozenset({"session_ref", "resumed_from", "error", "pid", "tmux_session", "ended_at", "model", "provider", "preset", "effort", "approval_mode", "sandbox_network", "workspace", "contributor_mode"})


def _now() -> datetime:
    return datetime.now(UTC)


def _row_to_dict(row: CliSessionRow) -> dict[str, Any]:
    return row.to_dict()


class CliSessionRepository:
    """Async CRUD over :class:`CliSessionRow`, always scoped to one owner."""

    def __init__(self, session_factory: async_sessionmaker) -> None:
        self._sf = session_factory

    async def create(
        self,
        *,
        created_by: str,
        status: str = "running",
        mode: str = "coding",
        session_ref: str | None = None,
        resumed_from: str | None = None,
        model: str | None = None,
        provider: str | None = None,
        preset: str | None = None,
        effort: str | None = None,
        contributor_mode: bool | None = None,
        approval_mode: str | None = None,
        sandbox_network: str | None = None,
        trust_workspace: bool = False,
        disable_approval: bool = False,
        disable_sandbox: bool = False,
        yolo: bool = False,
        workspace: str | None = None,
        prompt: str | None = None,
        pid: int | None = None,
        tmux_session: str | None = None,
        log_path: str | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        now = _now()
        row = CliSessionRow(
            id=session_id or uuid.uuid4().hex,
            session_ref=session_ref,
            resumed_from=resumed_from,
            status=status,
            mode=mode,
            model=model,
            provider=provider,
            preset=preset,
            effort=effort,
            contributor_mode=contributor_mode,
            approval_mode=approval_mode,
            sandbox_network=sandbox_network,
            trust_workspace=trust_workspace,
            disable_approval=disable_approval,
            disable_sandbox=disable_sandbox,
            yolo=yolo,
            workspace=workspace,
            prompt=prompt,
            pid=pid,
            tmux_session=tmux_session,
            log_path=log_path,
            started_at=now,
            created_by=created_by,
            created_at=now,
            updated_at=now,
        )
        async with self._sf() as session:
            session.add(row)
            await session.commit()
            await session.refresh(row)
            return _row_to_dict(row)

    async def get(self, session_id: str, *, user_id: str) -> dict[str, Any] | None:
        async with self._sf() as session:
            row = await session.get(CliSessionRow, session_id)
            if row is None or row.created_by != user_id:
                return None
            return _row_to_dict(row)

    async def list_by_owner(
        self,
        *,
        user_id: str,
        limit: int = 50,
        status: str | None = None,
        mode: str | None = None,
    ) -> list[dict[str, Any]]:
        stmt = select(CliSessionRow).where(CliSessionRow.created_by == user_id)
        if status is not None:
            stmt = stmt.where(CliSessionRow.status == status)
        if mode is not None:
            stmt = stmt.where(CliSessionRow.mode == mode)
        stmt = stmt.order_by(CliSessionRow.created_at.desc()).limit(limit)
        async with self._sf() as session:
            result = await session.execute(stmt)
            return [_row_to_dict(row) for row in result.scalars().all()]

    async def update(
        self,
        session_id: str,
        *,
        user_id: str,
        **fields: Any,
    ) -> dict[str, Any] | None:
        fields = {key: value for key, value in fields.items() if value is not None or key in _NULLABLE_WRITES}
        fields["updated_at"] = _now()
        async with self._sf() as session:
            row = await session.get(CliSessionRow, session_id)
            if row is None or row.created_by != user_id:
                return None
            for key, value in fields.items():
                setattr(row, key, value)
            await session.commit()
            await session.refresh(row)
            return _row_to_dict(row)

    async def list_running(self) -> list[dict[str, Any]]:
        """Return every non-terminal row, across owners (boot reconciliation only)."""
        stmt = select(CliSessionRow).where(CliSessionRow.status.notin_(_TERMINAL_STATUSES)).order_by(CliSessionRow.created_at.asc())
        async with self._sf() as session:
            result = await session.execute(stmt)
            return [_row_to_dict(row) for row in result.scalars().all()]

    async def mark_finished(self, session_id: str, *, user_id: str, exit_code: int) -> dict[str, Any] | None:
        """Complete a still-running row from a sentinel exit code; terminal rows are never rewritten."""
        status = "completed" if exit_code == 0 else "failed"
        async with self._sf() as session:
            result = await session.execute(
                update(CliSessionRow).where(CliSessionRow.id == session_id).where(CliSessionRow.created_by == user_id).where(CliSessionRow.status == "running").values(status=status, ended_at=_now(), updated_at=_now())
            )
            await session.commit()
            if result.rowcount != 1:
                return None
            row = await session.get(CliSessionRow, session_id)
            return None if row is None else _row_to_dict(row)

    async def mark_interrupted(self, session_id: str) -> bool:
        """Mark a still-running row interrupted; terminal rows are never rewritten."""
        async with self._sf() as session:
            result = await session.execute(update(CliSessionRow).where(CliSessionRow.id == session_id).where(CliSessionRow.status.notin_(_TERMINAL_STATUSES)).values(status="interrupted", ended_at=_now(), updated_at=_now()))
            await session.commit()
            return result.rowcount == 1


class CliPresetRepository:
    """Async CRUD over :class:`CliPresetRow` (named launch defaults)."""

    def __init__(self, session_factory: async_sessionmaker) -> None:
        self._sf = session_factory

    async def get(self, name: str) -> dict[str, Any] | None:
        async with self._sf() as session:
            row = await session.get(CliPresetRow, name)
            return None if row is None else row.to_dict()

    async def list(self) -> list[dict[str, Any]]:
        async with self._sf() as session:
            result = await session.execute(select(CliPresetRow).order_by(CliPresetRow.name.asc()))
            return [row.to_dict() for row in result.scalars().all()]

    async def upsert(
        self,
        name: str,
        *,
        updated_by: str,
        model: str | None = None,
        provider: str | None = None,
        preset: str | None = None,
        effort: str | None = None,
        contributor_mode: bool | None = None,
        yolo: bool = False,
        approval_mode: str | None = None,
    ) -> dict[str, Any]:
        async with self._sf() as session:
            row = await session.get(CliPresetRow, name)
            if row is None:
                row = CliPresetRow(name=name)
                session.add(row)
            row.model = model
            row.provider = provider
            row.preset = preset
            row.effort = effort
            row.contributor_mode = contributor_mode
            row.yolo = yolo
            row.approval_mode = approval_mode
            row.updated_by = updated_by
            row.updated_at = _now()
            await session.commit()
            await session.refresh(row)
            return row.to_dict()


class CliAuditRepository:
    """Append-only audit log over :class:`CliAuditRow`, read back per actor."""

    def __init__(self, session_factory: async_sessionmaker) -> None:
        self._sf = session_factory

    async def record(
        self,
        *,
        actor: str,
        action: str,
        result: str,
        session_id: str | None = None,
        command: str | None = None,
        preset: str | None = None,
        preset_version: str | None = None,
    ) -> dict[str, Any]:
        row = CliAuditRow(
            actor=actor,
            action=action,
            session_id=session_id,
            command=command,
            preset=preset,
            preset_version=preset_version,
            result=result,
            created_at=_now(),
        )
        async with self._sf() as session:
            session.add(row)
            await session.commit()
            await session.refresh(row)
            return row.to_dict()

    async def list_for_session(
        self,
        session_id: str,
        *,
        actor: str,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Return the actor's own audit rows for one CLI session, oldest first."""
        stmt = select(CliAuditRow).where(CliAuditRow.session_id == session_id).where(CliAuditRow.actor == actor).order_by(CliAuditRow.id.asc()).limit(limit)
        async with self._sf() as session:
            result = await session.execute(stmt)
            return [row.to_dict() for row in result.scalars().all()]

    async def list_by_actor(
        self,
        *,
        actor: str,
        limit: int = 100,
        action: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return the actor's own audit rows, newest first."""
        stmt = select(CliAuditRow).where(CliAuditRow.actor == actor)
        if action is not None:
            stmt = stmt.where(CliAuditRow.action == action)
        stmt = stmt.order_by(CliAuditRow.id.desc()).limit(limit)
        async with self._sf() as session:
            result = await session.execute(stmt)
            return [row.to_dict() for row in result.scalars().all()]
