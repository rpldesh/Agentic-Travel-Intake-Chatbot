"""Database engine/session management. Agents never touch this directly."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy import event
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from travel_bot.database.models import Base


class Database:
    def __init__(self, url: str, *, echo: bool = False) -> None:
        self.url = url
        self.is_sqlite = url.startswith("sqlite")
        connect_args = {"timeout": 15} if self.is_sqlite else {}
        self.engine: AsyncEngine = create_async_engine(
            url, echo=echo, connect_args=connect_args
        )
        if self.is_sqlite:
            event.listen(self.engine.sync_engine, "connect", _sqlite_pragmas)
        self._sessionmaker = async_sessionmaker(self.engine, expire_on_commit=False)

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        """A unit of work: commits on success, rolls back on error."""
        async with self._sessionmaker() as session:
            try:
                yield session
                await session.commit()
            except BaseException:
                await session.rollback()
                raise

    async def create_all(self) -> None:
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def dispose(self) -> None:
        await self.engine.dispose()


def _sqlite_pragmas(dbapi_conn, _record) -> None:  # pragma: no cover - driver hook
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA busy_timeout=5000")
    cursor.close()
