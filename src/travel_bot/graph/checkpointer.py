"""LangGraph checkpointer selection.

Development: `AsyncSqliteSaver` (async graph). Production: swap in
`AsyncPostgresSaver` from `langgraph-checkpoint-postgres`. Application code
never reads or writes checkpoint tables directly.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.types import Checkpointer


def strict_serializer() -> JsonPlusSerializer:
    # `allowed_msgpack_modules=None` = strict mode: only LangGraph/LangChain's
    # built-in safe types are revived (same as LANGGRAPH_STRICT_MSGPACK=true).
    return JsonPlusSerializer(allowed_msgpack_modules=None)


@asynccontextmanager
async def open_checkpointer(backend: str, sqlite_path: str) -> AsyncIterator[Checkpointer]:
    if backend == "memory":
        yield InMemorySaver(serde=strict_serializer())
        return
    if backend == "sqlite":
        import aiosqlite
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        conn = await aiosqlite.connect(sqlite_path)
        try:
            saver = AsyncSqliteSaver(conn, serde=strict_serializer())
            await saver.setup()
            yield saver
        finally:
            await conn.close()
        return
    if backend == "postgres":  # pragma: no cover - production path, not wired for dev
        raise NotImplementedError(
            "Install langgraph-checkpoint-postgres and use AsyncPostgresSaver "
            "(see README: production persistence)."
        )
    raise ValueError(f"unknown checkpoint backend: {backend}")
