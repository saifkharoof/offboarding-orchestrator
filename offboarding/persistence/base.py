"""Shared plumbing for the three repositories.

``RunRepository``, ``TraceRepository`` and ``SideEffectRepository`` each wrap
one table, but they lean on the same three things: a timestamp format, a way
to turn one back into a ``datetime``, and a transaction that takes the write
lock before it reads. Kept here once rather than three times.
"""

from __future__ import annotations

import sqlite3
from contextlib import AbstractContextManager, contextmanager
from datetime import UTC, datetime
from typing import Iterator


def now() -> str:
    """The current time in the ISO format every table stores timestamps in."""
    return datetime.now(UTC).isoformat()


def parse_ts(value: str | None) -> datetime | None:
    """Parse a stored timestamp back into a ``datetime``, or ``None``."""
    return datetime.fromisoformat(value) if value else None


@contextmanager
def immediate(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """Run a read-modify-write under a write lock taken up front.

    ``BEGIN IMMEDIATE`` acquires the write lock before we read, so two callers
    cannot both read status ``paused`` and then both transition away from it.
    """
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except Exception:
        conn.rollback()
        raise
    else:
        conn.commit()


class BaseRepository:
    """Common constructor and transaction helper for the three repositories."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def _immediate(self) -> AbstractContextManager[sqlite3.Connection]:
        """This repository's connection, under :func:`immediate`.

        A thin instance-level wrapper so subclasses write
        ``with self._immediate():`` against their own connection instead of
        importing the free function and passing ``self._conn`` at each call
        site.
        """
        return immediate(self._conn)
