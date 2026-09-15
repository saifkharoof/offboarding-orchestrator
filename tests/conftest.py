"""Shared fixtures.

``db_path`` is a real file rather than ``:memory:`` on purpose: the restart
recovery tests need to close every connection and reopen the database from
disk, which an in-memory database cannot do.
"""

from __future__ import annotations

import pytest

from offboarding.persistence.db import connect, initialize
from offboarding.persistence.repositories import (
    RunRepository,
    SideEffectRepository,
    TraceRepository,
)


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "test.db")


@pytest.fixture
def conn(db_path):
    connection = connect(db_path)
    initialize(connection)
    yield connection
    connection.close()


@pytest.fixture
def runs(conn):
    return RunRepository(conn)


@pytest.fixture
def trace(conn):
    return TraceRepository(conn)


@pytest.fixture
def ledger_store(conn):
    return SideEffectRepository(conn)
