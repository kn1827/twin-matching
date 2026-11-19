"""SQLite access (stage1.md, section 10.2), kept from B1.

One file `data/gnwt.db` in WAL mode, safe for several writer processes. Within a process, `Repo` shares
one locked connection. Every row written through `insert` gets `config_version` and `created_at`.
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gnwt.config import PROJECT_ROOT

SCHEMA_SQL = Path(__file__).with_name("schema.sql")
DEFAULT_DB = PROJECT_ROOT / "data" / "gnwt.db"

# Creation order in schema.sql (section 10.2).
TABLES: tuple[str, ...] = (
    "personas", "pairs", "context_texts", "context_demands", "demand_labels", "static_scores",
    "unmatched_contexts", "artifacts",
)


def utc_now() -> str:
    """Current time, ISO 8601 UTC, millisecond precision."""
    return datetime.now(UTC).isoformat(timespec="milliseconds")


class Repo:
    """Database connection of one process.

    `config_version` is stamped on every row. Write several rows atomically with `with repo.transaction():`.
    """

    def __init__(self, path: str | Path = DEFAULT_DB, *, config_version: str, create: bool = True) -> None:
        self.path = Path(path)
        self.config_version = config_version
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # isolation_level=None: transactions are managed explicitly with BEGIN IMMEDIATE / COMMIT.
        self._conn = sqlite3.connect(str(path), timeout=30.0, check_same_thread=False,
                                     isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._tx_depth = 0
        self._columns: dict[str, tuple[str, ...]] = {}
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA busy_timeout=30000")
        if create:
            self.init_schema()

    # ------------------------------------------------------------------------------------- schema

    def init_schema(self) -> None:
        """Create every table of section 10.2 if missing (idempotent)."""
        sql = SCHEMA_SQL.read_text(encoding="utf-8")
        with self._lock:
            self._conn.executescript(sql)
            self._columns.clear()

    def tables(self) -> list[str]:
        rows = self.query("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        return [r["name"] for r in rows]

    def columns(self, table: str) -> tuple[str, ...]:
        self._check_table(table)
        if table not in self._columns:
            rows = self.query(f"PRAGMA table_info({table})")
            self._columns[table] = tuple(r["name"] for r in rows)
        return self._columns[table]

    def journal_mode(self) -> str:
        return str(self.query("PRAGMA journal_mode")[0][0]).lower()

    # -------------------------------------------------------------------------------- transactions

    @contextmanager
    def transaction(self) -> Iterator[Repo]:
        """Write transaction (BEGIN IMMEDIATE). When nested, only the outermost one commits."""
        with self._lock:
            outer = self._tx_depth == 0
            if outer:
                self._conn.execute("BEGIN IMMEDIATE")
            self._tx_depth += 1
            try:
                yield self
            except BaseException:
                self._tx_depth -= 1
                if outer:
                    self._conn.execute("ROLLBACK")
                raise
            else:
                self._tx_depth -= 1
                if outer:
                    self._conn.execute("COMMIT")

    # ------------------------------------------------------------------------------- read, write

    def insert(self, table: str, row: Mapping[str, Any], *, replace: bool = False) -> None:
        """Insert one row; unknown columns raise. `replace=True` uses INSERT OR REPLACE."""
        self.insert_many(table, [row], replace=replace)

    def insert_many(self, table: str, rows: Sequence[Mapping[str, Any]], *, replace: bool = False) -> None:
        if not rows:
            return
        cols = set(self.columns(table))
        now = utc_now()
        prepared: list[dict[str, Any]] = []
        for row in rows:
            unknown = set(row) - cols
            if unknown:
                raise KeyError(f"table {table} has no column {sorted(unknown)}")
            full = dict(row)
            full.setdefault("config_version", self.config_version)
            full.setdefault("created_at", now)
            prepared.append(full)
        names = sorted({k for r in prepared for k in r})
        verb = "INSERT OR REPLACE" if replace else "INSERT"
        sql = f"{verb} INTO {table} ({', '.join(names)}) VALUES ({', '.join('?' for _ in names)})"
        values = [tuple(r.get(n) for n in names) for r in prepared]
        with self.transaction():
            self._conn.executemany(sql, values)

    def query(self, sql: str, params: Sequence[Any] | Mapping[str, Any] = ()) -> list[sqlite3.Row]:
        with self._lock:
            return list(self._conn.execute(sql, params).fetchall())

    def count(self, table: str) -> int:
        self._check_table(table)
        return int(self.query(f"SELECT COUNT(*) FROM {table}")[0][0])

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> Repo:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @staticmethod
    def _check_table(table: str) -> None:
        if table not in TABLES:
            raise KeyError(f"table not in schema: {table}")
