"""Parameterized access wrapper for a read-only PostgreSQL service account."""

from __future__ import annotations

import os
import re
from typing import Callable


_DANGEROUS_FUNCTIONS = re.compile(
    r"\b(?:pg_read_file|pg_read_binary_file|pg_ls_dir|pg_sleep|"
    r"pg_terminate_backend|pg_cancel_backend|set_config|dblink|lo_export|"
    r"lo_import|copy)\s*\(",
    re.IGNORECASE,
)
_WRITE_OR_CONTROL = re.compile(
    r"\b(?:insert|update|delete|merge|create|alter|drop|truncate|grant|revoke|"
    r"copy|call|do|execute|prepare|deallocate|vacuum|analyze|refresh|"
    r"set|reset|commit|rollback|begin|savepoint|release|listen|notify|"
    r"lock|discard|cluster|reindex|security)\b",
    re.IGNORECASE,
)
_FOR_UPDATE = re.compile(r"\bfor\s+(?:no\s+key\s+)?(?:update|share)\b", re.IGNORECASE)


class ReadOnlyDatabaseTool:
    """Accept one bounded SELECT, force a read-only transaction, and cap returned rows.

    Use a separate database account that has SELECT-only grants and no superuser,
    file access, extension, or network privileges. Transaction mode is an additional
    guard, not a replacement for least-privilege database credentials.
    """

    def __init__(
        self,
        connection_factory: Callable[[], object],
        *,
        max_rows: int = 500,
        statement_timeout_ms: int = 3000,
    ) -> None:
        if not 1 <= max_rows <= 2_000:
            raise ValueError("max_rows must be between 1 and 2,000")
        if not 100 <= statement_timeout_ms <= 30_000:
            raise ValueError("statement timeout must be between 100 and 30,000 ms")
        self.connection_factory = connection_factory
        self.max_rows = max_rows
        self.statement_timeout_ms = statement_timeout_ms

    @classmethod
    def from_env(cls, **kwargs: int) -> "ReadOnlyDatabaseTool":
        dsn = os.environ.get("FOMO_READONLY_DATABASE_URL")
        if not dsn:
            raise RuntimeError(
                "Set FOMO_READONLY_DATABASE_URL to a dedicated least-privilege "
                "read-only database account"
            )
        try:
            import psycopg
        except ImportError as exc:
            raise RuntimeError("Install psycopg to connect to PostgreSQL") from exc
        return cls(lambda: psycopg.connect(dsn), **kwargs)

    @staticmethod
    def _validate_query(query: str) -> str:
        if not isinstance(query, str) or not query.strip() or len(query) > 8_000:
            raise ValueError("query must contain 1–8,000 characters")
        sql = query.strip()
        # Comments and delimiters are disallowed rather than parsed ambiguously.
        if ";" in sql or "--" in sql or "/*" in sql or "*/" in sql:
            raise ValueError("SQL comments and statement delimiters are not allowed")
        if not re.match(r"(?is)^select\b", sql):
            raise ValueError("only a single SELECT statement is allowed")
        if _WRITE_OR_CONTROL.search(sql) or _DANGEROUS_FUNCTIONS.search(sql):
            raise ValueError("query contains a forbidden write or control operation")
        if re.search(r"\binto\b", sql, re.IGNORECASE) or _FOR_UPDATE.search(sql):
            raise ValueError("SELECT INTO and row-locking clauses are not allowed")
        return sql

    def query(self, sql: str) -> list[dict]:
        statement = self._validate_query(sql)
        connection = self.connection_factory()
        try:
            cursor = connection.cursor()
            cursor.execute("SET TRANSACTION READ ONLY")
            cursor.execute(f"SET LOCAL statement_timeout = {self.statement_timeout_ms}")
            cursor.execute(statement)
            column_names = [column.name for column in cursor.description or ()]
            rows = cursor.fetchmany(self.max_rows + 1)
            if len(rows) > self.max_rows:
                raise ValueError("query result exceeds the configured row limit")
            return [dict(zip(column_names, row)) for row in rows]
        finally:
            try:
                connection.rollback()
            finally:
                connection.close()