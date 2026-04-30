"""Aplica migraciones SQL incrementales usando SQLAlchemy (no requiere psql).

Dos capas de migraciones:
  1. Ficheros sql/00*.sql en orden lexicografico (migraciones DDL completas).
  2. schema_guard(): guarda de columnas criticas via information_schema.
     Funciona incluso si el fichero sql/ aun no esta en local (antes de git pull).

Es idempotente: IF NOT EXISTS / information_schema evitan duplicados.

Uso:
    from src.storage.migrations import apply_migrations
    apply_migrations(engine)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

from sqlalchemy import text
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SQL_DIR = _PROJECT_ROOT / "sql"

# (tabla, columna, definicion DDL)
_REQUIRED_COLUMNS: List[Tuple[str, str, str]] = [
    # Familia F: Head-to-Head (migration 004)
    ("match_features", "h2h_home_wins", "SMALLINT"),
    ("match_features", "h2h_draws",     "SMALLINT"),
    ("match_features", "h2h_away_wins", "SMALLINT"),
]


def _existing_columns(engine: Engine, table: str) -> set:
    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = :tbl"
        ), {"tbl": table}).fetchall()
    return {r[0] for r in rows}


def _schema_guard(engine: Engine) -> None:
    """Añade columnas criticas si no existen (safety net independiente de sql/)."""
    tables_checked: dict = {}
    for table, column, col_def in _REQUIRED_COLUMNS:
        if table not in tables_checked:
            tables_checked[table] = _existing_columns(engine, table)
        if column not in tables_checked[table]:
            logger.info("schema_guard: ADD COLUMN %s.%s %s", table, column, col_def)
            with engine.begin() as conn:
                conn.execute(text(
                    f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {column} {col_def}"
                ))
            tables_checked[table].add(column)
        else:
            logger.debug("schema_guard: %s.%s ya existe", table, column)


def _apply_sql_files(engine: Engine, sql_dir: Path) -> None:
    sql_files = sorted(sql_dir.glob("*.sql"))
    if not sql_files:
        logger.warning("No se encontraron ficheros .sql en %s", sql_dir)
        return
    for sql_file in sql_files:
        logger.info("Aplicando migracion: %s", sql_file.name)
        ddl = sql_file.read_text(encoding="utf-8")
        statements = [s.strip() for s in ddl.split(";") if s.strip()]
        applied = 0
        for stmt in statements:
            if stmt.lstrip().startswith("--"):
                continue
            try:
                with engine.begin() as conn:
                    conn.execute(text(stmt))
                applied += 1
            except Exception as exc:
                logger.debug("Statement omitido (%s): %.80s", type(exc).__name__, stmt)
        logger.info("  %d statements ejecutados en %s", applied, sql_file.name)


def apply_migrations(engine: Engine, sql_dir: Path | None = None) -> None:
    """Punto de entrada principal. Ejecuta ficheros SQL y luego schema_guard."""
    _apply_sql_files(engine, sql_dir or _SQL_DIR)
    _schema_guard(engine)
