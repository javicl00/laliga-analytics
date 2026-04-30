"""Aplica migraciones SQL incrementales usando SQLAlchemy (no requiere psql).

Lee los ficheros sql/00*.sql en orden lexicografico y ejecuta cada
statement separado por ';'. Es idempotente gracias a las clausulas
IF NOT EXISTS / IF EXISTS en los DDL.

Uso:
    from src.storage.migrations import apply_migrations
    apply_migrations(engine)
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from sqlalchemy.engine import Engine
from sqlalchemy import text

logger = logging.getLogger(__name__)

# Directorio raiz del proyecto (dos niveles arriba de este fichero)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SQL_DIR = _PROJECT_ROOT / "sql"


def apply_migrations(engine: Engine, sql_dir: Path | None = None) -> None:
    """Ejecuta todos los ficheros *.sql de sql_dir en orden lexicografico.

    Cada fichero se divide por ';' y cada statement no vacio se ejecuta
    en una transaccion independiente para que un fallo en uno no bloquee
    los demas (util si algunos ALTER ya se aplicaron en ejecuciones previas).
    """
    sql_dir = sql_dir or _SQL_DIR
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
            # Ignorar lineas de comentario puras
            if stmt.startswith("--"):
                continue
            try:
                with engine.begin() as conn:
                    conn.execute(text(stmt))
                applied += 1
            except Exception as exc:
                # Loguear pero continuar: puede ser un DDL ya aplicado
                logger.debug("Statement omitido (%s): %.80s", type(exc).__name__, stmt)
        logger.info("  %d statements ejecutados en %s", applied, sql_file.name)
