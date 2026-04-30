"""Scheduler de actualizacion automatica post-jornada.

Ejecuta el pipeline completo (ingesta + features + reentrenamiento) cada
dia a las 23:30 hora de Madrid, cuando la mayoria de partidos de LaLiga
ya han terminado.

Uso:
    python scripts/scheduler.py

Variables de entorno requeridas: las mismas que el servicio etl
    (DATABASE_URL, LALIGA_API_KEY, etc.)
"""
import logging
import subprocess
import sys
import time
from datetime import datetime

import schedule
import pytz

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [scheduler] %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("/app/logs/scheduler.log"),
    ],
)
logger = logging.getLogger("scheduler")

TZ = pytz.timezone("Europe/Madrid")


def run_step(cmd: list[str], step_name: str) -> bool:
    """Ejecuta un subproceso y registra resultado. Devuelve True si OK."""
    logger.info("[%s] Iniciando...", step_name)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        logger.info("[%s] Completado.", step_name)
        if result.stdout.strip():
            logger.debug(result.stdout[-2000:])
        return True
    else:
        logger.error("[%s] FALLO (rc=%d):\n%s", step_name, result.returncode, result.stderr[-2000:])
        return False


def run_pipeline():
    """Secuencia completa: ingesta -> features -> reentrenamiento."""
    now = datetime.now(TZ)
    logger.info("=" * 60)
    logger.info("Pipeline iniciado: %s", now.strftime("%Y-%m-%d %H:%M %Z"))

    steps = [
        (["python", "-m", "src.ingestion.etl_v2"],         "Ingesta ETL"),
        (["python", "-m", "src.features.pipeline"],         "Feature engineering"),
        (["python", "-m", "src.training.train"],            "Reentrenamiento modelo"),
    ]

    for cmd, name in steps:
        ok = run_step(cmd, name)
        if not ok:
            logger.error("Pipeline abortado en paso: %s", name)
            return

    logger.info("Pipeline completado correctamente.")
    logger.info("=" * 60)


# Programar ejecucion diaria a las 23:30 (hora Madrid)
schedule.every().day.at("23:30").do(run_pipeline)

logger.info("Scheduler iniciado. Pipeline se ejecutara diariamente a las 23:30 (Europe/Madrid).")
logger.info("Proxima ejecucion: %s", schedule.next_run())

while True:
    schedule.run_pending()
    time.sleep(30)
