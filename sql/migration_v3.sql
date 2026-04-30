-- Migration v3: features avanzadas (arbitro y temperatura)
-- Ejecutar manualmente: psql $DATABASE_URL -f sql/migration_v3.sql
-- O incluir en el flujo de migraciones automatizado.

-- Arbitro del partido
ALTER TABLE matches
    ADD COLUMN IF NOT EXISTS referee VARCHAR(120);

-- Temperatura en grados Celsius en el momento del kickoff
-- Fuente: Open-Meteo historical API (sin clave, gratuita)
ALTER TABLE matches
    ADD COLUMN IF NOT EXISTS temperature_c FLOAT;

-- Indice para queries de arbitro
CREATE INDEX IF NOT EXISTS idx_matches_referee ON matches (referee)
    WHERE referee IS NOT NULL;

-- Vista auxiliar: estadisticas de arbitro por equipo
-- Util como feature: tasa de victoria del equipo local con ese arbitro
CREATE OR REPLACE VIEW referee_stats AS
SELECT
    referee,
    COUNT(*)                                                         AS total_matches,
    ROUND(AVG(home_score + away_score)::numeric, 2)                  AS avg_goals,
    ROUND(SUM(CASE WHEN result = 'home' THEN 1 ELSE 0 END)::numeric
          / NULLIF(COUNT(*), 0), 3)                                  AS home_win_rate,
    ROUND(SUM(CASE WHEN result = 'draw' THEN 1 ELSE 0 END)::numeric
          / NULLIF(COUNT(*), 0), 3)                                  AS draw_rate,
    ROUND(SUM(CASE WHEN result = 'away' THEN 1 ELSE 0 END)::numeric
          / NULLIF(COUNT(*), 0), 3)                                  AS away_win_rate,
    ROUND(AVG(temperature_c)::numeric, 1)                            AS avg_temperature_c
FROM matches
WHERE referee IS NOT NULL
  AND result IS NOT NULL
GROUP BY referee
ORDER BY total_matches DESC;

COMMENT ON COLUMN matches.referee IS
    'Nombre del arbitro principal. Fuente: API LaLiga.';
COMMENT ON COLUMN matches.temperature_c IS
    'Temperatura exterior en grados Celsius al inicio del partido.
     Obtenida de Open-Meteo historical API usando las coordenadas del estadio.';
