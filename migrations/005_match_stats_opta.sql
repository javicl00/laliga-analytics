-- Migracion 005: stats brutas por partido desde API Opta (webview)
-- Ejecutar con: docker compose exec db psql -U laliga -d laliga_analytics -f /migrations/005_match_stats_opta.sql

-- Tabla raw: una fila por partido x equipo (home/away)
CREATE TABLE IF NOT EXISTS match_stats_opta (
    match_id          INTEGER      NOT NULL REFERENCES matches(match_id) ON DELETE CASCADE,
    is_home           BOOLEAN      NOT NULL,          -- TRUE = equipo local, FALSE = visitante
    opta_team_id      TEXT,                           -- 't954', 't184', ...
    possession_pct    NUMERIC(5,2),                   -- % posesion
    ppda              NUMERIC(6,2),                   -- Passes per Defensive Action (menor = mas presion)
    shots_on_target   SMALLINT,                       -- ontargetscoringatt
    total_shots       SMALLINT,                       -- totalscoringatt
    big_chances_created SMALLINT,                     -- bigchancecreated
    big_chances_missed  SMALLINT,                     -- bigchancemissed
    accurate_pass     SMALLINT,
    total_pass        SMALLINT,
    aerial_won        SMALLINT,
    aerial_lost       SMALLINT,
    fetched_at        TIMESTAMPTZ  NOT NULL DEFAULT now(),
    PRIMARY KEY (match_id, is_home)
);

CREATE INDEX IF NOT EXISTS idx_mso_match ON match_stats_opta(match_id);

-- Familia G en match_features: rolling avg ultimos 5 partidos (pre-partido)
ALTER TABLE match_features
    ADD COLUMN IF NOT EXISTS home_possession_last5    NUMERIC(5,2),
    ADD COLUMN IF NOT EXISTS away_possession_last5    NUMERIC(5,2),
    ADD COLUMN IF NOT EXISTS home_ppda_last5          NUMERIC(6,2),
    ADD COLUMN IF NOT EXISTS away_ppda_last5          NUMERIC(6,2),
    ADD COLUMN IF NOT EXISTS home_shots_ot_last5      NUMERIC(5,2),
    ADD COLUMN IF NOT EXISTS away_shots_ot_last5      NUMERIC(5,2),
    ADD COLUMN IF NOT EXISTS home_bigchances_last5    NUMERIC(5,2),
    ADD COLUMN IF NOT EXISTS away_bigchances_last5    NUMERIC(5,2);

COMMENT ON TABLE match_stats_opta IS
    'Stats brutas por partido ingestadas desde GET /webview/api/web/matches/opta/{id}/stats';
