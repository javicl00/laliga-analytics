-- Migration 003: reemplaza match_features con el schema rico v2
-- Safe to re-run: DROP + RECREATE (la tabla se repuebla íntegramente por el pipeline)

DROP TABLE IF EXISTS match_features;

CREATE TABLE match_features (
    match_id                 INTEGER      PRIMARY KEY
                                          REFERENCES matches(match_id),

    -- Familia D: Ratings ELO dinámicos (calculados antes del kickoff)
    home_elo                 NUMERIC(8,2),
    away_elo                 NUMERIC(8,2),
    elo_diff                 NUMERIC(8,2),

    -- Familia A: Estado competitivo (standings snapshot jornada anterior)
    home_points_total        NUMERIC(5,2),
    away_points_total        NUMERIC(5,2),
    home_table_position      SMALLINT,
    away_table_position      SMALLINT,
    position_diff            NUMERIC(5,2),
    home_gd_total            NUMERIC(6,2),
    away_gd_total            NUMERIC(6,2),

    -- Familia B: Forma reciente (últimos 5 partidos)
    home_goals_for_last5     NUMERIC(5,2),
    home_goals_against_last5 NUMERIC(5,2),
    away_goals_for_last5     NUMERIC(5,2),
    away_goals_against_last5 NUMERIC(5,2),

    -- Familia E: Contexto
    gameweek                 SMALLINT,
    home_rest_days           NUMERIC(5,1),
    away_rest_days           NUMERIC(5,1),
    home_pressure_index      NUMERIC(6,4),
    away_pressure_index      NUMERIC(6,4),

    computed_at              TIMESTAMPTZ  NOT NULL DEFAULT now()
);

COMMENT ON TABLE match_features IS
    'Features prepartido v2: ELO dinamico, estado competitivo, forma reciente, descanso y presion. '
    'Todas calculadas estrictamente antes del kickoff (anti-leakage). '
    'Fuente: src/features/build_features.py FeatureBuilder';
