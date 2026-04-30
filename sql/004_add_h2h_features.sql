-- Migration 004: add H2H features to match_features
-- Safe to re-run: IF NOT EXISTS

ALTER TABLE match_features
    ADD COLUMN IF NOT EXISTS h2h_home_wins SMALLINT,
    ADD COLUMN IF NOT EXISTS h2h_draws     SMALLINT,
    ADD COLUMN IF NOT EXISTS h2h_away_wins SMALLINT;

COMMENT ON COLUMN match_features.h2h_home_wins IS
    'Victorias del equipo local en los ultimos 10 enfrentamientos directos (perspectiva del partido actual)';
COMMENT ON COLUMN match_features.h2h_draws IS
    'Empates en los ultimos 10 enfrentamientos directos';
COMMENT ON COLUMN match_features.h2h_away_wins IS
    'Victorias del equipo visitante en los ultimos 10 enfrentamientos directos';
