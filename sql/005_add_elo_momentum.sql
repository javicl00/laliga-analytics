-- Familia D: ELO momentum (variacion ELO en los ultimos N partidos)
-- Negativo => equipo perdiendo nivel; positivo => en racha ascendente
ALTER TABLE match_features
    ADD COLUMN IF NOT EXISTS home_elo_momentum REAL;

ALTER TABLE match_features
    ADD COLUMN IF NOT EXISTS away_elo_momentum REAL;
