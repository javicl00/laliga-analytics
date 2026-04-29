-- ============================================================
-- Schema LaLiga Analytics
-- ============================================================

-- Raw JSON (append-only, auditoria completa)
CREATE TABLE IF NOT EXISTS raw_snapshots (
    id               BIGSERIAL PRIMARY KEY,
    resource         TEXT        NOT NULL,
    competition_slug TEXT        NOT NULL,
    season_label     TEXT        NOT NULL,
    payload          JSONB       NOT NULL,
    fetched_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_raw_resource ON raw_snapshots (resource, competition_slug, season_label);

-- Equipos
CREATE TABLE IF NOT EXISTS teams (
    team_id     INT  PRIMARY KEY,
    slug        TEXT NOT NULL,
    name        TEXT NOT NULL,
    shortname   TEXT,
    color       TEXT,
    opta_id     TEXT,
    lde_id      INT
);

-- Temporadas
CREATE TABLE IF NOT EXISTS seasons (
    season_id   INT  PRIMARY KEY,
    name        TEXT NOT NULL,
    year        INT  NOT NULL,
    slug        TEXT NOT NULL UNIQUE
);

-- Jornadas
CREATE TABLE IF NOT EXISTS gameweeks (
    gameweek_id INT  PRIMARY KEY,
    season_id   INT  NOT NULL REFERENCES seasons(season_id),
    week        INT  NOT NULL,
    name        TEXT,
    date        DATE
);

-- Partidos (fuente principal de verdad)
CREATE TABLE IF NOT EXISTS matches (
    match_id         INT     PRIMARY KEY,
    season_id        INT     NOT NULL REFERENCES seasons(season_id),
    gameweek_id      INT     REFERENCES gameweeks(gameweek_id),
    gameweek_week    INT,
    kickoff_at       TIMESTAMPTZ,
    home_team_id     INT     REFERENCES teams(team_id),
    away_team_id     INT     REFERENCES teams(team_id),
    home_score       SMALLINT,
    away_score       SMALLINT,
    result           TEXT    CHECK (result IN ('home','draw','away')),
    status           TEXT,
    raw_status       TEXT,
    home_formation   TEXT,
    away_formation   TEXT,
    competition_id   INT,
    competition_main BOOLEAN DEFAULT TRUE,
    venue_id         INT,
    venue_name       TEXT,
    opta_id          TEXT,
    lde_id           INT,
    is_brand_day     BOOLEAN DEFAULT FALSE,
    inserted_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_matches_season ON matches (season_id, gameweek_week);
CREATE INDEX IF NOT EXISTS idx_matches_teams  ON matches (home_team_id, away_team_id);

-- Clasificacion (snapshot por extraccion)
CREATE TABLE IF NOT EXISTS standings (
    id              BIGSERIAL PRIMARY KEY,
    season_id       INT  NOT NULL REFERENCES seasons(season_id),
    team_id         INT  NOT NULL REFERENCES teams(team_id),
    position        SMALLINT,
    points          SMALLINT,
    played          SMALLINT,
    won             SMALLINT,
    drawn           SMALLINT,
    lost            SMALLINT,
    goals_for       SMALLINT,
    goals_against   SMALLINT,
    goal_difference SMALLINT,
    qualify_name    TEXT,
    fetched_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Features calculadas (cache para el modelo)
CREATE TABLE IF NOT EXISTS match_features (
    match_id             INT  PRIMARY KEY REFERENCES matches(match_id),
    home_form_pts        NUMERIC(5,2),  -- pts ultimas 5 jornadas como local
    away_form_pts        NUMERIC(5,2),
    home_gf_avg          NUMERIC(5,2),  -- goles marcados media ultimas 5
    home_gc_avg          NUMERIC(5,2),
    away_gf_avg          NUMERIC(5,2),
    away_gc_avg          NUMERIC(5,2),
    home_position        SMALLINT,      -- posicion en tabla jornada previa
    away_position        SMALLINT,
    h2h_home_wins        SMALLINT,      -- H2H ultimas 5 temporadas
    h2h_draws            SMALLINT,
    h2h_away_wins        SMALLINT,
    computed_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Predicciones
CREATE TABLE IF NOT EXISTS predictions (
    id           BIGSERIAL PRIMARY KEY,
    match_id     INT  NOT NULL REFERENCES matches(match_id),
    model_name   TEXT NOT NULL,
    model_version TEXT,
    prob_home    NUMERIC(6,4),
    prob_draw    NUMERIC(6,4),
    prob_away    NUMERIC(6,4),
    predicted_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_predictions_match ON predictions (match_id, model_name);
