-- LaLiga Analytics: DDL inicial para PostgreSQL
-- Tablas maestras, partidos, eventos, standings, stats y marts ML

create table if not exists competitions (
  competition_id serial primary key,
  slug text unique not null,
  name text not null
);

create table if not exists seasons (
  season_id serial primary key,
  competition_id int not null references competitions(competition_id),
  start_year int not null,
  end_year int not null,
  season_label text not null,
  subscription_slug text not null,
  subscription_id int,
  unique (competition_id, start_year)
);

create table if not exists teams (
  team_id int primary key,
  opta_id text,
  lde_id text,
  slug text,
  name text not null,
  shortname text,
  metadata_json jsonb
);

create table if not exists players (
  player_id int primary key,
  opta_id text,
  slug text,
  name text not null,
  position_id int,
  country_id text,
  team_id int references teams(team_id),
  metadata_json jsonb
);

create table if not exists gameweeks (
  gameweek_id int primary key,
  season_id int not null references seasons(season_id),
  week int not null,
  name text,
  shortname text,
  date_start date,
  date_end date,
  unique (season_id, week)
);

create table if not exists matches (
  match_id bigint primary key,
  season_id int not null references seasons(season_id),
  gameweek_id int references gameweeks(gameweek_id),
  kickoff_at timestamptz,
  home_team_id int not null references teams(team_id),
  away_team_id int not null references teams(team_id),
  home_goals int,
  away_goals int,
  status text,
  venue_json jsonb,
  raw_ref text
);

create table if not exists match_events (
  event_id bigserial primary key,
  match_id bigint not null references matches(match_id),
  minute int,
  second int,
  period int,
  team_id int references teams(team_id),
  player_id int references players(player_id),
  event_type text,
  outcome text,
  x numeric,
  y numeric,
  qualifier_json jsonb,
  raw_ref text
);

create table if not exists standings_snapshots (
  season_id int not null references seasons(season_id),
  gameweek_id int not null references gameweeks(gameweek_id),
  team_id int not null references teams(team_id),
  snapshot_ts timestamptz not null,
  points numeric,
  position numeric,
  won numeric,
  drawn numeric,
  lost numeric,
  goals_for numeric,
  goals_against numeric,
  extra_stats_json jsonb,
  primary key (season_id, gameweek_id, team_id, snapshot_ts)
);

create table if not exists team_stats_snapshots (
  season_id int not null references seasons(season_id),
  gameweek_id int references gameweeks(gameweek_id),
  team_id int not null references teams(team_id),
  snapshot_ts timestamptz not null,
  stat_name text not null,
  stat_value numeric,
  scope text default 'season_to_date',
  primary key (season_id, coalesce(gameweek_id, -1), team_id, snapshot_ts, stat_name)
);

create table if not exists player_stats_snapshots (
  season_id int not null references seasons(season_id),
  gameweek_id int references gameweeks(gameweek_id),
  player_id int not null references players(player_id),
  team_id int references teams(team_id),
  snapshot_ts timestamptz not null,
  stat_name text not null,
  stat_value numeric,
  primary key (season_id, coalesce(gameweek_id, -1), player_id, snapshot_ts, stat_name)
);

create table if not exists raw_payloads (
  raw_id bigserial primary key,
  competition_slug text not null,
  season_label text not null,
  resource text not null,
  request_url text not null default '',
  fetched_at timestamptz not null default now(),
  payload jsonb not null,
  payload_hash text not null
);

create table if not exists features_match_pre (
  match_id bigint not null references matches(match_id),
  feature_version text not null,
  feature_name text not null,
  feature_value numeric,
  primary key (match_id, feature_version, feature_name)
);

create table if not exists targets_match (
  match_id bigint primary key references matches(match_id),
  y_1x2 text not null,
  y_home_goals int not null,
  y_away_goals int not null
);

create table if not exists season_team_targets (
  season_id int not null references seasons(season_id),
  team_id int not null references teams(team_id),
  final_points int,
  final_position int,
  relegated boolean,
  european_slot boolean,
  primary key (season_id, team_id)
);

-- Índices de rendimiento
create index if not exists idx_matches_season_gw on matches(season_id, gameweek_id);
create index if not exists idx_matches_kickoff on matches(kickoff_at);
create index if not exists idx_standings_season_gw on standings_snapshots(season_id, gameweek_id);
create index if not exists idx_team_stats_season_gw on team_stats_snapshots(season_id, gameweek_id, team_id);
create index if not exists idx_player_stats_season_gw on player_stats_snapshots(season_id, gameweek_id, player_id);
create index if not exists idx_raw_payloads_resource on raw_payloads(resource, season_label, payload_hash);
create index if not exists idx_features_match_version on features_match_pre(match_id, feature_version);
