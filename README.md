# LaLiga Analytics

Pipeline completo de extraccion, almacenamiento, feature engineering y prediccion de partidos de **LaLiga EA Sports** a partir de la API publica oficial.

```
┌──────────────┐   GET /matches?subscriptionSlug+week   ┌─────────────┐
│  LaLiga API  │ ────────────────────────────────────► │  ETL v2     │
└──────────────┘                                        └──────┬──────┘
                                                               │ raw JSON
                                                        ┌──────▼──────┐
                                                        │  PostgreSQL  │
                                                        │  raw +       │
                                                        │  normalized  │
                                                        └──────┬──────┘
                                                               │
                                              ┌────────────────┼────────────────┐
                                              │                │                │
                                       ┌──────▼──────┐ ┌──────▼──────┐ ┌──────▼──────┐
                                       │  Features   │ │  Training   │ │  API REST   │
                                       │  pipeline   │ │  LightGBM   │ │  FastAPI    │
                                       └─────────────┘ └─────────────┘ └─────────────┘
```

## Requisitos

- Docker + Docker Compose
- Python 3.11+ (opcional, para desarrollo local)

## Inicio rapido con Docker

```bash
# 1. Clonar el repositorio
git clone https://github.com/javicl00/laliga-analytics.git
cd laliga-analytics

# 2. Configurar variables de entorno
cp .env.example .env
# Editar .env si es necesario (las claves por defecto funcionan)

# 3. Levantar la base de datos (aplica el schema automaticamente)
docker compose up db -d

# 4. Ejecutar el ETL (extrae todas las jornadas disponibles)
docker compose run --rm etl

# 5. Levantar la API
docker compose up api -d
```

La API estara disponible en `http://localhost:8000`. Documentacion interactiva en `http://localhost:8000/docs`.

## Paso a paso completo

### 1. Extraccion de datos

El ETL extrae las 38 jornadas de la temporada configurada en `.env`:

```bash
# Con Docker
docker compose run --rm etl

# Sin Docker (requiere DATABASE_URL en el entorno)
python -m src.ingestion.etl_v2
```

Solo jornadas ya jugadas (ej. J1-J33):

```python
from src.ingestion.etl_v2 import ETLv2, SeasonConfig
from src.clients.laliga_api import LaLigaClient
from src.storage.repository import PostgresRawRepository

client = LaLigaClient()
repo   = PostgresRawRepository()
season = SeasonConfig(
    competition_slug="primera-division",
    season_label="2025",
    subscription_slug="laliga-easports-2025",
    weeks=list(range(1, 34)),
)
ETLv2(client=client, repo=repo, season=season).run()
```

### 2. Feature engineering

```bash
python -m src.features.pipeline
```

Calcula para cada partido terminado:
- **Forma reciente**: puntos medios en las ultimas 5 jornadas (local/visitante separados)
- **Goles medios**: GF y GC en ventana de 5 partidos
- **H2H**: historico de enfrentamientos directos (ultimos 10)

Todas las ventanas son estrictamente anteriores a `kickoff_at` del partido para evitar data leakage.

### 3. Entrenamiento del modelo

```bash
python -m src.training.train
```

Entrena tres modelos en validacion walk-forward temporal:

| Modelo | Metrica | Split |
|---|---|---|
| Baseline (distribucion marginal) | RPS | Val (J26-J30) |
| Logistic Regression | RPS + Log-loss | Val |
| **LightGBM** (mejor) | RPS + Log-loss | Val + Test (J31-J33) |

El modelo ganador se guarda en `models/lgbm_v1.pkl`.

### 4. API REST

```bash
# Con Docker
docker compose up api -d

# Sin Docker
uvicorn src.serving.app:app --host 0.0.0.0 --port 8000 --reload
```

#### Endpoints disponibles

| Metodo | Ruta | Descripcion |
|---|---|---|
| GET | `/health` | Estado del servicio y si el modelo esta cargado |
| GET | `/standings` | Clasificacion actual |
| GET | `/matches/upcoming` | Proximos partidos con probabilidades |
| POST | `/predict` | Prediccion para un partido concreto |

#### Ejemplo de prediccion

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"home_team_id": 4, "away_team_id": 15}'

# Respuesta
{
  "home_team_id": 4,
  "away_team_id": 15,
  "prob_home": 0.4821,
  "prob_draw": 0.2314,
  "prob_away": 0.2865,
  "model": "lgbm_v1"
}
```

## Estructura del proyecto

```
laliga-analytics/
├── config/
│   └── competitions.yml        # Configuracion de temporadas
├── data/                       # Datos locales (gitignored)
├── logs/                       # Logs (gitignored)
├── models/
│   └── lgbm_v1.pkl             # Modelo entrenado (gitignored, se genera)
├── scripts/
│   └── probe_*.py              # Scripts de exploracion de API
├── sql/
│   └── 01_schema.sql           # Schema PostgreSQL
├── src/
│   ├── clients/
│   │   └── laliga_api.py       # Cliente HTTP LaLiga API
│   ├── features/
│   │   ├── builder.py          # Feature engineering sin leakage
│   │   └── pipeline.py         # Pipeline: BD → features → BD
│   ├── ingestion/
│   │   └── etl_v2.py           # ETL principal
│   ├── normalize/
│   │   ├── normalize_matches.py
│   │   └── normalize_standing.py
│   ├── serving/
│   │   └── app.py              # API FastAPI
│   ├── storage/
│   │   └── repository.py       # Repository pattern (PostgreSQL)
│   └── training/
│       └── train.py            # Entrenamiento walk-forward
├── .env.example
├── .gitignore
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

## Mapa de la API LaLiga (verificado)

| Endpoint | Parametros clave | Scores |
|---|---|---|
| `GET /subscriptions/{slug}` | — | — |
| `GET /subscriptions/{slug}/standing` | — | — |
| `GET /subscriptions/{slug}/teams/stats` | `limit`, `offset` | — |
| `GET /subscriptions/{slug}/players/stats` | `limit`, `offset` | — |
| `GET /matches` | `subscriptionSlug`, `week` | ✅ `home_score`, `away_score` |

> **Nota critica**: el parametro correcto para partidos con scores es `subscriptionSlug` (no `subscriptionId`) y `week` (numero de jornada, no `gameweekId`). Usando `subscriptionId+gameweekId` el endpoint responde 200 pero sin campos de puntuacion.

## Roadmap

- [ ] Ingesta de temporadas historicas (2022, 2023, 2024)
- [ ] Features avanzados: descanso entre partidos, arbitro, temperatura
- [ ] Modelo de Poisson / Dixon-Coles para prediccion de goles
- [ ] Dashboard Streamlit / Grafana
- [ ] Actualizacion automatica post-jornada (cron / Airflow)
