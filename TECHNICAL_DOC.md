# LaLiga Analytics — Documentación Técnica

> Versión: 0.2.0 | Modelo: `lgbm_v1` | Feature version: `v2.2.0`

---

## Índice

1. [Arquitectura del sistema](#1-arquitectura-del-sistema)
2. [Pipeline de datos](#2-pipeline-de-datos)
3. [Feature engineering](#3-feature-engineering)
4. [Modelo predictivo](#4-modelo-predictivo)
5. [Simulación Montecarlo](#5-simulación-montecarlo)
6. [API REST](#6-api-rest)
7. [Interfaz de usuario](#7-interfaz-de-usuario)
8. [Consideraciones de calidad y limitaciones](#8-consideraciones-de-calidad-y-limitaciones)

---

## 1. Arquitectura del sistema

El sistema se despliega íntegramente mediante **Docker Compose** con cuatro servicios:

```
┌───────────────────────────────────────────────────────────────────┐
│  db (PostgreSQL 16)                                               │
│  Almacena matches, teams, standings, match_features               │
└───────────────────────────────────────────────────────────────────┘
         ↑                        ↑                     ↑
┌──────────────┐  ┌────────────────┐  ┌────────────────┐
│  etl            │  │  api (FastAPI)  │  │  ui (Streamlit) │
│  Ingesta +      │  │  :8000          │  │  :8501          │
│  features +     │  │  Predicciones   │  │  Dashboard      │
│  entrenamiento  │  │  Simulaciones   │  │  interactivo    │
└──────────────┘  └────────────────┘  └────────────────┘
```

- **db**: PostgreSQL 16. Volumen persistente `postgres_data`. Las tablas se inicializan con los scripts SQL de `sql/`.
- **etl**: Contenedor de ejecución puntual (`restart: no`). Ejecuta la ingesta, computa features y entrena el modelo. Monta `src/`, `data/`, `logs/` y `models/` como volúmenes.
- **api**: FastAPI + Uvicorn con `--reload`. Carga el modelo entrenado (`models/lgbm_v1.pkl`) al arrancar y expone los endpoints de predicción y simulación. Monta `src/` y `models/`.
- **ui**: Streamlit. Se comunica exclusivamente con la api vía HTTP (`API_URL=http://api:8000`). No accede directamente a la BD.

---

## 2. Pipeline de datos

El pipeline se ejecuta como un proceso secuencial en el contenedor **etl**:

```
API LaLiga
    ↓
[1] Ingesta (src/ingestion/etl_v2.py)
    │  Descarga partidos, equipos, clasificaciones
    │  Normaliza y persiste en tablas matches, teams, standings
    ↓
[2] Feature Engineering (src/features/build_features.py)
    │  Computa 24 features prepartido por cada partido con resultado
    │  Guarda en tabla match_features (JOIN con matches via match_id)
    ↓
[3] Entrenamiento (src/training/)
    │  Carga match_features + etiqueta result (home/draw/away)
    │  Split temporal: train (temporadas históricas) / val / test
    │  Entrena LightGBM clasificador multiclase
    │  Persiste modelo en models/lgbm_v1.pkl
    ↓
[4] Serving (src/serving/app.py)
    │  Carga modelo en memoria al startup
    │  Para partidos futuros: recupera features del último enfrentamiento
    │    histórico entre los dos equipos como proxy de contexto
    ↓
Predicción
```

### Esquema de tablas relevantes

| Tabla | Descripción |
|---|---|
| `matches` | Partidos con `match_id`, `home_team_id`, `away_team_id`, `kickoff_at`, `result`, `status`, `gameweek_week`, `competition_main` |
| `teams` | Catálogo de equipos (todas las divisiones históricas) |
| `standings` | Clasificación actual, refrescada en cada ingesta |
| `match_features` | 24 features numéricas por `match_id`, calculadas prepartido |

---

## 3. Feature engineering

Todas las features se calculan con datos **estrictamente anteriores** al `kickoff_at` del partido a predecir, garantizando ausencia de *data leakage*.

Se agrupan en 5 familias (24 features en total):

### Familia A — Estado competitivo (standings prepartido)

Calculado internamente desde el historial de resultados, sin datos externos:

| Feature | Descripción |
|---|---|
| `home/away_points_total` | Puntos acumulados antes del partido |
| `home/away_table_position` | Posición en la tabla calculada |
| `position_diff` | Diferencia de posición (home − away) |
| `home/away_gd_total` | Diferencia de goles acumulada |
| `home/away_pressure_index` | Índice de presión competitiva (zona de descenso/champions) |

### Familia B — Forma reciente

Media de goles en los últimos 5 partidos del equipo (cualquier campo):

| Feature | Descripción |
|---|---|
| `home/away_goals_for_last5` | Media de goles marcados en últimos 5 |
| `home/away_goals_against_last5` | Media de goles recibidos en últimos 5 |

### Familia D — Ratings ELO dinámicos

Sistema ELO clásico con ventaja de campo (`home_advantage = 70` puntos):

```
E(home) = 1 / (1 + 10^((R_away - (R_home + 70)) / 400))
ΔR = K * (resultado_real - resultado_esperado)    K = 32
```

| Feature | Descripción |
|---|---|
| `home_elo` / `away_elo` | Rating ELO antes del partido |
| `elo_diff` | Diferencia de ELO (home − away) |
| `home/away_elo_momentum` | Delta ELO en los últimos 5 partidos jugados |

### Familia E — Contexto

| Feature | Descripción |
|---|---|
| `gameweek` | Número de jornada |
| `home/away_rest_days` | Días desde el último partido jugado |
| `home/away_pressure_index` | Calculado en Familia A |

### Familia F — Head-to-Head

Ultimos 10 enfrentamientos directos entre ambos equipos:

| Feature | Descripción |
|---|---|
| `h2h_home_wins` | Victorias del equipo local en esos 10 partidos |
| `h2h_draws` | Empates |
| `h2h_away_wins` | Victorias del equipo visitante |

---

## 4. Modelo predictivo

### Tipo

**LightGBM Classifier** — clasificación multiclase con 3 salidas:

| Clase | Significado |
|---|---|
| `home` | Victoria del equipo local |
| `draw` | Empate |
| `away` | Victoria del equipo visitante |

### Split temporal

Para evitar leakage temporal, el split se hace por temporada (ID), no aleatoriamente:

```
Train : temporadas históricas (T:[1..329])   ≈ 4180 partidos
Val   : temporada T:351                        ≈ 380 partidos
Test  : temporada T:375 (más reciente)         ≈ 330 partidos
```

### Métrica de evaluación

Se usa **Ranked Probability Score (RPS)** en lugar de accuracy, ya que evalúa la calidad de la distribución de probabilidades completa (no solo la clase predicha):

```
RPS = (1/2) * Σ (CDF_predicha - CDF_real)^2
```

Valores menores son mejores. El modelo obtiene **RPS val = 0.2019**, comparable a los mejores modelos públicos de predicción de fútbol (≈20-21%).

### Inferencia para partidos futuros

Cuando se predice un partido pendiente, el modelo no dispone de features calculadas para ese partido (no hay resultado aún). La API recupera las features del **último partido histórico** entre ambos equipos como vector de contexto proxy:

```sql
SELECT f.*
FROM match_features f
JOIN matches m USING (match_id)
WHERE m.home_team_id = :home AND m.away_team_id = :away
ORDER BY m.kickoff_at DESC LIMIT 1
```

Si no existe historial entre los equipos, se usan `NULL`s (LightGBM maneja NaN nativamente).

> **Limitación importante**: las features ELO, forma y posición en tabla del vector proxy corresponden al estado de la temporada en que se jugó el último enfrentamiento, no al momento actual. Esto introduce sesgo en las predicciones de partidos futuros. La solución correcta es computar un vector de features "online" con el estado actual de cada equipo.

---

## 5. Simulación Montecarlo

El endpoint `POST /simulate/standings` estima la distribución de probabilidad de la posición final de un equipo dado:

### Algoritmo

```
1. Cargar clasificación actual (puntos, diferencia de goles) desde standings
2. Obtener todos los partidos pendientes (result IS NULL, competition_main = TRUE)
3. Para cada partido pendiente: calcular (p_home, p_draw, p_away) con el modelo
4. Repetir N veces (N ≤ 20.000):
   a. Para cada partido pendiente:
      - Sortear resultado: random.choices([home, draw, away], weights=[p_h, p_d, p_a])
      - Actualizar puntos y diferencia de goles de ambos equipos
   b. Ordenar los 20 equipos por (puntos DESC, gd DESC)
   c. Registrar la posición final del equipo objetivo
5. Devolver frecuencia relativa de cada posición como distribución de probabilidad
```

### Campos de respuesta

```json
{
  "team_id": 123,
  "simulations": 5000,
  "pending_matches_count": 50,
  "team_pending_count": 4,
  "season_complete": false,
  "position_distribution": {
    "16": 0.12,
    "17": 0.35,
    "18": 0.38,
    "19": 0.11,
    "20": 0.04
  }
}
```

### Consideraciones

- La simulación usa los resultados de **toda la liga** (50 partidos), no solo los del equipo objetivo. Esto es correcto: la posición de un equipo depende de los resultados de todos los demás.
- El desempate usa únicamente diferencia de goles. En LaLiga real el criterio incluye el enfrentamiento directo, goles marcados, etc. Esto puede introducir pequeñas desviaciones en equipos muy igualados en puntos.
- Si `pending_matches_count = 0`, la temporada está terminada y la distribución es determinista (100% en la posición actual).

---

## 6. API REST

Base URL: `http://localhost:8000`

| Método | Endpoint | Descripción |
|---|---|---|
| GET | `/health` | Estado del servicio y modelo |
| GET | `/teams` | Catálogo completo de equipos (todas las divisiones) |
| GET | `/standings` | Clasificación actual de Primera División |
| GET | `/matches/upcoming?limit=N` | Próximos N partidos pendientes |
| GET | `/matches/by-jornada?jornada=N` | Partidos pendientes de la jornada N |
| POST | `/predict` | Predicción de un partido (`home_team_id`, `away_team_id`) |
| POST | `/simulate/standings` | Simulación Montecarlo de posición final (`team_id`, `simulations`) |

### Ejemplo: predicción

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"home_team_id": 85, "away_team_id": 92}'
```

```json
{
  "home_team_id": 85,
  "away_team_id": 92,
  "prob_home": 0.4821,
  "prob_draw": 0.2734,
  "prob_away": 0.2445,
  "model": "lgbm_v1"
}
```

---

## 7. Interfaz de usuario

La UI (Streamlit, puerto 8501) expone tres secciones:

### Predicción por Jornada
- Obtiene jornadas pendientes vía `GET /matches/upcoming?limit=200`
- Muestra los partidos de la jornada seleccionada
- Permite predecir partido a partido o la jornada completa
- Genera tabla exportable con pronósticos

### Partido Individual
- Selector con los **20 equipos actuales de Primera División** (obtenidos de `/standings`, no de `/teams`)
- Permite elegir local y visitante libremente
- Muestra barra de probabilidades tricolor y métricas

### Simulación Clasificación
- Selector con los 20 equipos de Primera
- Ejecuta Montecarlo sobre partidos pendientes
- Gráfico de barras con 5 zonas coloreadas (UCL / UEL / UECL / Permanencia / Descenso)
- Métricas por zona y tabla de posiciones con probabilidad ≥ 0.5%
- Muestra partidos pendientes del equipo seleccionado vs total liga

---

## 8. Consideraciones de calidad y limitaciones

### Fortalezas
- Pipeline completamente reproducible y sin leakage temporal
- Features calculadas internamente (sin APIs externas de estadísticas)
- ELO dinámico con momentum para capturar tendencias recientes
- Simulación que considera toda la liga (no solo el equipo objetivo)
- Métrica RPS adecuada para evaluación probabilista

### Limitaciones conocidas

| Limitación | Impacto | Mitigación posible |
|---|---|---|
| Features proxy para partidos futuros | Sesgo en predicciones futuras | Computar features "online" con estado actual |
| Desempate solo por GD | Pequeñas desviaciones en simulación | Añadir criterios reglamentarios (enfrentamiento directo) |
| Modelo estacionario | No reentrenar automáticamente | Programar reentrenamiento periódico |
| Sin features de lesiones/sanciones | Partidos con bajas importantes menos precisos | Integrar APIs de plantillas/alineaciones |
| Cache Streamlit de 60-300s | Datos ligeramente desactualizados en UI | Aceptable en uso normal |

### Recomendaciones para trabajo futuro

1. **Features online**: calcular ELO, forma y posición en tabla en tiempo real para cada predicción futura, en lugar de usar el vector histórico.
2. **Reentrenamiento automático**: trigger al finalizar cada jornada.
3. **Calibración**: aplicar Platt scaling o isotonic regression para mejorar la calibración de probabilidades.
4. **Criterios de desempate completos**: implementar el reglamento LaLiga para la simulación.
5. **Cobertura de tests**: añadir tests unitarios para `FeatureBuilder` y `EloRating`.
