"""Interfaz grafica LaLiga Analytics — Streamlit.

Secciones:
  1. Prediccion por Jornada  — selecciona jornada completa o partido individual
  2. Simulacion de Clasificacion — probabilidad de posicion final por equipo
"""
import os

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

API = os.getenv("API_URL", "http://api:8000")

st.set_page_config(
    page_title="LaLiga Analytics",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS minimo ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .match-card {
        background: #1e1e2e;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 12px;
        border-left: 4px solid #e63946;
    }
    .team-name { font-size: 1.1rem; font-weight: 600; }
    .prob-label { font-size: 0.75rem; color: #aaa; text-transform: uppercase; letter-spacing: 1px; }
    .prob-value { font-size: 1.4rem; font-weight: 700; }
    .win  { color: #2ec4b6; }
    .draw { color: #f4a261; }
    .loss { color: #e63946; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def get_teams():
    r = requests.get(f"{API}/teams", timeout=5)
    r.raise_for_status()
    return r.json()["teams"]


@st.cache_data(ttl=60)
def get_jornadas():
    r = requests.get(f"{API}/matches/upcoming?limit=200", timeout=5)
    r.raise_for_status()
    matches = r.json()["matches"]
    jornadas = sorted({m["gameweek_week"] for m in matches if m["gameweek_week"]})
    return jornadas, matches


def get_jornada_matches(jornada: int):
    r = requests.get(f"{API}/matches/by-jornada", params={"jornada": jornada}, timeout=5)
    r.raise_for_status()
    return r.json()["matches"]


def predict_match(home_id: int, away_id: int):
    r = requests.post(f"{API}/predict",
                      json={"home_team_id": home_id, "away_team_id": away_id},
                      timeout=10)
    r.raise_for_status()
    return r.json()


def simulate_standings(team_id: int, simulations: int = 5000):
    r = requests.post(f"{API}/simulate/standings",
                      json={"team_id": team_id, "simulations": simulations},
                      timeout=30)
    r.raise_for_status()
    return r.json()


def prob_bar(home: float, draw: float, away: float,
             home_name: str, away_name: str) -> go.Figure:
    fig = go.Figure()
    labels = [home_name, "Empate", away_name]
    values = [home * 100, draw * 100, away * 100]
    colors = ["#2ec4b6", "#f4a261", "#e63946"]
    for label, val, color in zip(labels, values, colors):
        fig.add_trace(go.Bar(
            x=[val], y=["prob"],
            orientation="h",
            name=label,
            marker_color=color,
            text=f"{label}<br><b>{val:.1f}%</b>",
            textposition="inside",
            insidetextanchor="middle",
        ))
    fig.update_layout(
        barmode="stack",
        height=70,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
        xaxis=dict(showticklabels=False, showgrid=False, range=[0, 100]),
        yaxis=dict(showticklabels=False),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def standings_chart(dist: dict, team_name: str) -> go.Figure:
    positions = list(range(1, 21))
    probs = [dist.get(str(p), 0) * 100 for p in positions]
    colors = []
    for p in positions:
        if p <= 4:      colors.append("#2ec4b6")
        elif p <= 6:    colors.append("#f4a261")
        elif p >= 18:   colors.append("#e63946")
        else:           colors.append("#6c757d")

    fig = go.Figure(go.Bar(
        x=positions, y=probs,
        marker_color=colors,
        text=[f"{v:.1f}%" if v >= 2 else "" for v in probs],
        textposition="outside",
    ))
    fig.update_layout(
        title=f"Distribución de posición final — {team_name}",
        xaxis_title="Posición",
        yaxis_title="Probabilidad (%)",
        xaxis=dict(tickvals=positions, ticktext=[str(p) for p in positions]),
        height=420,
        margin=dict(l=20, r=20, t=50, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#eee"),
        bargap=0.15,
    )
    fig.add_vrect(x0=0.5, x1=4.5,   fillcolor="#2ec4b6", opacity=0.08, line_width=0)
    fig.add_vrect(x0=4.5, x1=6.5,   fillcolor="#f4a261", opacity=0.08, line_width=0)
    fig.add_vrect(x0=17.5, x1=20.5, fillcolor="#e63946", opacity=0.08, line_width=0)
    return fig


# ── Sidebar ───────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/LaLiga_logo_2023.svg/200px-LaLiga_logo_2023.svg.png",
             width=140)
    st.markdown("## ⚽ LaLiga Analytics")
    st.markdown("---")
    seccion = st.radio(
        "Sección",
        ["📅 Predicción por Jornada", "🔮 Partido Individual", "📊 Simulación Clasificación"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption("Modelo: Logistic Regression | RPS val: 0.2019")


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 1 — Predicción por Jornada
# ══════════════════════════════════════════════════════════════════════════════

if seccion == "📅 Predicción por Jornada":
    st.title("📅 Predicción por Jornada")

    try:
        jornadas, _ = get_jornadas()
    except Exception as e:
        st.error(f"No se pudo conectar con la API: {e}")
        st.stop()

    if not jornadas:
        st.warning("No hay jornadas programadas disponibles.")
        st.stop()

    col1, col2 = st.columns([2, 1])
    with col1:
        jornada = st.selectbox("Selecciona jornada", jornadas,
                               format_func=lambda j: f"Jornada {j}")
    with col2:
        if st.button("🔄 Predecir jornada completa", use_container_width=True):
            st.session_state["predecir_jornada"] = jornada

    matches = get_jornada_matches(jornada)
    if not matches:
        st.info("No hay partidos disponibles para esta jornada.")
        st.stop()

    st.markdown(f"### Jornada {jornada} — {len(matches)} partidos")

    for m in matches:
        home = m["home_team"]
        away = m["away_team"]
        home_id = m["home_team_id"]
        away_id = m["away_team_id"]
        kickoff = m.get("kickoff_at", "")[:16].replace("T", " ") if m.get("kickoff_at") else ""

        with st.container():
            st.markdown(f"""
            <div class="match-card">
                <span class="prob-label">{kickoff}</span><br>
                <span class="team-name">{home}</span>
                <span style="color:#aaa; margin: 0 8px;">vs</span>
                <span class="team-name">{away}</span>
            </div>
            """, unsafe_allow_html=True)

            col_pred, col_btn = st.columns([4, 1])
            pred_key = f"pred_{home_id}_{away_id}"

            with col_btn:
                if st.button("Predecir", key=f"btn_{home_id}_{away_id}",
                             use_container_width=True):
                    with st.spinner("Calculando..."):
                        try:
                            st.session_state[pred_key] = predict_match(home_id, away_id)
                        except Exception as e:
                            st.error(str(e))

            with col_pred:
                if pred_key in st.session_state:
                    p = st.session_state[pred_key]
                    st.plotly_chart(
                        prob_bar(p["prob_home"], p["prob_draw"], p["prob_away"], home, away),
                        use_container_width=True, config={"displayModeBar": False},
                        key=f"chart_{home_id}_{away_id}",
                    )
                    c1, c2, c3 = st.columns(3)
                    c1.metric(f"🟢 {home}",     f"{p['prob_home']*100:.1f}%")
                    c2.metric("🟡 Empate",       f"{p['prob_draw']*100:.1f}%")
                    c3.metric(f"🔴 {away}",      f"{p['prob_away']*100:.1f}%")

        st.divider()

    if st.session_state.get("predecir_jornada") == jornada:
        st.markdown("### Predicciones completas")
        rows = []
        prog = st.progress(0)
        for i, m in enumerate(matches):
            try:
                p = predict_match(m["home_team_id"], m["away_team_id"])
                rows.append({
                    "Local": m["home_team"],
                    "Visitante": m["away_team"],
                    "P(Local)": f"{p['prob_home']*100:.1f}%",
                    "P(Empate)": f"{p['prob_draw']*100:.1f}%",
                    "P(Visitante)": f"{p['prob_away']*100:.1f}%",
                    "Pronóstico": (
                        m["home_team"] if p["prob_home"] > max(p["prob_draw"], p["prob_away"])
                        else ("Empate" if p["prob_draw"] > p["prob_away"] else m["away_team"])
                    ),
                })
            except Exception:
                pass
            prog.progress((i + 1) / len(matches))
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.session_state.pop("predecir_jornada", None)


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 2 — Partido Individual
# ══════════════════════════════════════════════════════════════════════════════

elif seccion == "🔮 Partido Individual":
    st.title("🔮 Predicción Partido Individual")

    try:
        teams = get_teams()
    except Exception as e:
        st.error(f"No se pudo conectar con la API: {e}")
        st.stop()

    team_map = {t["name"]: t["team_id"] for t in teams}
    names = sorted(team_map.keys())

    col1, col_vs, col2 = st.columns([5, 1, 5])
    with col1:
        home_name = st.selectbox("🏠 Equipo Local", names, key="ind_home")
    with col_vs:
        st.markdown("<br><h3 style='text-align:center'>vs</h3>", unsafe_allow_html=True)
    with col2:
        away_opts = [n for n in names if n != home_name]
        away_name = st.selectbox("✈️ Equipo Visitante", away_opts, key="ind_away")

    if st.button("⚡ Calcular probabilidades", use_container_width=True, type="primary"):
        with st.spinner("Calculando..."):
            try:
                p = predict_match(team_map[home_name], team_map[away_name])
                st.markdown("### Resultado")
                st.plotly_chart(
                    prob_bar(p["prob_home"], p["prob_draw"], p["prob_away"],
                             home_name, away_name),
                    use_container_width=True, config={"displayModeBar": False},
                )
                c1, c2, c3 = st.columns(3)
                c1.metric(f"🟢 {home_name}",  f"{p['prob_home']*100:.1f}%")
                c2.metric("🟡 Empate",          f"{p['prob_draw']*100:.1f}%")
                c3.metric(f"🔴 {away_name}",   f"{p['prob_away']*100:.1f}%")

                ganador = (
                    home_name if p["prob_home"] > max(p["prob_draw"], p["prob_away"])
                    else ("Empate" if p["prob_draw"] > p["prob_away"] else away_name)
                )
                st.info(f"**Pronóstico más probable:** {ganador}")
            except Exception as e:
                st.error(f"Error al predecir: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 3 — Simulación Clasificación
# ══════════════════════════════════════════════════════════════════════════════

else:
    st.title("📊 Simulación de Clasificación Final")
    st.caption("Simulación Montecarlo sobre los partidos pendientes de la temporada.")

    try:
        teams = get_teams()
    except Exception as e:
        st.error(f"No se pudo conectar con la API: {e}")
        st.stop()

    team_map  = {t["name"]: t["team_id"] for t in teams}
    names     = sorted(team_map.keys())

    col1, col2 = st.columns([3, 1])
    with col1:
        team_name = st.selectbox("Selecciona equipo", names)
    with col2:
        sims = st.number_input("Simulaciones", min_value=1000, max_value=20000,
                               value=5000, step=1000)

    if st.button("🎲 Simular clasificación", use_container_width=True, type="primary"):
        with st.spinner(f"Ejecutando {sims:,} simulaciones..."):
            try:
                result = simulate_standings(team_map[team_name], sims)
                dist   = result["position_distribution"]
                pending = result.get("pending_matches_count", -1)
                season_complete = result.get("season_complete", False)

                # Aviso si la temporada ya esta completada en BD
                if season_complete or pending == 0:
                    st.warning(
                        "⚠️ La temporada almacenada está completada — no hay partidos pendientes "
                        "en la base de datos. La posición mostrada es la clasificación final real. "
                        "Ejecuta la ingestión de la temporada actual para simulaciones predictivas."
                    )
                else:
                    st.info(f"🔄 Simulando sobre **{pending} partidos pendientes**")

                st.plotly_chart(
                    standings_chart(dist, team_name),
                    use_container_width=True,
                    config={"displayModeBar": False},
                )

                rows = []
                for pos in range(1, 21):
                    prob = dist.get(str(pos), 0) * 100
                    if prob >= 0.5:
                        zona = (
                            "🔵 Champions" if pos <= 4
                            else "🟠 Europa" if pos <= 6
                            else "🔴 Descenso" if pos >= 18
                            else "—"
                        )
                        rows.append({"Posición": pos, "Probabilidad": f"{prob:.1f}%", "Zona": zona})

                if rows:
                    st.markdown("#### Posiciones con probabilidad ≥ 0.5%")
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

                if not season_complete:
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Champions (Top 4)",
                              f"{sum(dist.get(str(p), 0) for p in range(1, 5)) * 100:.1f}%")
                    c2.metric("Europa (5-6)",
                              f"{sum(dist.get(str(p), 0) for p in range(5, 7)) * 100:.1f}%")
                    c3.metric("Permanencia (7-17)",
                              f"{sum(dist.get(str(p), 0) for p in range(7, 18)) * 100:.1f}%")
                    c4.metric("Descenso (18-20)",
                              f"{sum(dist.get(str(p), 0) for p in range(18, 21)) * 100:.1f}%")

            except Exception as e:
                st.error(f"Error en simulación: {e}")
