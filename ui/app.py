"""LaLiga Analytics — Streamlit UI v2.

Secciones:
  1. Jornada Completa   — Grid visual con prediccion masiva
  2. Partido Individual — LightGBM + Dixon-Coles + mercados
  3. Simulacion Final   — Montecarlo de posicion final
  4. Clasificacion      — Tabla oficial + graficos
  5. Marcador Exacto    — Heatmap Dixon-Coles
  6. Ratings Equipos    — Attack/Defense scatter + barras
  7. Head-to-Head       — Historial + prediccion combinada
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

# ── CSS ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .match-card {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border-radius: 14px; padding: 16px 20px 12px;
    margin-bottom: 8px; border: 1px solid #2d3561;
  }
  .team-home { font-size:1.05rem; font-weight:700; color:#2ec4b6; }
  .team-away { font-size:1.05rem; font-weight:700; color:#e63946; }
  .vs-badge  { font-size:0.8rem; color:#aaa; background:#2a2a4a;
               padding:2px 8px; border-radius:10px; margin:0 6px; }
  .ko-label  { font-size:0.72rem; color:#777; letter-spacing:.5px; }
  .score-badge { display:inline-block; background:#2d3561; color:#fff;
                 font-weight:700; font-size:0.88rem;
                 padding:2px 9px; border-radius:7px; margin-left:6px; }
  .pill { display:inline-block; padding:2px 9px; border-radius:20px;
          font-size:0.75rem; font-weight:600; margin:2px; }
  .pill-g { background:#1a4a3a; color:#2ec4b6; border:1px solid #2ec4b6; }
  .pill-o { background:#3a2a1a; color:#f4a261; border:1px solid #f4a261; }
  .pill-r { background:#3a1a1a; color:#e63946; border:1px solid #e63946; }
  .pill-b { background:#1a2a4a; color:#74b9ff; border:1px solid #74b9ff; }
  .h2h-win  { color:#2ec4b6; font-weight:700; }
  .h2h-draw { color:#f4a261; font-weight:700; }
  .h2h-loss { color:#e63946; font-weight:700; }
</style>
""", unsafe_allow_html=True)

# ── Zonas ───────────────────────────────────────────────────────────────────
ZONAS = [
    (1,  4,  "🔵 Champions League",  "#1d6fa4"),
    (5,  5,  "🟠 Europa League",     "#e07b00"),
    (6,  7,  "🟡 Conference League", "#b5a000"),
    (8, 17,  "— Permanencia",        "#6c757d"),
    (18, 20, "🔴 Descenso",          "#e63946"),
]


def zona_para(pos: int) -> tuple[str, str]:
    for pmin, pmax, label, color in ZONAS:
        if pmin <= pos <= pmax:
            return label, color
    return "—", "#6c757d"


# ── API helpers ───────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def get_primera_teams():
    r = requests.get(f"{API}/standings", timeout=5); r.raise_for_status()
    return sorted([{"team_id": s["team_id"], "name": s["name"]} for s in r.json()["standings"]],
                  key=lambda t: t["name"])


@st.cache_data(ttl=60)
def get_jornadas():
    r = requests.get(f"{API}/matches/upcoming?limit=200", timeout=5); r.raise_for_status()
    ms = r.json()["matches"]
    return sorted({m["gameweek_week"] for m in ms if m["gameweek_week"]}), ms


@st.cache_data(ttl=60)
def get_jornada_matches(jornada: int):
    r = requests.get(f"{API}/matches/by-jornada", params={"jornada": jornada}, timeout=5)
    r.raise_for_status(); return r.json()["matches"]


@st.cache_data(ttl=300)
def get_standings():
    r = requests.get(f"{API}/standings", timeout=5); r.raise_for_status()
    return r.json()["standings"]


@st.cache_data(ttl=600)
def get_ratings():
    r = requests.get(f"{API}/model/ratings", timeout=20)
    return r.json()["ratings"] if r.status_code == 200 else None


@st.cache_data(ttl=300)
def get_match_history(home_id: int, away_id: int):
    r = requests.get(f"{API}/matches/history",
                     params={"home_team_id": home_id, "away_team_id": away_id}, timeout=5)
    return r.json()["matches"] if r.status_code == 200 else []


def predict_match(home_id: int, away_id: int):
    r = requests.post(f"{API}/predict",
                      json={"home_team_id": home_id, "away_team_id": away_id}, timeout=10)
    r.raise_for_status(); return r.json()


def predict_goals(home_id: int, away_id: int):
    r = requests.post(f"{API}/predict/goals",
                      json={"home_team_id": home_id, "away_team_id": away_id}, timeout=20)
    return r.json() if r.status_code == 200 else None


def simulate_standings(team_id: int, simulations: int = 5000):
    r = requests.post(f"{API}/simulate/standings",
                      json={"team_id": team_id, "simulations": simulations}, timeout=60)
    r.raise_for_status(); return r.json()


# ── Graficos compartidos ───────────────────────────────────────────────────────────

def prob_bar(home: float, draw: float, away: float,
             hn: str, an: str, height: int = 62) -> go.Figure:
    fig = go.Figure()
    for label, val, color in [(hn, home*100, "#2ec4b6"), ("Empate", draw*100, "#f4a261"), (an, away*100, "#e63946")]:
        fig.add_trace(go.Bar(x=[val], y=[""], orientation="h", name=label,
                             marker_color=color,
                             text=f"{label}<br><b>{val:.1f}%</b>",
                             textposition="inside", insidetextanchor="middle"))
    fig.update_layout(barmode="stack", height=height,
                      margin=dict(l=0, r=0, t=0, b=0), showlegend=False,
                      xaxis=dict(showticklabels=False, showgrid=False, range=[0, 100]),
                      yaxis=dict(showticklabels=False),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig


def score_heatmap(matrix, hn: str, an: str) -> go.Figure:
    MAX = 6
    z = [[v * 100 for v in row[:MAX]] for row in matrix[:MAX]]
    z_plot = list(reversed(z))
    y_labels = [str(i) for i in range(MAX-1, -1, -1)]
    x_labels = [str(i) for i in range(MAX)]
    text = [[f"{v:.1f}%" if v >= 0.5 else "" for v in row] for row in z_plot]
    fig = go.Figure(go.Heatmap(
        z=z_plot, x=x_labels, y=y_labels,
        text=text, texttemplate="%{text}",
        colorscale="Blues", showscale=True,
        colorbar=dict(title="%", ticksuffix="%"),
    ))
    fig.update_layout(
        title=dict(text=f"Heatmap marcadores — {hn} (Y) vs {an} (X)", font=dict(size=14)),
        xaxis_title=f"Goles {an}", yaxis_title=f"Goles {hn}",
        height=400, margin=dict(l=20, r=20, t=50, b=40),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#eee"),
    )
    return fig


def standings_dist_chart(dist: dict, team_name: str) -> go.Figure:
    positions = list(range(1, 21))
    probs  = [dist.get(str(p), 0) * 100 for p in positions]
    colors = [zona_para(p)[1] for p in positions]
    fig = go.Figure(go.Bar(
        x=positions, y=probs, marker_color=colors,
        text=[f"{v:.1f}%" if v >= 1.5 else "" for v in probs],
        textposition="outside",
    ))
    for x0, x1, c in [(0.5,4.5,"#1d6fa4"),(4.5,5.5,"#e07b00"),(5.5,7.5,"#b5a000"),(17.5,20.5,"#e63946")]:
        fig.add_vrect(x0=x0, x1=x1, fillcolor=c, opacity=0.07, line_width=0)
    fig.update_layout(
        title=f"Distribución posición final — {team_name}",
        xaxis_title="Posición", yaxis_title="Probabilidad (%)",
        xaxis=dict(tickvals=positions, ticktext=[str(p) for p in positions]),
        height=420, margin=dict(l=20, r=20, t=70, b=40),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#eee"), bargap=0.15,
        annotations=[
            dict(x=2.5,  y=1.06, text="UCL",  showarrow=False, xref="x", yref="paper", font=dict(size=10, color="#1d6fa4"), xanchor="center"),
            dict(x=5,    y=1.06, text="UEL",  showarrow=False, xref="x", yref="paper", font=dict(size=10, color="#e07b00"), xanchor="center"),
            dict(x=6.5,  y=1.06, text="UECL", showarrow=False, xref="x", yref="paper", font=dict(size=10, color="#b5a000"), xanchor="center"),
            dict(x=19,   y=1.06, text="DESC", showarrow=False, xref="x", yref="paper", font=dict(size=10, color="#e63946"), xanchor="center"),
        ],
    )
    return fig


def compute_markets(matrix) -> dict:
    """BTTS, Over/Under 2.5 y doble oportunidad desde score matrix."""
    rows, cols = len(matrix), len(matrix[0])
    btts  = sum(matrix[i][j] for i in range(rows) for j in range(cols) if i > 0 and j > 0)
    over  = sum(matrix[i][j] for i in range(rows) for j in range(cols) if i + j > 2)
    ph    = sum(matrix[i][j] for i in range(rows) for j in range(cols) if i > j)
    pd_   = sum(matrix[i][j] for i in range(rows) for j in range(cols) if i == j)
    pa    = sum(matrix[i][j] for i in range(rows) for j in range(cols) if i < j)
    return {
        "btts":   round(btts * 100, 1),
        "over25": round(over * 100, 1),
        "under25":round((1 - over) * 100, 1),
        "dc_1x":  round((ph + pd_) * 100, 1),
        "dc_x2":  round((pd_ + pa) * 100, 1),
        "dc_12":  round((ph + pa) * 100, 1),
    }


def render_pills(mkts: dict):
    items = [
        (f"BTTS {mkts['btts']}%",      "pill-g" if mkts["btts"] > 50 else "pill-o"),
        (f"O2.5 {mkts['over25']}%",     "pill-b" if mkts["over25"] > 50 else "pill-o"),
        (f"U2.5 {mkts['under25']}%",    "pill-b" if mkts["under25"] > 50 else "pill-o"),
        (f"1X {mkts['dc_1x']}%",        "pill-g"),
        (f"X2 {mkts['dc_x2']}%",        "pill-g"),
        (f"12 {mkts['dc_12']}%",        "pill-r"),
    ]
    st.markdown(
        '<div style="margin-top:4px">' +
        "".join(f'<span class="pill {c}">{t}</span>' for t, c in items) +
        "</div>",
        unsafe_allow_html=True,
    )


def top_scores(matrix, n: int = 8) -> pd.DataFrame:
    rows = [
        (f"{i}-{j}", round(matrix[i][j] * 100, 2))
        for i in range(len(matrix)) for j in range(len(matrix[0]))
    ]
    rows.sort(key=lambda x: x[1], reverse=True)
    return pd.DataFrame(rows[:n], columns=["Marcador", "Prob (%)"])


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/LaLiga_logo_2023.svg/200px-LaLiga_logo_2023.svg.png",
        width=130,
    )
    st.markdown("## ⚽ LaLiga Analytics")
    st.markdown("---")
    seccion = st.radio(
        "Sección",
        [
            "📅 Jornada Completa",
            "🔮 Partido Individual",
            "📊 Simulación Final",
            "🏅 Clasificación",
            "🎯 Marcador Exacto",
            "⚡ Ratings Equipos",
            "🇨🇦 Head-to-Head",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption("LightGBM · Dixon-Coles · Montecarlo")
    st.markdown("---")
    st.markdown("**Zonas**")
    for _, _, label, color in ZONAS:
        st.markdown(
            f'<span style="display:inline-block;width:12px;height:12px;'
            f'background:{color};border-radius:2px;margin-right:6px;"></span>{label}',
            unsafe_allow_html=True,
        )


# ════════════════════════════════════════════════════════════════════════════════
# 1 — JORNADA COMPLETA
# ════════════════════════════════════════════════════════════════════════════════

if seccion == "📅 Jornada Completa":
    st.title("📅 Predicción Jornada Completa")

    try:
        jornadas, _ = get_jornadas()
    except Exception as e:
        st.error(f"No se pudo conectar con la API: {e}"); st.stop()

    if not jornadas:
        st.warning("No hay jornadas programadas."); st.stop()

    col_sel, col_btn = st.columns([3, 1])
    with col_sel:
        jornada = st.selectbox("", jornadas,
                               format_func=lambda j: f"Jornada {j}",
                               label_visibility="collapsed")
    with col_btn:
        do_pred = st.button("⚡ Predecir Jornada", use_container_width=True, type="primary")

    matches = get_jornada_matches(jornada)
    if not matches:
        st.info("No hay partidos disponibles para esta jornada."); st.stop()

    cache_key = f"jornada_{jornada}_preds"

    # —— Cargar predicciones
    if do_pred:
        preds = {}
        with st.spinner(f"Calculando {len(matches)} partidos…"):
            prog = st.progress(0)
            for i, m in enumerate(matches):
                try:
                    lgbm = predict_match(m["home_team_id"], m["away_team_id"])
                    dc   = predict_goals(m["home_team_id"], m["away_team_id"])
                    preds[m["match_id"]] = {"lgbm": lgbm, "dc": dc}
                except Exception:
                    pass
                prog.progress((i + 1) / len(matches))
            prog.empty()
        st.session_state[cache_key] = preds

    preds = st.session_state.get(cache_key, {})

    st.markdown(f"### Jornada {jornada} — {len(matches)} partidos")
    if not preds:
        # Vista previa sin predicciones
        for m in matches:
            ko = (m.get("kickoff_at") or "")[:16].replace("T", " ")
            st.markdown(
                f'<div class="match-card">'
                f'<div class="ko-label">🕐 {ko}</div>'
                f'<div style="margin:8px 0;">'
                f'<span class="team-home">{m["home_team"]}</span>'
                f'<span class="vs-badge">VS</span>'
                f'<span class="team-away">{m["away_team"]}</span>'
                f'</div></div>',
                unsafe_allow_html=True,
            )
        st.info("👆 Pulsa **Predecir Jornada** para calcular todos los partidos a la vez")
    else:
        # —— Grid 2 columnas
        cols = st.columns(2)
        for idx, m in enumerate(matches):
            col = cols[idx % 2]
            pred = preds.get(m["match_id"], {})
            p, dc = pred.get("lgbm"), pred.get("dc")
            ko = (m.get("kickoff_at") or "")[:16].replace("T", " ")

            with col:
                if p:
                    prob_max = max(p["prob_home"], p["prob_draw"], p["prob_away"])
                    winner = (
                        m["home_team"] if p["prob_home"] == prob_max
                        else ("Empate" if p["prob_draw"] == prob_max else m["away_team"])
                    )
                    pill_cls = "pill-g" if prob_max > 0.55 else "pill-o"
                    score_html = (
                        f'<span class="score-badge">{dc["most_likely_score"]}</span>'
                        if dc else ""
                    )
                    xg_html = (
                        f'<span style="font-size:.72rem;color:#aaa;margin-left:6px;">'
                        f'xG {dc["lambda_home"]:.2f} — {dc["lambda_away"]:.2f}</span>'
                        if dc else ""
                    )
                    st.markdown(
                        f'<div class="match-card">'
                        f'<div class="ko-label">🕐 {ko}</div>'
                        f'<div style="display:flex;justify-content:space-between;align-items:center;margin:8px 0 4px;">'
                        f'<span class="team-home">{m["home_team"]}</span>'
                        f'<span class="vs-badge">VS</span>'
                        f'<span class="team-away">{m["away_team"]}</span>'
                        f'</div>'
                        f'<div><span class="pill {pill_cls}">✓ {winner} ({prob_max*100:.0f}%)</span>'
                        f'{score_html}{xg_html}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                    st.plotly_chart(
                        prob_bar(p["prob_home"], p["prob_draw"], p["prob_away"],
                                 m["home_team"], m["away_team"]),
                        use_container_width=True,
                        config={"displayModeBar": False},
                        key=f"jbar_{m['match_id']}",
                    )
                    if dc:
                        render_pills(compute_markets(dc["score_matrix"]))
                else:
                    st.markdown(
                        f'<div class="match-card"><div class="ko-label">🕐 {ko}</div>'
                        f'<div style="margin:8px 0;">'
                        f'<span class="team-home">{m["home_team"]}</span>'
                        f'<span class="vs-badge">VS</span>'
                        f'<span class="team-away">{m["away_team"]}</span></div>'
                        f'<span style="color:#888;font-size:.8rem;">Error al predecir</span></div>',
                        unsafe_allow_html=True,
                    )
                st.markdown("<br>", unsafe_allow_html=True)

        # —— Tabla resumen
        st.markdown("---")
        st.markdown("#### 📋 Resumen jornada")
        rows = []
        for m in matches:
            pred = preds.get(m["match_id"], {})
            p, dc = pred.get("lgbm"), pred.get("dc")
            if p:
                pm = max(p["prob_home"], p["prob_draw"], p["prob_away"])
                win = (
                    m["home_team"] if p["prob_home"] == pm
                    else ("Empate" if p["prob_draw"] == pm else m["away_team"])
                )
                row = {
                    "Local":      m["home_team"],
                    "Visitante":  m["away_team"],
                    "P(L) %":     f"{p['prob_home']*100:.1f}",
                    "P(E) %":     f"{p['prob_draw']*100:.1f}",
                    "P(V) %":     f"{p['prob_away']*100:.1f}",
                    "Pronóstico": win,
                    "Confianza":  f"{pm*100:.0f}%",
                }
                if dc:
                    row["Marcador"] = dc["most_likely_score"]
                    row["xG L"] = f"{dc['lambda_home']:.2f}"
                    row["xG V"] = f"{dc['lambda_away']:.2f}"
                    mkts = compute_markets(dc["score_matrix"])
                    row["BTTS"]   = f"{mkts['btts']}%"
                    row["O2.5"]   = f"{mkts['over25']}%"
                rows.append(row)
        if rows:
            df = pd.DataFrame(rows)
            st.dataframe(
                df.sort_values("Confianza", ascending=False),
                use_container_width=True, hide_index=True,
            )


# ════════════════════════════════════════════════════════════════════════════════
# 2 — PARTIDO INDIVIDUAL
# ════════════════════════════════════════════════════════════════════════════════

elif seccion == "🔮 Partido Individual":
    st.title("🔮 Partido Individual")
    st.caption("🏆 Solo equipos actuales de Primera División")

    try:
        teams = get_primera_teams()
    except Exception as e:
        st.error(f"Error API: {e}"); st.stop()

    team_map = {t["name"]: t["team_id"] for t in teams}
    names = sorted(team_map.keys())

    c1, c_vs, c2 = st.columns([5, 1, 5])
    with c1:
        home_name = st.selectbox("🏠 Local", names, key="ind_home")
    with c_vs:
        st.markdown("<br><h3 style='text-align:center'>vs</h3>", unsafe_allow_html=True)
    with c2:
        away_name = st.selectbox("✈️ Visitante", [n for n in names if n != home_name], key="ind_away")

    if st.button("⚡ Calcular predicciones", use_container_width=True, type="primary"):
        hid, aid = team_map[home_name], team_map[away_name]
        with st.spinner("Calculando…"):
            try:
                p  = predict_match(hid, aid)
                dc = predict_goals(hid, aid)
            except Exception as e:
                st.error(f"Error: {e}"); st.stop()

        st.markdown("---")
        tab_lgbm, tab_dc = st.tabs(["🤖 LightGBM — Resultado", "🎯 Dixon-Coles — Marcador & Mercados"])

        with tab_lgbm:
            st.plotly_chart(prob_bar(p["prob_home"], p["prob_draw"], p["prob_away"],
                                     home_name, away_name, height=70),
                            use_container_width=True, config={"displayModeBar": False})
            c1, c2, c3 = st.columns(3)
            c1.metric(f"🟢 {home_name}", f"{p['prob_home']*100:.1f}%")
            c2.metric("🟡 Empate",        f"{p['prob_draw']*100:.1f}%")
            c3.metric(f"🔴 {away_name}",  f"{p['prob_away']*100:.1f}%")
            winner = (
                home_name if p["prob_home"] > max(p["prob_draw"], p["prob_away"])
                else ("Empate" if p["prob_draw"] > p["prob_away"] else away_name)
            )
            st.info(f"**Pronóstico más probable:** {winner}")
            if dc:
                mkts = compute_markets(dc["score_matrix"])
                st.markdown("#### 📀 Mercados de apuestas")
                m1, m2, m3 = st.columns(3)
                m1.metric("BTTS (Ambos marcan)", f"{mkts['btts']}%")
                m2.metric("Over 2.5 goles",       f"{mkts['over25']}%")
                m3.metric("Under 2.5 goles",       f"{mkts['under25']}%")
                m4, m5, m6 = st.columns(3)
                m4.metric("Doble op. 1X", f"{mkts['dc_1x']}%")
                m5.metric("Doble op. X2", f"{mkts['dc_x2']}%")
                m6.metric("Doble op. 12", f"{mkts['dc_12']}%")

        with tab_dc:
            if dc:
                c1, c2, c3 = st.columns(3)
                c1.metric(f"⚽ xG {home_name}", f"{dc['lambda_home']:.2f}")
                c2.metric(f"⚽ xG {away_name}", f"{dc['lambda_away']:.2f}")
                c3.metric("Marcador más probable", dc["most_likely_score"])
                st.plotly_chart(score_heatmap(dc["score_matrix"], home_name, away_name),
                                use_container_width=True, config={"displayModeBar": False})
                st.markdown("#### Top 8 marcadores")
                st.dataframe(top_scores(dc["score_matrix"], 8),
                             use_container_width=True, hide_index=True)
            else:
                st.warning("⚠️ Dixon-Coles aún no disponible. Reintenta en unos segundos.")


# ════════════════════════════════════════════════════════════════════════════════
# 3 — SIMULACIÓN FINAL
# ════════════════════════════════════════════════════════════════════════════════

elif seccion == "📊 Simulación Final":
    st.title("📊 Simulación de Clasificación Final")
    st.caption("Simulación Montecarlo sobre los partidos pendientes de la temporada.")

    try:
        teams = get_primera_teams()
    except Exception as e:
        st.error(f"Error API: {e}"); st.stop()

    team_map = {t["name"]: t["team_id"] for t in teams}
    names    = sorted(team_map.keys())

    c1, c2 = st.columns([3, 1])
    with c1:
        team_name = st.selectbox("Equipo", names)
    with c2:
        sims = st.number_input("Simulaciones", 1000, 20000, 5000, step=1000)

    if st.button("🎲 Simular clasificación", use_container_width=True, type="primary"):
        with st.spinner(f"Ejecutando {sims:,} simulaciones…"):
            try:
                result = simulate_standings(team_map[team_name], sims)
            except Exception as e:
                st.error(f"Error: {e}"); st.stop()

        dist            = result["position_distribution"]
        season_complete = result.get("season_complete", False)
        team_pending    = result.get("team_pending_count", 0)
        league_pending  = result.get("pending_matches_count", 0)

        if season_complete:
            st.warning("⚠️ Temporada completa — clasificación ya definida.")
        else:
            st.info(f"🔄 **{team_name}** tiene **{team_pending} partido(s) pendiente(s)** "
                    f"({league_pending} en total en la liga)")

        st.plotly_chart(standings_dist_chart(dist, team_name),
                        use_container_width=True, config={"displayModeBar": False})

        # Metricas de zona
        if not season_complete:
            z1, z2, z3, z4, z5 = st.columns(5)
            z1.metric("🔵 Champions (1–4)", f"{sum(dist.get(str(p),0) for p in range(1, 5))*100:.1f}%")
            z2.metric("🟠 Europa (5)",        f"{dist.get('5',0)*100:.1f}%")
            z3.metric("🟡 Conference (6–7)", f"{sum(dist.get(str(p),0) for p in range(6, 8))*100:.1f}%")
            z4.metric("— Permanencia",         f"{sum(dist.get(str(p),0) for p in range(8, 18))*100:.1f}%")
            z5.metric("🔴 Descenso (18–20)",  f"{sum(dist.get(str(p),0) for p in range(18, 21))*100:.1f}%")

        rows = [
            {"Posición": p, "Probabilidad": f"{dist.get(str(p),0)*100:.1f}%",
             "Zona": zona_para(p)[0]}
            for p in range(1, 21) if dist.get(str(p), 0) * 100 >= 0.5
        ]
        if rows:
            st.markdown("#### Posiciones con probabilidad ≥ 0.5%")
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════════════════════
# 4 — CLASIFICACIÓN ACTUAL
# ════════════════════════════════════════════════════════════════════════════════

elif seccion == "🏅 Clasificación":
    st.title("🏅 Clasificación Actual")

    try:
        rows = get_standings()
    except Exception as e:
        st.error(f"Error API: {e}"); st.stop()

    if not rows:
        st.warning("Sin datos de clasificación."); st.stop()

    df = pd.DataFrame(rows)
    df["Zona"] = df["position"].apply(lambda p: zona_para(p)[0])
    df["DG"]   = df["goal_difference"].apply(lambda x: f"+{x}" if x > 0 else str(x))

    display = df[["position", "name", "points", "played",
                  "won", "drawn", "lost", "goals_for", "goals_against", "DG", "Zona"]].rename(columns={
        "position": "Pos", "name": "Equipo", "points": "Pts", "played": "PJ",
        "won": "G", "drawn": "E", "lost": "P",
        "goals_for": "GF", "goals_against": "GC",
    })
    st.dataframe(display, use_container_width=True, hide_index=True, height=740)

    st.markdown("---")
    c1, c2 = st.columns(2)
    zone_colors = [zona_para(p)[1] for p in df["position"]]

    with c1:
        fig_pts = go.Figure(go.Bar(
            x=df["shortname"].fillna(df["name"]),
            y=df["points"],
            marker_color=zone_colors,
            text=df["points"], textposition="outside",
        ))
        fig_pts.update_layout(
            title="Puntos por equipo", xaxis_tickangle=-40, height=380,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#eee"), margin=dict(l=10, r=10, t=40, b=90),
            yaxis=dict(showgrid=True, gridcolor="#2a2a4a"),
        )
        st.plotly_chart(fig_pts, use_container_width=True, config={"displayModeBar": False})

    with c2:
        fig_gd = go.Figure(go.Bar(
            x=df["shortname"].fillna(df["name"]),
            y=df["goal_difference"],
            marker_color=["#2ec4b6" if v >= 0 else "#e63946" for v in df["goal_difference"]],
            text=df["DG"], textposition="outside",
        ))
        fig_gd.update_layout(
            title="Diferencia de goles", xaxis_tickangle=-40, height=380,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#eee"), margin=dict(l=10, r=10, t=40, b=90),
            yaxis=dict(showgrid=True, gridcolor="#2a2a4a"),
        )
        st.plotly_chart(fig_gd, use_container_width=True, config={"displayModeBar": False})

    # Goles a favor vs en contra
    c3, c4 = st.columns(2)
    with c3:
        fig_gf = go.Figure()
        fig_gf.add_trace(go.Bar(
            name="Goles a favor",
            x=df["shortname"].fillna(df["name"]), y=df["goals_for"],
            marker_color="#2ec4b6",
        ))
        fig_gf.add_trace(go.Bar(
            name="Goles en contra",
            x=df["shortname"].fillna(df["name"]), y=df["goals_against"],
            marker_color="#e63946",
        ))
        fig_gf.update_layout(
            barmode="group", title="GF vs GC", xaxis_tickangle=-40, height=380,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#eee"), margin=dict(l=10, r=10, t=40, b=90),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            yaxis=dict(showgrid=True, gridcolor="#2a2a4a"),
        )
        st.plotly_chart(fig_gf, use_container_width=True, config={"displayModeBar": False})

    with c4:
        fig_wr = go.Figure(go.Bar(
            x=df["shortname"].fillna(df["name"]),
            y=(df["won"] / df["played"] * 100).round(1),
            marker_color=zone_colors,
            text=(df["won"] / df["played"] * 100).round(1).astype(str) + "%",
            textposition="outside",
        ))
        fig_wr.update_layout(
            title="% Victorias", xaxis_tickangle=-40, height=380,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#eee"), margin=dict(l=10, r=10, t=40, b=90),
            yaxis=dict(showgrid=True, gridcolor="#2a2a4a", ticksuffix="%"),
        )
        st.plotly_chart(fig_wr, use_container_width=True, config={"displayModeBar": False})


# ════════════════════════════════════════════════════════════════════════════════
# 5 — MARCADOR EXACTO
# ════════════════════════════════════════════════════════════════════════════════

elif seccion == "🎯 Marcador Exacto":
    st.title("🎯 Predicción de Marcador Exacto")
    st.caption("Modelo Dixon-Coles — Bivariate Poisson con corrección ρ para goles bajos")

    try:
        teams = get_primera_teams()
    except Exception as e:
        st.error(f"Error API: {e}"); st.stop()

    team_map = {t["name"]: t["team_id"] for t in teams}
    names = sorted(team_map.keys())

    c1, c_vs, c2 = st.columns([5, 1, 5])
    with c1:
        home_name = st.selectbox("🏠 Local", names, key="me_home")
    with c_vs:
        st.markdown("<br><h3 style='text-align:center'>vs</h3>", unsafe_allow_html=True)
    with c2:
        away_name = st.selectbox("✈️ Visitante", [n for n in names if n != home_name], key="me_away")

    if st.button("🎯 Calcular marcadores", use_container_width=True, type="primary"):
        with st.spinner("Calculando modelo Dixon-Coles…"):
            dc = predict_goals(team_map[home_name], team_map[away_name])

        if not dc:
            st.warning("⚠️ Modelo no disponible aún. Reintenta en unos segundos."); st.stop()

        # Metricas principales
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(f"⚽ xG {home_name}", f"{dc['lambda_home']:.2f}")
        c2.metric(f"⚽ xG {away_name}", f"{dc['lambda_away']:.2f}")
        c3.metric("🎯 Marcador probable", dc["most_likely_score"])
        mkts = compute_markets(dc["score_matrix"])
        c4.metric("BTTS (Ambos marcan)", f"{mkts['btts']}%")

        st.markdown("---")
        left, right = st.columns([2, 1])
        with left:
            st.plotly_chart(score_heatmap(dc["score_matrix"], home_name, away_name),
                            use_container_width=True, config={"displayModeBar": False})
        with right:
            st.markdown("#### 📊 Top 10 marcadores")
            st.dataframe(top_scores(dc["score_matrix"], 10),
                         use_container_width=True, hide_index=True)
            st.markdown("#### 📀 Mercados")
            st.metric("Over 2.5",  f"{mkts['over25']}%")
            st.metric("Under 2.5", f"{mkts['under25']}%")
            st.metric("Doble 1X",  f"{mkts['dc_1x']}%")
            st.metric("Doble X2",  f"{mkts['dc_x2']}%")
            st.metric("Doble 12",  f"{mkts['dc_12']}%")

        # Probabilidades por numero de goles totales
        st.markdown("---")
        st.markdown("#### Distribución goles totales")
        mat = dc["score_matrix"]
        goal_probs = {}
        for i in range(len(mat)):
            for j in range(len(mat[0])):
                g = i + j
                goal_probs[g] = goal_probs.get(g, 0) + mat[i][j]
        gdf = pd.DataFrame(
            [(g, round(v * 100, 2)) for g, v in sorted(goal_probs.items()) if g <= 8],
            columns=["Goles totales", "Prob (%)"]
        )
        fig_g = go.Figure(go.Bar(
            x=gdf["Goles totales"], y=gdf["Prob (%)"],
            marker_color=["#e63946" if g <= 2 else "#2ec4b6" for g in gdf["Goles totales"]],
            text=gdf["Prob (%)"].astype(str) + "%", textposition="outside",
        ))
        fig_g.add_vline(x=2.5, line_dash="dash", line_color="#f4a261",
                        annotation_text="Over/Under 2.5", annotation_position="top right")
        fig_g.update_layout(
            height=320, xaxis_title="Goles totales", yaxis_title="Probabilidad (%)",
            xaxis=dict(tickvals=list(range(9))),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#eee"), margin=dict(l=20, r=20, t=20, b=40),
            yaxis=dict(showgrid=True, gridcolor="#2a2a4a"),
        )
        st.plotly_chart(fig_g, use_container_width=True, config={"displayModeBar": False})


# ════════════════════════════════════════════════════════════════════════════════
# 6 — RATINGS EQUIPOS
# ════════════════════════════════════════════════════════════════════════════════

elif seccion == "⚡ Ratings Equipos":
    st.title("⚡ Ratings de Equipos (Dixon-Coles)")
    st.caption(
        "α ataque: cuantos goles genera el equipo en promedio. "
        "β defensa: cuantos goles permite (menor = mejor defensa)."
    )

    ratings = get_ratings()
    if not ratings:
        st.warning("⚠️ El modelo Dixon-Coles aún no está listo. Reintenta en unos segundos.")
        st.stop()

    df = pd.DataFrame(ratings).sort_values("attack", ascending=False)

    # Rankings numéricos
    df["rank_att"] = df["attack"].rank(ascending=False).astype(int)
    df["rank_def"] = df["defense"].rank(ascending=True).astype(int)  # menor beta = mejor defensa

    c1, c2 = st.columns(2)

    with c1:
        fig_att = go.Figure(go.Bar(
            x=df["attack"], y=df["name"],
            orientation="h",
            marker_color=["#2ec4b6" if v > 0 else "#e63946" for v in df["attack"]],
            text=df["attack"].round(3), textposition="outside",
        ))
        fig_att.update_layout(
            title="🔥 Rating de Ataque (α)",
            height=600,
            xaxis_title="α (mayor = más goleador)",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#eee"), margin=dict(l=10, r=60, t=50, b=20),
            yaxis=dict(autorange="reversed"),
            xaxis=dict(showgrid=True, gridcolor="#2a2a4a"),
        )
        st.plotly_chart(fig_att, use_container_width=True, config={"displayModeBar": False})

    with c2:
        df_def = df.sort_values("defense")
        fig_def = go.Figure(go.Bar(
            x=df_def["defense"], y=df_def["name"],
            orientation="h",
            marker_color=["#2ec4b6" if v < 0 else "#e63946" for v in df_def["defense"]],
            text=df_def["defense"].round(3), textposition="outside",
        ))
        fig_def.update_layout(
            title="🛡️ Rating de Defensa (β — menor = mejor)",
            height=600,
            xaxis_title="β (menor = menos goles concedidos)",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#eee"), margin=dict(l=10, r=60, t=50, b=20),
            yaxis=dict(autorange="reversed"),
            xaxis=dict(showgrid=True, gridcolor="#2a2a4a"),
        )
        st.plotly_chart(fig_def, use_container_width=True, config={"displayModeBar": False})

    # Mapa de calidad: Ataque vs Defensa
    st.markdown("---")
    st.markdown("#### 🗺️ Mapa de Calidad (Ataque vs Defensa)")
    st.caption("Arriba-derecha = mejor ataque Y mejor defensa (equipos élite)")

    fig_map = go.Figure()
    fig_map.add_vline(x=0, line_dash="dash", line_color="#444")
    fig_map.add_hline(y=0, line_dash="dash", line_color="#444")
    fig_map.add_trace(go.Scatter(
        x=df["attack"],
        y=[-d for d in df["defense"]],  # negar: mayor y = mejor defensa
        mode="markers+text",
        text=df["name"],
        textposition="top center",
        textfont=dict(size=9, color="#ccc"),
        marker=dict(
            size=14,
            color=df["attack"],
            colorscale="RdYlGn",
            showscale=True,
            colorbar=dict(title="Ataque"),
            line=dict(width=1, color="#333"),
        ),
        hovertemplate="<b>%{text}</b><br>α ataque: %{x:.3f}<br>β defensa: %{y:.3f}<extra></extra>",
    ))
    fig_map.update_layout(
        height=520,
        xaxis_title="α Ataque (mayor = más goleador)",
        yaxis_title="-β Defensa (mayor = más sólida)",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#eee"), margin=dict(l=40, r=20, t=20, b=50),
        xaxis=dict(showgrid=True, gridcolor="#2a2a4a"),
        yaxis=dict(showgrid=True, gridcolor="#2a2a4a"),
    )
    st.plotly_chart(fig_map, use_container_width=True, config={"displayModeBar": False})


# ════════════════════════════════════════════════════════════════════════════════
# 7 — HEAD-TO-HEAD
# ════════════════════════════════════════════════════════════════════════════════

else:  # seccion == "Head-to-Head"
    st.title("🇨🇦 Head-to-Head")
    st.caption("Historial directo entre dos equipos + predicción combinada")

    try:
        teams = get_primera_teams()
    except Exception as e:
        st.error(f"Error API: {e}"); st.stop()

    team_map = {t["name"]: t["team_id"] for t in teams}
    names = sorted(team_map.keys())

    c1, c_vs, c2 = st.columns([5, 1, 5])
    with c1:
        h2h_home = st.selectbox("🏠 Equipo A", names, key="h2h_home")
    with c_vs:
        st.markdown("<br><h3 style='text-align:center'>vs</h3>", unsafe_allow_html=True)
    with c2:
        h2h_away = st.selectbox("✈️ Equipo B", [n for n in names if n != h2h_home], key="h2h_away")

    if st.button("🔍 Analizar H2H", use_container_width=True, type="primary"):
        hid, aid = team_map[h2h_home], team_map[h2h_away]

        with st.spinner("Cargando historial y predicciones…"):
            history = get_match_history(hid, aid)
            try:
                p  = predict_match(hid, aid)
                dc = predict_goals(hid, aid)
            except Exception as e:
                p, dc = None, None
                st.warning(f"Error al predecir: {e}")

        st.markdown("---")

        # —— Estadísticas históricas
        if history:
            wins_h = sum(1 for m in history if
                         (m["home_team_id"] == hid and m["result"] == "home") or
                         (m["away_team_id"] == hid and m["result"] == "away"))
            draws   = sum(1 for m in history if m["result"] == "draw")
            wins_a  = len(history) - wins_h - draws

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Partidos analizados", len(history))
            c2.metric(f"🟢 Victorias {h2h_home}", wins_h)
            c3.metric("🟡 Empates", draws)
            c4.metric(f"🔴 Victorias {h2h_away}", wins_a)

            # Pie chart H2H
            fig_pie = go.Figure(go.Pie(
                labels=[h2h_home, "Empate", h2h_away],
                values=[wins_h, draws, wins_a],
                marker=dict(colors=["#2ec4b6", "#f4a261", "#e63946"]),
                textinfo="label+percent",
                hole=0.38,
            ))
            fig_pie.update_layout(
                height=320, showlegend=False,
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#eee"),
                margin=dict(l=20, r=20, t=20, b=20),
            )
            st.plotly_chart(fig_pie, use_container_width=True,
                            config={"displayModeBar": False})

            # Tabla de partidos
            st.markdown("#### Partidos recientes")
            h_rows = []
            for m in history:
                is_h_home = m["home_team_id"] == hid
                h_score   = m["home_score"] if is_h_home else m["away_score"]
                a_score   = m["away_score"] if is_h_home else m["home_score"]
                result    = m["result"]
                outcome   = (
                    "V" if (is_h_home and result == "home") or (not is_h_home and result == "away")
                    else ("E" if result == "draw" else "D")
                )
                ko = (m.get("kickoff_at") or "")[:10]
                h_rows.append({
                    "Fecha":     ko,
                    "Local":     m["home_team"],
                    "Resultado": f"{m['home_score']}-{m['away_score']}",
                    "Visitante": m["away_team"],
                    f"{h2h_home}": outcome,
                })
            st.dataframe(pd.DataFrame(h_rows), use_container_width=True, hide_index=True)
        else:
            st.info("ℹ️ No hay partidos históricos entre estos equipos en la base de datos.")

        # —— Predicción próximo enfrentamiento
        if p:
            st.markdown("---")
            st.markdown(f"#### 🔮 Predicción: {h2h_home} — {h2h_away}")
            st.plotly_chart(
                prob_bar(p["prob_home"], p["prob_draw"], p["prob_away"],
                         h2h_home, h2h_away, height=70),
                use_container_width=True, config={"displayModeBar": False},
            )
            pc1, pc2, pc3 = st.columns(3)
            pc1.metric(f"🟢 {h2h_home}", f"{p['prob_home']*100:.1f}%")
            pc2.metric("🟡 Empate",        f"{p['prob_draw']*100:.1f}%")
            pc3.metric(f"🔴 {h2h_away}",  f"{p['prob_away']*100:.1f}%")

            if dc:
                st.markdown("---")
                dcl, dcr = st.columns([2, 1])
                with dcl:
                    st.plotly_chart(
                        score_heatmap(dc["score_matrix"], h2h_home, h2h_away),
                        use_container_width=True, config={"displayModeBar": False},
                    )
                with dcr:
                    st.metric(f"⚽ xG {h2h_home}", f"{dc['lambda_home']:.2f}")
                    st.metric(f"⚽ xG {h2h_away}", f"{dc['lambda_away']:.2f}")
                    st.metric("🎯 Marcador probable", dc["most_likely_score"])
                    mkts = compute_markets(dc["score_matrix"])
                    st.markdown("**Mercados**")
                    render_pills(mkts)
