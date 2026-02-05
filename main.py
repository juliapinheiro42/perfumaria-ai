import streamlit as st
import pandas as pd
import time
import os
import torch
import numpy as np
import altair as alt
import plotly.graph_objects as go
import random

# Tenta importar o backend (Mantido igual)
try:
    from infra.gemini_client import GeminiClient
    from core.strategy import StrategyAgent
    from core.discovery import DiscoveryEngine
    from core.model import MoleculeGNN
    from core.market import PerfumeBusinessEngine
    from core.compliance import ComplianceEngine
except ImportError as e:
    # Mock para rodar visualmente caso falte o backend, remova em produção
    st.warning(f"Modo de Visualização (Backend não encontrado: {e})")

    class MockEngine:
        pass
    engine = MockEngine()

# =========================================================
# CONFIGURAÇÃO VISUAL & CSS AVANÇADO
# =========================================================
st.set_page_config(
    page_title="L'Oréal Luxe • AI Lab",
    page_icon="🧴",
    layout="wide",
    initial_sidebar_state="collapsed"  # Começa fechada para focar no painel
)

# Paleta de Cores L'Oréal Luxe
LOREAL_BLACK = "#1A1A1A"
LOREAL_GOLD = "#C5A059"
LOREAL_GOLD_LIGHT = "#E5C585"
LOREAL_WHITE = "#FFFFFF"
LOREAL_BG = "#FAFAFA"
LOREAL_GREEN = "#2D5A27"

# CSS INJETADO
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;1,400&family=Roboto:wght@300;400;500&display=swap');

    /* Reset Geral */
    .stApp {{
        background-color: {LOREAL_BG};
        font-family: 'Roboto', sans-serif;
    }}
    
    /* Remove Padding excessivo do topo */
    .block-container {{
        padding-top: 2rem;
        padding-bottom: 5rem;
    }}

    /* Tipografia de Luxo */
    h1, h2, h3 {{
        font-family: 'Playfair Display', serif !important;
        color: {LOREAL_BLACK};
    }}
    
    h1 {{ font-size: 2.5rem !important; letter-spacing: -1px; }}
    h3 {{ font-size: 1.2rem !important; text-transform: uppercase; letter-spacing: 2px; font-weight: 400; }}

    /* Cards Personalizados */
    .luxury-card {{
        background-color: white;
        padding: 24px;
        border-radius: 2px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.03);
        border-top: 4px solid {LOREAL_GOLD};
        margin-bottom: 20px;
        transition: transform 0.3s ease;
    }}
    .luxury-card:hover {{
        transform: translateY(-2px);
    }}

    /* KPIs */
    .kpi-label {{
        font-family: 'Roboto', sans-serif;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #888;
        margin-bottom: 8px;
    }}
    .kpi-value {{
        font-family: 'Playfair Display', serif;
        font-size: 2.2rem;
        color: {LOREAL_BLACK};
    }}
    .kpi-sub {{
        font-size: 0.8rem;
        color: {LOREAL_GOLD};
    }}

    /* Botões */
    div.stButton > button {{
        background-color: {LOREAL_BLACK};
        color: {LOREAL_WHITE};
        border: none;
        padding: 0.6rem 1.2rem;
        font-family: 'Roboto', sans-serif;
        text-transform: uppercase;
        letter-spacing: 2px;
        font-size: 0.8rem;
        border-radius: 0px; /* Quadrado luxuoso */
        transition: all 0.3s;
        width: 100%;
    }}
    div.stButton > button:hover {{
        background-color: {LOREAL_GOLD};
        color: {LOREAL_WHITE};
        border: none;
        box-shadow: 0 5px 15px rgba(197, 160, 89, 0.3);
    }}
    
    /* Secondary Button (Ghost) */
    div.stButton > button[kind="secondary"] {{
        background-color: transparent;
        border: 1px solid {LOREAL_BLACK};
        color: {LOREAL_BLACK};
    }}

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 20px;
    }}
    .stTabs [data-baseweb="tab"] {{
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 0px;
        color: #888;
        font-family: 'Roboto', sans-serif;
        text-transform: uppercase;
        font-size: 0.8rem;
        letter-spacing: 1px;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: transparent;
        color: {LOREAL_BLACK};
        border-bottom: 2px solid {LOREAL_GOLD};
    }}

    /* Tabelas */
    [data-testid="stDataFrame"] {{
        border: none;
    }}
    
    /* Sidebar */
    [data-testid="stSidebar"] {{
        background-color: #111;
        border-right: 1px solid #333;
    }}
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {{
        color: {LOREAL_WHITE} !important;
    }}
    [data-testid="stSidebar"] label {{
        color: #CCC !important;
    }}
</style>
""", unsafe_allow_html=True)

# =========================================================
# HELPER FUNCTIONS (VISUAL REFINED)
# =========================================================


def render_kpi(label, value, subtext=None, color=LOREAL_BLACK):
    sub_html = f"<div class='kpi-sub'>{subtext}</div>" if subtext else ""
    return f"""
    <div class="luxury-card" style="text-align:center; padding: 15px;">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value" style="color:{color}">{value}</div>
        {sub_html}
    </div>
    """


def refined_olfactory_pyramid(formula):
    """Pirâmide visual mais limpa e elegante."""
    if not formula:
        return go.Figure()

    # Cores refinadas
    COLORS = {"Top": "#E5C585", "Heart": "#C5A059", "Base": "#1A1A1A"}

    fig = go.Figure()

    # Desenhar o triângulo de fundo (Sutil)
    fig.add_shape(type="path", path="M 0,100 L 60,0 L -60,0 Z",
                  fillcolor="rgba(240,240,240, 0.5)", line=dict(color="rgba(0,0,0,0)"))

    categories = {"Top": [], "Heart": [], "Base": []}
    for m in formula:
        cat = m.get('category', 'Heart')
        categories[cat if cat in categories else 'Heart'].append(m)

    # Configuração de posicionamento
    y_ranges = {"Top": (70, 95), "Heart": (35, 65), "Base": (5, 30)}

    for cat, mols in categories.items():
        x_vals, y_vals, sizes, texts = [], [], [], []
        for m in mols:
            y = random.uniform(*y_ranges[cat])
            # Cálculo para manter dentro do triângulo
            max_x = (100 - y) * 0.55
            x = random.uniform(-max_x + 5, max_x - 5)

            x_vals.append(x)
            y_vals.append(y)
            sizes.append(15 + (m.get('weight_factor', 1) * 10))
            texts.append(f"<b>{m['name']}</b><br>{cat}")

        fig.add_trace(go.Scatter(
            x=x_vals, y=y_vals, mode='markers+text',
            marker=dict(size=sizes, color=COLORS[cat], line=dict(
                width=1, color='white'), opacity=0.9),
            text=[t.split('<')[0] if s > 25 else '' for t, s in zip(
                texts, sizes)],  # Só mostra texto se a bolha for grande
            textposition="bottom center",
            textfont=dict(family="Roboto", size=10, color="#555"),
            hoverinfo='text', hovertext=texts
        ))

    fig.update_layout(
        xaxis=dict(visible=False, fixedrange=True),
        yaxis=dict(visible=False, fixedrange=True),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=350,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False
    )
    return fig

# =========================================================
# STATE & MOCK DATA (Se o backend falhar)
# =========================================================
# (O código de carregamento do modelo permanece o mesmo do seu script original)
# ... [INSERIR CÓDIGO DE LOAD_MODEL E SESSION STATE AQUI] ...


# SIMULANDO DADOS PARA VISUALIZAÇÃO (Remova isso ao conectar seu backend)
if 'current_formula' not in st.session_state:
    st.session_state.current_formula = None
if 'round_count' not in st.session_state:
    st.session_state.round_count = 0
if 'history' not in st.session_state:
    st.session_state.history = []

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/9/9d/L%27Or%C3%A9al_logo.svg/2560px-L%27Or%C3%A9al_logo.svg.png", width=150)
    st.markdown(
        f"<p style='color:{LOREAL_GOLD}; font-size:10px; letter-spacing:3px; margin-top:-10px; margin-bottom: 30px;'>LUXE R&D AI LAB</p>", unsafe_allow_html=True)

    st.markdown("### 🧬 DNA Sequence")
    anchors = st.multiselect("Core Accords", [
                             "Bergamot", "Vetiver", "Iris", "Oud", "Rose Centifolia"], default=["Iris", "Oud"])

    st.markdown("### 🎯 Targets")
    st.slider("Sillage Intensity", 0, 10, 8)
    st.slider("Longevity (Hours)", 4, 12, 8)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("RESET LABORATORY"):
        st.session_state.current_formula = None
        st.rerun()

# =========================================================
# MAIN UI
# =========================================================

# HEADER
c1, c2 = st.columns([3, 1])
with c1:
    st.title("Molecular Discovery")
    st.markdown(
        f"AI-Driven Formulation • Batch **#{st.session_state.round_count:02d}**")

with c2:
    if st.button("✨ START SYNTHESIS", type="primary"):
        # Simulação de processamento
        with st.spinner("AI Synthesizing Molecules..."):
            time.sleep(1.5)
            # Mock Data
            st.session_state.current_formula = {
                'molecules': [
                    {'name': 'Bergamot Oil', 'category': 'Top',
                        'weight_factor': 2.5, 'bio': True},
                    {'name': 'Pink Pepper', 'category': 'Top',
                        'weight_factor': 1.2, 'bio': True},
                    {'name': 'Iris Concrete', 'category': 'Heart',
                        'weight_factor': 3.0, 'bio': False},
                    {'name': 'Jasmin Sambac', 'category': 'Heart',
                        'weight_factor': 1.8, 'bio': True},
                    {'name': 'Oud Wood', 'category': 'Base',
                        'weight_factor': 4.5, 'bio': False},
                    {'name': 'White Musk', 'category': 'Base',
                        'weight_factor': 2.0, 'bio': False},
                ],
                'eco_score': 0.82,
                'longevity': 8.4,
                'projection': 7.5,
                'price': 125.00
            }
            st.session_state.round_count += 1
            st.rerun()

st.markdown("---")

# DASHBOARD CONTENT
if st.session_state.current_formula:
    data = st.session_state.current_formula

    # 1. KPI ROW
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(render_kpi("Eco-Score", f"{data['eco_score']*100:.0f}",
                    "Biodegradable Index", LOREAL_GREEN), unsafe_allow_html=True)
    with k2:
        st.markdown(render_kpi(
            "Longevity", f"{data['longevity']}h", "Skin Retention"), unsafe_allow_html=True)
    with k3:
        st.markdown(render_kpi(
            "Projection", f"{data['projection']}/10", "Spatial Sillage"), unsafe_allow_html=True)
    with k4:
        st.markdown(render_kpi(
            "Est. Cost", f"${data['price']}", "Per 100ml Unit"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 2. MAIN ANALYSIS AREA
    row2_1, row2_2 = st.columns([1.5, 1], gap="medium")

    with row2_1:
        st.markdown("### 👃 Olfactory Composition")

        # Tabs para alternar visualizações sem poluir
        tab_viz, tab_data = st.tabs(["MOLECULAR MAP", "FORMULA TABLE"])

        with tab_viz:
            # Gráfico de Pirâmide Customizado
            fig = refined_olfactory_pyramid(data['molecules'])
            st.plotly_chart(fig, use_container_width=True,
                            config={'displayModeBar': False})

        with tab_data:
            df = pd.DataFrame(data['molecules'])
            st.dataframe(
                df,
                column_config={
                    "bio": st.column_config.CheckboxColumn("Green?", width="small"),
                    "weight_factor": st.column_config.ProgressColumn("Concentration", format="%.1f", min_value=0, max_value=5, width="medium")
                },
                hide_index=True,
                use_container_width=True
            )

    with row2_2:
        st.markdown("### 🛠️ Optimization Strategy")

        # Card de "Ação"
        st.markdown(f"""
        <div style="background-color:{LOREAL_BLACK}; color:white; padding:20px; border-radius:2px; margin-bottom:20px;">
            <p style="color:{LOREAL_GOLD}; font-size:10px; letter-spacing:2px; margin:0;">MARKET FIT</p>
            <h2 style="color:white; margin:5px 0 15px 0;">Prestige Segment</h2>
            <p style="font-size:12px; opacity:0.8; line-height:1.6;">
                This formula exhibits high volatility in top notes with a stable oud-base structure. 
                Recommended for Evening/Winter collections.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### Sensory Feedback Loop")
        st.slider("💖 Hedonic Rating", 0, 10, 5, key="hedonic")
        st.slider("🧠 Technical Balance", 0, 10, 5, key="technical")

        c_btn1, c_btn2 = st.columns(2)
        with c_btn1:
            st.button("REINFORCE", type="secondary", use_container_width=True)
        with c_btn2:
            st.button("REJECT", type="secondary", use_container_width=True)

    # 3. COMPLIANCE & HISTORY
    with st.expander("🔎 TECHNICAL COMPLIANCE REPORT (IFRA)", expanded=False):
        st.success("✅ Formula complies with IFRA 51st Amendment standards.")
        st.markdown("- **Allergens:** Low detection")
        st.markdown("- **Phototoxicity:** Negative")

else:
    # EMPTY STATE (Welcome Screen)
    st.markdown(f"""
    <div style="text-align:center; padding: 50px; background-color: white; border: 1px dashed #DDD;">
        <h2 style="color:#CCC;">Waiting for Synthesis</h2>
        <p style="color:#999;">Configure parameters in the sidebar and initialize the AI Engine.</p>
    </div>
    """, unsafe_allow_html=True)
