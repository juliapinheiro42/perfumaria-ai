import streamlit as st
import pandas as pd
import time
import os
import torch
import numpy as np
import altair as alt

# Imports de infraestrutura (mantidos conforme original)
from infra.gemini_client import GeminiClient
from core.strategy import StrategyAgent
from core.discovery import DiscoveryEngine
from core.model import MoleculeGNN
from core.market import PerfumeBusinessEngine

# =========================================================
# CONFIGURAÇÃO VISUAL L'ORÉAL LUXE (DESIGN EDITORIAL)
# =========================================================
st.set_page_config(
    page_title="L'Oréal Luxe • AI Lab",
    page_icon="🧴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paleta de Cores de Luxo
LOREAL_BLACK = "#000000"
LOREAL_GOLD = "#C5A059" 
LOREAL_WHITE = "#FFFFFF"
LOREAL_GREY = "#F2F2F2"

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@100;300;400;600&display=swap');

    .stApp {{
        background-color: {LOREAL_WHITE};
        font-family: 'Inter', sans-serif;
    }}
    
    /* Headers de Luxo */
    h1, h2, h3 {{
        font-family: 'Inter', sans-serif;
        font-weight: 100 !important;
        text-transform: uppercase;
        letter-spacing: 5px !important;
        color: {LOREAL_BLACK};
    }}

    /* Sidebar Customizada */
    [data-testid="stSidebar"] {{
        background-color: {LOREAL_BLACK};
        border-right: 1px solid {LOREAL_GOLD};
    }}
    
    /* Botões Estilo Boutique */
    .stButton > button {{
        background-color: transparent;
        color: {LOREAL_BLACK};
        border: 1px solid {LOREAL_BLACK};
        border-radius: 0px;
        letter-spacing: 2px;
        font-weight: 300;
        transition: all 0.3s;
        width: 100%;
    }}
    .stButton > button:hover {{
        background-color: {LOREAL_BLACK};
        color: {LOREAL_GOLD} !important;
    }}

    /* KPI Cards Robustos */
    .kpi-card {{
        background-color: {LOREAL_WHITE};
        padding: 20px;
        border: 1px solid #EEE;
        border-bottom: 3px solid {LOREAL_GOLD};
        text-align: center;
    }}
    .kpi-val {{
        font-size: 2rem;
        font-weight: 100;
        color: {LOREAL_BLACK};
    }}
    .kpi-lab {{
        font-size: 0.65rem;
        letter-spacing: 2px;
        color: {LOREAL_GOLD};
        text-transform: uppercase;
        margin-bottom: 5px;
    }}

    /* Container de Negócios (Dark Mode) */
    .biz-card {{
        background-color: {LOREAL_BLACK};
        color: {LOREAL_WHITE};
        padding: 30px;
        border-radius: 0px;
    }}
</style>
""", unsafe_allow_html=True)

# =========================================================
# SISTEMA E CACHE
# =========================================================
@st.cache_resource
def load_model():
    model = MoleculeGNN(num_node_features=5)
    try:
        if os.path.exists("results/perfume_gnn.pth"):
            model.load_state_dict(torch.load("results/perfume_gnn.pth"))
    except: pass
    return model

def get_engine(model):
    try:
        llm_client = GeminiClient()
        strategy_agent = StrategyAgent(llm_client)
    except: strategy_agent = None
    return DiscoveryEngine(model=model, strategy_agent=strategy_agent, csv_path="insumos.csv")

model = load_model()
if 'engine' not in st.session_state:
    st.session_state.engine = get_engine(model)
engine = st.session_state.engine

if 'current_formula' not in st.session_state: st.session_state.current_formula = None
if 'history' not in st.session_state: st.session_state.history = []
if 'round_count' not in st.session_state: st.session_state.round_count = 0

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown(f"""
        <div style="text-align: center; padding: 20px 0;">
            <h2 style="color:white; margin:0;">L'ORÉAL</h2>
            <p style="color:{LOREAL_GOLD}; font-size:9px; letter-spacing:4px;">LUXE R&D AI</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown(f"""<h1 style="color:{LOREAL_GOLD}; font-size:16px; letter-spacing:2px;">🧬 PARAMETERS</h1>""", unsafe_allow_html=True)
    all_ingredients = sorted(engine.insumos_dict.keys())
    anchors = st.multiselect("MANDATORY NOTES", options=all_ingredients)
    
    if anchors != engine.anchors:
        engine.anchors = anchors
        st.toast("Formula DNA Updated")

    st.markdown("<br>"*10, unsafe_allow_html=True)
    if st.button("RESET LABORATORY"):
        st.session_state.history = []
        st.session_state.round_count = 0
        st.session_state.current_formula = None
        st.rerun()

# =========================================================
# LÓGICA DE GERAÇÃO
# =========================================================
def generate_next():
    with st.spinner("SYNTHESIZING..."):
        discoveries = engine.discover(rounds=1)
        if discoveries:
            st.session_state.current_formula = discoveries[-1]
            st.session_state.round_count += 1

def submit_feedback():
    score = st.session_state.feedback_slider
    engine.register_human_feedback(-1, score)
    data = st.session_state.current_formula
    st.session_state.history.insert(0, {
        "GEN": f"#{st.session_state.round_count}", "SCORE": score, 
        "COMPLEXITY": f"{data['chemistry'].get('complexity', 0):.1f}", "NOTES": len(data['molecules'])
    })
    generate_next()

# =========================================================
# UI PRINCIPAL
# =========================================================

# --- CABEÇALHO ---
col_h1, col_h2 = st.columns([2, 1])
with col_h1:
    st.title("AI Perfumer")
    st.markdown("<p style='letter-spacing:2px; color:#666;'>MOLECULAR DISCOVERY & OLFACTORY STRATEGY</p>", unsafe_allow_html=True)

with col_h2:
    if st.session_state.current_formula:
        st.markdown(f"""
            <div style="text-align:right;">
                <span style="font-size:10px; color:{LOREAL_GOLD}; letter-spacing:3px;">BATCH ID</span>
                <h1 style="margin:0;">#{st.session_state.round_count:02d}</h1>
            </div>
        """, unsafe_allow_html=True)

st.markdown("<hr style='border-color:#EEE;'>", unsafe_allow_html=True)

# --- ESTADO VAZIO OU CONTEÚDO ---
if st.session_state.current_formula is None:
    st.markdown("<div style='height:200px;'></div>", unsafe_allow_html=True)
    col_btn, _ = st.columns([1, 2])
    if col_btn.button("START SYNTHESIS SEQUENCE", type="primary"):
        generate_next()
        st.rerun()
else:
    data = st.session_state.current_formula
    chem = data['chemistry']
    mols = data['molecules']
    biz_engine = PerfumeBusinessEngine()
    market = biz_engine.calculate_global_fit(mols)
    finances = biz_engine.estimate_financials(mols, chem.get('complexity',0), chem.get('neuro_score',0))

    # 1. KPIs SUPERIORES
    k1, k2, k3, k4 = st.columns(4)
    for col, lab, val in zip([k1,k2,k3,k4], 
                             ["Longevity", "Sillage", "Uniqueness", "Harmony"],
                             [f"{chem.get('longevity',0):.1f}h", f"{chem.get('projection',0):.1f}/10", f"{chem.get('complexity',0):.1f}/10", f"{chem.get('evolution',0):.1f}/10"]):
        col.markdown(f'<div class="kpi-card"><div class="kpi-lab">{lab}</div><div class="kpi-val">{val}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 2. COLUNA DUPLA: CIÊNCIA vs NEGÓCIO
    col_left, col_right = st.columns([1.8, 1], gap="large")

    with col_left:
        st.markdown("### ⚗️ Formula Analysis")
        df_mols = pd.DataFrame(mols)
        st.dataframe(df_mols[['name', 'category', 'weight_factor']], 
                     column_config={
                         "weight_factor": st.column_config.ProgressColumn("Concentration", format="%.2f", min_value=0, max_value=1),
                         "name": "Ingredient", "category": "Family"
                     }, use_container_width=True, hide_index=True)

        

        st.markdown("### 🧠 Neuro-Olfactive Impact")
        vectors = chem.get('neuro_vectors', {'energy': 0.1, 'calm': 0.1, 'mood': 0.1})
        n_df = pd.DataFrame(list(vectors.items()), columns=['Axis', 'Value'])
        n_chart = alt.Chart(n_df).mark_bar(color=LOREAL_GOLD).encode(
            x=alt.X('Value', scale=alt.Scale(domain=[0, 2])),
            y=alt.Y('Axis', sort='-x')
        ).properties(height=150)
        st.altair_chart(n_chart, use_container_width=True)

    with col_right:
        st.markdown("### 💼 Business Strategy")
        st.markdown(f"""
            <div class="biz-card">
                <p style="color:{LOREAL_GOLD}; font-size:10px; letter-spacing:2px; margin-bottom:5px;">MARKET POSITION</p>
                <h2 style="color:white; margin-bottom:20px;">{finances.get('market_tier', 'Prestige').upper()}</h2>
                <div style="display:flex; justify-content:space-between; margin-bottom:10px; border-bottom:1px solid #333; padding-bottom:5px;">
                    <span>Target Price</span><span style="color:{LOREAL_GOLD}; font-weight:bold;">${finances.get('price', 0):.2f}</span>
                </div>
                <div style="display:flex; justify-content:space-between; margin-bottom:10px; border-bottom:1px solid #333; padding-bottom:5px;">
                    <span>Est. Margin</span><span>{finances.get('margin_pct', 0)}%</span>
                </div>
                <div style="margin-top:20px;">
                    <p style="color:{LOREAL_GOLD}; font-size:10px; letter-spacing:2px; margin-bottom:5px;">PRIMARY MARKET</p>
                    <p style="font-size:18px;">{market.get('best', 'Global')}</p>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Alertas de Risco
        risks, _ = engine.chemistry._detect_chemical_risks(mols)
        if risks:
            for r in risks: st.error(f"STABILITY ALERT: {r}")
        else:
            st.caption("✨ FORMULA STABILITY CERTIFIED")

        # Feedback Loop
        st.markdown("<br>", unsafe_allow_html=True)
        st.slider("SENSORY QUALITY", 0.0, 10.0, 5.0, key="feedback_slider")
        st.button("EVOLVE FORMULA ➤", on_click=submit_feedback)

# --- FOOTER / HISTORY ---
if st.session_state.history:
    st.markdown("---")
    with st.expander("VIEW PREVIOUS ITERATIONS"):
        st.table(pd.DataFrame(st.session_state.history))