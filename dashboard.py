import streamlit as st
import pandas as pd
import time
import os
import torch
import numpy as np
import altair as alt # Adicionado para gráficos elegantes

# Ajuste os imports conforme a estrutura da sua pasta
from infra.gemini_client import GeminiClient
from core.strategy import StrategyAgent
from core.discovery import DiscoveryEngine
from core.model import MoleculeGNN
from core.market import PerfumeBusinessEngine

# =========================================================
# CONFIGURAÇÃO VISUAL L'ORÉAL (CSS INJECT)
# =========================================================
st.set_page_config(
    page_title="L'Oréal Luxe • AI Lab",
    page_icon="🧴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paleta de Cores L'Oréal Luxe
LOREAL_BLACK = "#000000"
LOREAL_GOLD = "#C5A059" 
LOREAL_WHITE = "#FFFFFF"
LOREAL_GREY = "#F9F9F9"
LOREAL_DARK_GREY = "#333333"

st.markdown(f"""
<style>
    /* RESET GERAL E FONTE */
    .stApp {{
        background-color: {LOREAL_WHITE};
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }}
    
    /* REMOVER PADDING PADRÃO DO STREAMLIT PARA CABEÇALHO LIMPO */
    .block-container {{
        padding-top: 2rem;
        padding-bottom: 5rem;
    }}

    /* HEADERS */
    h1 {{
        font-weight: 200 !important;
        text-transform: uppercase;
        letter-spacing: 4px;
        font-size: 2.5rem;
        margin-bottom: 0px;
        color: {LOREAL_BLACK};
    }}
    h3 {{
        font-weight: 400 !important;
        text-transform: uppercase;
        letter-spacing: 2px;
        font-size: 1.1rem;
        color: {LOREAL_BLACK};
        border-bottom: 2px solid {LOREAL_GOLD};
        padding-bottom: 10px;
        margin-top: 30px;
    }}
    
    /* SIDEBAR */
    [data-testid="stSidebar"] {{
        background-color: {LOREAL_BLACK};
        border-right: 1px solid #222;
    }}
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {{
        color: {LOREAL_WHITE} !important;
        border-color: {LOREAL_GOLD};
    }}
    
    /* CARDS DE KPI (Minimalistas) */
    .kpi-container {{
        background-color: {LOREAL_WHITE};
        border-left: 3px solid {LOREAL_GOLD};
        padding: 15px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.03);
        transition: transform 0.2s;
    }}
    .kpi-container:hover {{
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.08);
    }}
    .kpi-label {{
        font-size: 0.75rem;
        color: #888;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-bottom: 5px;
    }}
    .kpi-value {{
        font-size: 1.8rem;
        color: {LOREAL_BLACK};
        font-weight: 300;
    }}
    .kpi-sub {{
        font-size: 0.7rem;
        color: {LOREAL_GOLD};
        margin-top: 2px;
        font-weight: 600;
    }}

    /* BOTÕES (Estilo Editorial) */
    .stButton > button {{
        background-color: transparent;
        color: {LOREAL_BLACK} !important;
        border: 1px solid {LOREAL_BLACK};
        border-radius: 0px;
        text-transform: uppercase;
        font-size: 12px;
        letter-spacing: 2px;
        padding: 15px 25px;
        transition: all 0.4s ease;
        font-weight: 600;
    }}
    .stButton > button:hover {{
        background-color: {LOREAL_BLACK};
        color: {LOREAL_GOLD} !important;
        border-color: {LOREAL_BLACK};
    }}
    /* Botão Primário (Confirm) */
    div[data-testid="stVerticalBlock"] > div > div > div > div > button:focus {{
        border-color: {LOREAL_GOLD};
        color: {LOREAL_GOLD};
    }}

    /* TABELAS */
    [data-testid="stDataFrame"] {{
        border: 1px solid #eee;
    }}
    
    /* CUSTOM ALERTS */
    .stToast {{
        background-color: {LOREAL_BLACK} !important;
        color: {LOREAL_WHITE} !important;
    }}
</style>
""", unsafe_allow_html=True)

# =========================================================
# 1. CACHE E SISTEMA
# =========================================================
@st.cache_resource
def load_engine():
    model = MoleculeGNN(num_node_features=5)
    try:
        if os.path.exists("results/perfume_gnn.pth"):
            model.load_state_dict(torch.load("results/perfume_gnn.pth"))
    except Exception as e:
        print(f"⚠️ Init Warning: {e}")

    try:
        llm_client = GeminiClient()
        strategy_agent = StrategyAgent(llm_client)
    except:
        strategy_agent = None
        
    engine = DiscoveryEngine(
        model=model,
        strategy_agent=strategy_agent,
        csv_path="insumos.csv"
    )
    return engine

try:
    engine = load_engine()
except Exception as e:
    st.error(f"System Failure: {e}")
    st.stop()

# =========================================================
# 2. STATE
# =========================================================
if 'current_formula' not in st.session_state:
    st.session_state.current_formula = None
if 'history' not in st.session_state:
    st.session_state.history = []
if 'round_count' not in st.session_state:
    st.session_state.round_count = 0

# =========================================================
# 3. SIDEBAR
# =========================================================
with st.sidebar:
    # --- LOGO TIPOGRÁFICO (Nunca quebra) ---
    st.markdown(f"""
        <div style="text-align: center; margin-bottom: 20px;">
            <h1 style="color:{LOREAL_WHITE}; font-size:32px; letter-spacing:4px; margin:0; padding:0;">L'ORÉAL</h1>
            <span style="color:{LOREAL_GOLD}; font-size:10px; letter-spacing:5px; text-transform:uppercase;">Paris • Luxe</span>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"<div style='font-size:10px; letter-spacing:3px; color:#666; margin-bottom:30px; text-align:center;'>AI R&D DIVISION</div>", unsafe_allow_html=True)
    
    st.markdown("### 🧬 FORMULA DNA")
    
    all_ingredients = sorted(engine.insumos_dict.keys())
    anchors = st.multiselect(
        "MANDATORY INGREDIENTS", 
        options=all_ingredients,
        help="Locks specific molecules in the generation process."
    )
    
    if anchors != engine.anchors:
        engine.anchors = anchors
        st.toast("Constraints Updated", icon="⚓")

    st.markdown("<br><br>", unsafe_allow_html=True)
    if st.button("🗑️ RESET LAB BENCH"):
        st.session_state.history = []
        st.session_state.round_count = 0
        st.session_state.current_formula = None
        st.rerun()

# =========================================================
# 4. LÓGICA
# =========================================================
def generate_next():
    with st.spinner("⚗️ SYNTHESIZING MOLECULAR STRUCTURE..."):
        discoveries = engine.discover(rounds=1)
        if discoveries:
            st.session_state.current_formula = discoveries[-1]
            st.session_state.round_count += 1
        else:
            st.error("Convergence Failure. Adjust constraints.")

def submit_feedback():
    score = st.session_state.feedback_slider
    if st.session_state.current_formula:
        engine.register_human_feedback(-1, score)
        data = st.session_state.current_formula
        st.session_state.history.insert(0, {
            "GEN": f"#{st.session_state.round_count:02d}",
            "SCORE": score,
            "COMPLEXITY": f"{data['chemistry'].get('complexity', 0):.1f}",
            "NOTES": f"{len(data['molecules'])}"
        })
        st.toast(f"Learning Applied: Score {score}", icon="🧠")
        generate_next()

# =========================================================
# 5. UI PRINCIPAL
# =========================================================

# --- HERO SECTION ---
c_title, c_gen = st.columns([3, 1], gap="large")
with c_title:
    st.title("AI PERFUMER")
    st.markdown(f"<div style='color:#666; letter-spacing:1px;'>COMPUTATIONAL OLFACTORY DESIGN // WORKBENCH</div>", unsafe_allow_html=True)

with c_gen:
    if st.session_state.current_formula:
        st.markdown(f"""
        <div style="text-align:right; border-right: 4px solid {LOREAL_GOLD}; padding-right:15px; margin-top:10px;">
            <div style="font-size:10px; color:#999; letter-spacing:2px;">GENERATION</div>
            <div style="font-size:32px; font-weight:100;">#{st.session_state.round_count:02d}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# --- CONTEÚDO PRINCIPAL ---
if st.session_state.current_formula is None:
    # Estado Vazio Elegante
    st.markdown(f"""
    <div style='display:flex; justify-content:center; align-items:center; height:300px; flex-direction:column; color:#999;'>
        <div style='font-size:40px; margin-bottom:10px;'>🧪</div>
        <div style='text-transform:uppercase; letter-spacing:2px;'>System Ready for Synthesis</div>
    </div>
    """, unsafe_allow_html=True)
    
    col_center = st.columns([1,1,1])
    with col_center[1]:
        if st.button("🚀 INITIALIZE SEQUENCE", type="primary", use_container_width=True):
            generate_next()
            st.rerun()

else:
    data = st.session_state.current_formula
    mols = data['molecules']
    chem = data['chemistry']
    
    biz_engine = PerfumeBusinessEngine()
    tech_score = chem.get('complexity', 0.0)
    neuro_score = chem.get('neuro_score', 0.0)
    market_analysis = biz_engine.calculate_global_fit(mols)
    financials = biz_engine.estimate_financials(mols, tech_score, neuro_score)

    # 1. CARDS DE KPI (HTML/CSS Customizado)
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    
    def render_kpi(col, label, value, sub):
        col.markdown(f"""
        <div class="kpi-container">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-sub">{sub}</div>
        </div>
        """, unsafe_allow_html=True)

    render_kpi(kpi1, "LONGEVITY", f"{chem.get('longevity',0):.1f}h", "SKIN TEST")
    render_kpi(kpi2, "SILLAGE", f"{chem.get('projection',0):.1f}<span style='font-size:12px'>/10</span>", "PROJECTION")
    render_kpi(kpi3, "UNIQUENESS", f"{tech_score:.1f}<span style='font-size:12px'>/10</span>", "ANTI-DUPE")
    render_kpi(kpi4, "HARMONY", f"{chem.get('evolution',0):.1f}<span style='font-size:12px'>/10</span>", "OLFACTIVE SCORE")

    st.markdown("<br>", unsafe_allow_html=True)

    # 2. COLUNAS DE DETALHE
    col_left, col_right = st.columns([2, 1], gap="large")

    with col_left:
        st.markdown("### OLFACTIVE PYRAMID & COMPOSITION")
        
        # Preparar dados para o DataFrame
        df_display = pd.DataFrame(mols)
        total_w = sum([m.get('weight_factor', 1) for m in mols])
        df_display['formula_pct'] = [m.get('weight_factor', 1)/total_w for m in mols]
        df_display['Tier'] = df_display['name'].apply(lambda x: engine.insumos_dict.get(x, {}).get('complexity_tier', 1))
        
        # Tabela Limpa
        st.dataframe(
            df_display[['name', 'category', 'formula_pct', 'Tier']],
            column_config={
                "name": st.column_config.TextColumn("INGREDIENT", width="medium"),
                "category": st.column_config.TextColumn("FAMILY", width="small"),
                "formula_pct": st.column_config.ProgressColumn("CONC.", format="%.1f%%", min_value=0, max_value=0.5),
                "Tier": st.column_config.NumberColumn("TIER", format="%d 💎")
            },
            hide_index=True,
            use_container_width=True,
            height=250
        )
        
        st.markdown("### MARKET RESONANCE")
        # Visualização Altair no lugar de HTML cru
        rankings = market_analysis.get('rankings', {})
        df_rank = pd.DataFrame(list(rankings.items()), columns=['Region', 'Score'])
        
        chart = alt.Chart(df_rank).mark_bar(color=LOREAL_BLACK).encode(
            x=alt.X('Score', scale=alt.Scale(domain=[0, 10])),
            y=alt.Y('Region', sort='-x', title=None),
            tooltip=['Region', 'Score']
        ).properties(height=200).configure_axis(
            grid=False, domain=False
        ).configure_view(strokeWidth=0)
        
        st.altair_chart(chart, use_container_width=True)

    with col_right:
        # BUSINESS CARD PREMIUM
        st.markdown("### PRODUCT STRATEGY")
        
        tier = financials.get('market_tier', 'Luxury').upper()
        price = financials.get('price', 0)
        cost = financials.get('cost', 0)
        margin = financials.get('margin_pct', 0)
        target_mkt = market_analysis.get('best', 'Global')
        label_mkt = market_analysis.get('label', '')
        
        # ATENÇÃO: O HTML abaixo deve ficar colado na esquerda (sem indentação)
        card_html = f"""
<div style="background-color:{LOREAL_BLACK}; color:white; padding:20px; border-top:4px solid {LOREAL_GOLD}; border-radius: 0px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
<div style="font-size:10px; letter-spacing:2px; color:#aaa; margin-bottom:5px;">POSITIONING</div>
<div style="font-size:24px; font-weight:bold; margin-bottom:20px; letter-spacing:1px; color:white;">{tier}</div>
<div style="border-top:1px solid #444; padding-top:15px; margin-top:10px;">
<div style="display:flex; justify-content:space-between; margin-bottom:8px;">
<span style="font-size:12px; color:#ccc;">RETAIL TARGET</span>
<span style="font-size:16px; font-weight:bold; color:{LOREAL_GOLD};">${price:.2f}</span>
</div>
<div style="display:flex; justify-content:space-between; margin-bottom:8px;">
<span style="font-size:12px; color:#ccc;">COG ESTIMATE</span>
<span style="font-size:14px; color:white;">${cost:.2f}</span>
</div>
<div style="display:flex; justify-content:space-between;">
<span style="font-size:12px; color:#ccc;">GROSS MARGIN</span>
<span style="font-size:14px; color:white;">{margin}%</span>
</div>
</div>
</div>
"""
        # Renderiza o HTML
        st.markdown(card_html, unsafe_allow_html=True)
        
        # TARGET MARKET (Estilo Box Clean)
        target_html = f"""<div style="margin-top: 15px; padding: 15px; background-color: {LOREAL_GREY}; border-left: 3px solid {LOREAL_BLACK};"><div style="font-size: 14px; font-weight: bold; color: {LOREAL_BLACK};">🎯 Target Market: {target_mkt}</div><div style="font-size: 12px; color: #666; margin-top: 5px;">{label_mkt}</div></div>"""
        st.markdown(target_html, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # ÁREA DE FEEDBACK
        with st.container():
            st.markdown(f"""
            <div style='background-color:{LOREAL_GREY}; padding:15px; border:1px solid #eee;'>
                <div style='text-align:center; font-size:11px; letter-spacing:1px; margin-bottom:10px; color:{LOREAL_BLACK};'>HUMAN-IN-THE-LOOP EVALUATION</div>
            </div>
            """, unsafe_allow_html=True)
            
            # O slider precisa ficar fora do HTML puro para funcionar a interatividade
            st.slider("Qualidade", 0.0, 10.0, 5.0, 0.1, key="feedback_slider", label_visibility="collapsed")
            
            st.button("APPROVE & EVOLVE ➤", on_click=submit_feedback, use_container_width=True)

# --- RODAPÉ COM HISTÓRICO ---
if st.session_state.history:
    st.markdown("<br><br>", unsafe_allow_html=True)
    with st.expander("📜 VIEW GENERATION LINEAGE", expanded=False):
        st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True)