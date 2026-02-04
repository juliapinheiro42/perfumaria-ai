import streamlit as st
import pandas as pd
import time
import os
import torch
import numpy as np
import altair as alt
import plotly.graph_objects as go
import random

# Importações dos módulos internos
# Certifique-se de que os arquivos existem em core/ e infra/
try:
    from infra.gemini_client import GeminiClient
    from core.strategy import StrategyAgent
    from core.discovery import DiscoveryEngine
    from core.model import MoleculeGNN
    from core.market import PerfumeBusinessEngine
    from core.compliance import ComplianceEngine
except ImportError as e:
    st.error(f"Erro de importação: {e}. Verifique se a estrutura de pastas está correta.")
    st.stop()

# =========================================================
# CONFIGURAÇÃO VISUAL L'ORÉAL LUXE (DESIGN EDITORIAL)
# =========================================================
st.set_page_config(
    page_title="L'Oréal Luxe • AI Lab",
    page_icon="🧴",
    layout="wide",
    initial_sidebar_state="expanded"
)

LOREAL_BLACK = "#000000"
LOREAL_GOLD = "#C5A059" 
LOREAL_WHITE = "#FFFFFF"
LOREAL_GREY = "#F2F2F2"
LOREAL_GREEN = "#2D5A27" # Verde escuro elegante para o tema Green

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
        border-color: {LOREAL_GOLD};
    }}

    /* KPI Cards Robustos */
    .kpi-card {{
        background-color: {LOREAL_WHITE};
        padding: 20px;
        border: 1px solid #EEE;
        border-bottom: 3px solid {LOREAL_GOLD};
        text-align: center;
        height: 100%;
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
    
    /* Container Green Chemistry */
    .green-card {{
        background-color: #F9FDF9;
        border: 1px solid {LOREAL_GREEN};
        padding: 20px;
        margin-top: 20px;
    }}
</style>
""", unsafe_allow_html=True)

# =========================================================
# HELPER FUNCTIONS (GRÁFICOS)
# =========================================================
def render_family_evolution(family_data):
    """Streamgraph da evolução das famílias."""
    if not family_data: return None
    FAMILY_COLORS = {
        "Cítrico": "#FFD700", "Verde": "#8FBC8F", "Floral": "#FFB6C1", 
        "Floral Branco": "#FFFACD", "Especiado": "#CD853F", "Amadeirado": "#8B4513", 
        "Musk": "#E6E6FA", "Âmbar": "#DAA520", "Gourmand": "#D2691E", 
        "Aquático": "#ADD8E6", "Aldeídico": "#F0F8FF"
    }

    df = pd.DataFrame(family_data)
    chart = alt.Chart(df).mark_area(opacity=0.8).encode(
        x=alt.X('Time', title='Horas após aplicação', scale=alt.Scale(domain=[0, 10])),
        y=alt.Y('Intensity', stack='center', title='Dominância', axis=None),
        color=alt.Color('Family', scale=alt.Scale(domain=list(FAMILY_COLORS.keys()), range=list(FAMILY_COLORS.values())), legend=alt.Legend(title="Famílias", orient="bottom")),
        tooltip=['Time', 'Family', 'Intensity']
    ).properties(height=300, title="4D SCENT EVOLUTION").interactive()
    return chart

def render_olfactory_pyramid(formula):
    """Pirâmide olfativa visual."""
    if not formula: return None
    LUXE_PALETTE = {"Top": "rgba(197, 160, 89, 0.9)", "Heart": "rgba(212, 175, 55, 0.6)", "Base": "rgba(40, 40, 40, 0.85)"}
    categories = {"Top": [], "Heart": [], "Base": []}
    
    for mol in formula:
        cat = mol.get('category', 'Heart')
        if cat not in categories: cat = "Heart"
        categories[cat].append(mol)

    fig = go.Figure()
    fig.update_layout(
        shapes=[
            dict(type="path", path="M 0,100 L 50,0 L -50,0 Z", fillcolor="rgba(0,0,0,0)", line=dict(color="#E0E0E0", width=1)),
            dict(type="line", x0=-15, y0=70, x1=15, y1=70, line=dict(color="#EEE", width=1, dash="dot")),
            dict(type="line", x0=-35, y0=35, x1=35, y1=35, line=dict(color="#EEE", width=1, dash="dot")),
        ],
        xaxis=dict(visible=False, range=[-60, 60], fixedrange=True),
        yaxis=dict(visible=False, range=[-10, 110], fixedrange=True),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", height=400, showlegend=False, margin=dict(l=0, r=0, t=20, b=0)
    )

    y_ranges = {"Top": (75, 95), "Heart": (40, 65), "Base": (5, 30)}
    for cat_name, mols in categories.items():
        x_vals, y_vals, sizes, hover_texts, colors = [], [], [], [], []
        for m in mols:
            y = random.uniform(*y_ranges[cat_name])
            width_factor = (100 - y) / 100 * 45
            x_vals.append(random.uniform(-width_factor + 3, width_factor - 3))
            y_vals.append(y)
            w = m.get('weight_factor', 1.0)
            sizes.append(10 + (w * 15))
            colors.append(LUXE_PALETTE.get(cat_name))
            hover_texts.append(f"<b>{m.get('name')}</b><br>Fam: {m.get('olfactive_family', 'N/A')}<br>Conc: {w:.2f}")

        fig.add_trace(go.Scatter(
            x=x_vals, y=y_vals, mode='markers', marker=dict(size=sizes, color=colors, line=dict(width=1, color='white')),
            hoverinfo='text', hovertext=hover_texts
        ))

    return fig

def render_evaporation_curve(temporal_curve):
    """Gráfico de projeção."""
    if not temporal_curve: return None
    time_labels = ["0.1h", "0.5h", "1h", "3h", "6h", "10h"]
    df_curve = pd.DataFrame({"Time": time_labels, "Projection": temporal_curve, "Order": range(len(time_labels))})
    
    line = alt.Chart(df_curve).mark_area(
        line={'color': LOREAL_GOLD},
        color=alt.Gradient(gradient='linear', stops=[alt.GradientStop(color='white', offset=0), alt.GradientStop(color=LOREAL_GOLD, offset=1)], x1=1, x2=1, y1=1, y2=0),
        opacity=0.3
    ).encode(
        x=alt.X('Time', sort=alt.EncodingSortField(field="Order", order="ascending")),
        y=alt.Y('Projection', scale=alt.Scale(domain=[0, 10])),
        tooltip=['Time', 'Projection']
    ).properties(height=250)
    
    return line + line.mark_point(color=LOREAL_BLACK)

# =========================================================
# LÓGICA DE REFORMULAÇÃO (Nova Função Local)
# =========================================================
def run_green_reformulation(current_formula_dict):
    """
    Usa o ComplianceEngine para encontrar substitutos 'Green'.
    Substitui ingredientes não-verdes por alternativas biodegradáveis/renováveis.
    """
    mols = current_formula_dict['molecules']
    new_mols = []
    log = []
    
    for m in mols:
        # Verifica se o ingrediente já é Green (Bio e Renovável)
        is_bio = m.get('biodegradability') == True or str(m.get('biodegradability')).lower() == 'true'
        is_renew = m.get('renewable_source') == True or str(m.get('renewable_source')).lower() == 'true'
        
        if is_bio and is_renew:
            new_mols.append(m)
        else:
            # Tenta encontrar substituto
            subs = st.session_state.compliance_engine.find_substitutes(m['name'], top_n=1)
            if subs:
                best = subs[0]
                # Só troca se o score for decente (>30)
                if best['score'] > 30:
                    new_m = m.copy()
                    new_m['name'] = best['name']
                    # Atualiza flags para o novo ingrediente (assumindo que find_substitutes prioriza green)
                    new_m['biodegradability'] = True 
                    new_m['renewable_source'] = True
                    new_mols.append(new_m)
                    log.append(f"♻️ Swapped {m['name']} -> {best['name']} (Score: {best['score']:.1f})")
                else:
                    new_mols.append(m)
            else:
                new_mols.append(m)
                
    # Recalcula Eco-Score
    new_formula = current_formula_dict.copy()
    new_formula['molecules'] = new_mols
    eco_score, stats = st.session_state.compliance_engine.calculate_eco_score(new_mols)
    
    # Atualiza estruturas
    new_formula['eco_score'] = eco_score
    if 'chemistry' not in new_formula: new_formula['chemistry'] = {}
    new_formula['chemistry']['eco_stats'] = stats
    
    return new_formula, log

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

# Inicialização de Estado
if 'compliance_engine' not in st.session_state:
    st.session_state.compliance_engine = ComplianceEngine() # Carrega o CSV automaticamente

comp_engine = st.session_state.compliance_engine
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
    
    all_ingredients = sorted(engine.insumos_dict.keys()) if hasattr(engine, 'insumos_dict') else []
    anchors = st.multiselect("MANDATORY NOTES", options=all_ingredients)
    
    if hasattr(engine, 'anchors') and anchors != engine.anchors:
        engine.anchors = anchors
        st.toast("Formula DNA Updated")

    st.markdown("<br>", unsafe_allow_html=True)
    
    # --- CONTROLES DE REFORMULAÇÃO GREEN ---
    if st.session_state.current_formula:
        st.markdown(f"""<h1 style="color:{LOREAL_GREEN}; font-size:14px; letter-spacing:2px;">♻️ SUSTAINABILITY</h1>""", unsafe_allow_html=True)
        
        if st.button("REFORMULATE GREEN", help="Substitui ingredientes por alternativas bio"):
            with st.spinner("SEARCHING FOR BIO-ALTERNATIVES..."):
                # Executa a lógica local de reformulação
                new_formula, logs = run_green_reformulation(st.session_state.current_formula)
                
                if logs:
                    st.session_state.current_formula = new_formula
                    st.session_state.round_count += 1
                    
                    # Salva no histórico
                    st.session_state.history.insert(0, {
                        "GEN": f"#{st.session_state.round_count} (GREEN)",
                        "SCORE": "N/A",
                        "COMPLEXITY": "REF",
                        "NOTES": len(new_formula['molecules'])
                    })
                    
                    st.toast(f"Reformulated! {len(logs)} changes made.", icon="🌿")
                    st.rerun()
                else:
                    st.toast("Formula is already optimal or no substitutes found.", icon="✅")

    st.markdown("<br>"*5, unsafe_allow_html=True)
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
            new_f = discoveries[-1]
            # Calcula Eco Score na geração também
            eco, stats = comp_engine.calculate_eco_score(new_f['molecules'])
            new_f['eco_score'] = eco
            if 'chemistry' not in new_f: new_f['chemistry'] = {}
            new_f['chemistry']['eco_stats'] = stats
            
            st.session_state.current_formula = new_f
            st.session_state.round_count += 1

# =========================================================
# UI PRINCIPAL
# =========================================================

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
    chem = data.get('chemistry', {})
    mols = data['molecules']
    
    # Simulação de Business (se disponível)
    biz_engine = PerfumeBusinessEngine()
    market = biz_engine.calculate_global_fit(mols)
    finances = biz_engine.estimate_financials(mols, chem.get('complexity',0), chem.get('neuro_score',0))
    
    # Recupera Eco-Stats e Score
    eco_score = data.get('eco_score', 0.0)
    eco_stats = chem.get('eco_stats', {"biodegradable_pct": 0, "renewable_pct": 0, "avg_carbon_footprint": 10})

    # KPIs
    k1, k2, k3, k4, k5 = st.columns(5)
    metrics = [
        ("Longevity", f"{chem.get('longevity',0):.1f}h"),
        ("Sillage", f"{chem.get('projection',0):.1f}/10"),
        ("Uniqueness", f"{chem.get('complexity',0):.1f}/10"),
        ("Harmony", f"{chem.get('evolution',0):.1f}/10"),
        ("Eco-Score", f"{eco_score:.2f}")
    ]
    
    cols = [k1, k2, k3, k4, k5]
    for col, (lab, val) in zip(cols, metrics):
        color = LOREAL_GREEN if lab == "Eco-Score" and float(val) > 0.7 else LOREAL_BLACK
        col.markdown(f'<div class="kpi-card"><div class="kpi-lab">{lab}</div><div class="kpi-val" style="color:{color}">{val}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_left, col_right = st.columns([1.8, 1], gap="large")

    with col_left:
        st.markdown("### ⚗️ Formula Analysis")
        df_mols = pd.DataFrame(mols)
        
        # Garante colunas booleanas
        if 'biodegradability' not in df_mols.columns: df_mols['biodegradability'] = False
        if 'renewable_source' not in df_mols.columns: df_mols['renewable_source'] = False
        
        st.dataframe(
            df_mols[['name', 'category', 'weight_factor', 'biodegradability', 'renewable_source']], 
            column_config={
                "weight_factor": st.column_config.ProgressColumn("Conc.", format="%.2f", min_value=0, max_value=5),
                "name": "Ingredient", 
                "category": "Family",
                "biodegradability": st.column_config.CheckboxColumn("Bio?", width="small"),
                "renewable_source": st.column_config.CheckboxColumn("Renew?", width="small")
            }, 
            use_container_width=True, 
            hide_index=True
        )

        st.markdown("### 👃 Olfactory Structure")
        col_graph, col_data = st.columns([2, 1])
        with col_graph:
            fig = render_olfactory_pyramid(mols)
            if fig: st.plotly_chart(fig, use_container_width=True)
            
        st.markdown("#### ⏳ 4D Temporal Simulation")
        tab1, tab2 = st.tabs(["📉 Intensity Decay", "🌊 Family Dominance"])
        
        with tab1:
            curve_data = chem.get('temporal_curve')
            if curve_data:
                chart = render_evaporation_curve(curve_data)
                st.altair_chart(chart, use_container_width=True)
            
        with tab2:
            fam_data = chem.get('temporal_families')
            if fam_data:
                fam_chart = render_family_evolution(fam_data)
                st.altair_chart(fam_chart, use_container_width=True)
            else:
                st.caption("No evolution data available.")

    with col_right:
        st.markdown("### 💼 Business Strategy")
        st.markdown(f"""
            <div class="biz-card">
                <p style="color:{LOREAL_GOLD}; font-size:10px; letter-spacing:2px; margin-bottom:5px;">MARKET POSITION</p>
                <h2 style="color:white; margin-bottom:20px;">{finances.get('market_tier', 'Prestige').upper()}</h2>
                <div style="display:flex; justify-content:space-between; margin-bottom:10px; border-bottom:1px solid #333; padding-bottom:5px;">
                    <span>Target Price</span><span style="color:{LOREAL_GOLD}; font-weight:bold;">${finances.get('price', 0):.2f}</span>
                </div>
                <div style="margin-top:20px;">
                    <p style="color:{LOREAL_GOLD}; font-size:10px; letter-spacing:2px; margin-bottom:5px;">PRIMARY MARKET</p>
                    <p style="font-size:18px;">{market.get('best', 'Global')}</p>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # --- PAINEL GREEN CHEMISTRY ---
        st.markdown(f"""
            <div class="green-card">
                <h3 style="color:{LOREAL_GREEN}; font-size:16px; margin-top:0;">🌿 Green Chemistry</h3>
                <div style="display:flex; justify-content:space-between; margin-bottom:5px;">
                    <span>Biodegradable</span><b>{eco_stats.get('biodegradable_pct', 0)}%</b>
                </div>
                <div style="display:flex; justify-content:space-between; margin-bottom:5px;">
                    <span>Renewable</span><b>{eco_stats.get('renewable_pct', 0)}%</b>
                </div>
                <div style="display:flex; justify-content:space-between; margin-bottom:5px;">
                    <span>Carbon Footprint</span><b>{eco_stats.get('avg_carbon_footprint', 10):.1f}</b>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # --- COMPLIANCE CHECK ---
        # Chamada corrigida ao novo ComplianceEngine
        is_safe, report, stats = comp_engine.check_safety(mols)
    
        if not is_safe:
            st.error("⚠️ IFRA COMPLIANCE VIOLATION")
            for r in report:
                st.warning(r)
        else:
            st.success("✨ IFRA COMPLIANT")

        st.markdown("---")
        st.markdown("### 🧠 Sensory Training")
        
        c1, c2, c3 = st.columns(3)
        with c1: f_hedonic = st.slider("💖 Hedonic", 0, 10, 5)
        with c2: f_tech = st.slider("🛠️ Technical", 0, 10, 5)
        with c3: f_creative = st.slider("🎨 Creative", 0, 10, 5)

        if st.button("🧬 TRAIN NEURAL NETWORK", type="primary", use_container_width=True):
            if st.session_state.current_formula:
                current_data = st.session_state.current_formula
                feedback_vector = {"hedonic": f_hedonic, "technical": f_tech, "creative": f_creative}
                
                # 1. Registra feedback humano real
                engine.register_human_feedback(current_data.get("id"), feedback_vector)
        
                # 2. Gera exemplos negativos artificiais (Adversarial Training)
                # Tenta corromper a fórmula com ingredientes proibidos para ensinar a IA o que "NÃO FAZER"
                restricted_chems = list(comp_engine.IFRA_LIMITS.keys())
                corrupted_mols = [m.copy() for m in current_data['molecules']]
                
                # Adiciona intencionalmente uma overdose de um químico restrito
                target_bad_chem = random.choice(restricted_chems)
                corrupted_mols.append({'name': target_bad_chem, 'weight_factor': 10.0, 'category': 'Base'}) # Overdose massiva

                # Registra como feedback negativo (Score baixo)
                engine.register_human_feedback(
                    None,
                    {"hedonic": 0.1, "technical": 0.0, "creative": 0.0}, 
                    custom_mols=corrupted_mols 
                )

                # Executa passo de treino
                if hasattr(engine, 'trainer') and engine.trainer:
                    engine.trainer.train_step(engine.buffer)
                    st.success("Neuro-weights updated with Adversarial Examples!")
                else:
                    st.warning("Trainer module not active.")

                time.sleep(0.5)
                generate_next()
                st.rerun()

if st.session_state.history:
    st.markdown("---")
    with st.expander("VIEW PREVIOUS ITERATIONS"):
        st.table(pd.DataFrame(st.session_state.history))