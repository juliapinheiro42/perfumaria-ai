import streamlit as st
import pandas as pd
import time
import os
import torch
import numpy as np
import altair as alt
import plotly.graph_objects as go
import random

from infra.gemini_client import GeminiClient
from core.strategy import StrategyAgent
from core.discovery import DiscoveryEngine
from core.model import MoleculeGNN
from core.market import PerfumeBusinessEngine
from core.compliance import ComplianceEngine

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

def render_family_evolution(family_data):
    """
    Renderiza um gráfico de área empilhada (Streamgraph) mostrando a evolução das famílias.
    """
    if not family_data: return None
    FAMILY_COLORS = {
        "Cítrico": "#FFD700",       # Ouro Amarelo
        "Verde": "#8FBC8F",         # Verde Sálvia
        "Floral": "#FFB6C1",        # Rosa Pálido
        "Floral Branco": "#FFFACD", # Creme
        "Especiado": "#CD853F",     # Bronze
        "Amadeirado": "#8B4513",    # Marrom Couro
        "Musk": "#E6E6FA",          # Lavanda Pálido
        "Âmbar": "#DAA520",         # Ouro Velho
        "Gourmand": "#D2691E",      # Chocolate
        "Aquático": "#ADD8E6",      # Azul Claro
        "Aldeídico": "#F0F8FF"      # Alice Blue
    }

    df = pd.DataFrame(family_data)
    chart = alt.Chart(df).mark_area(opacity=0.8).encode(
        x=alt.X('Time', title='Horas após aplicação (Evolução Temporal)', 
                scale=alt.Scale(domain=[0, 10])),
        y=alt.Y('Intensity', stack='center', title='Dominância Olfativa (Relativa)', axis=None),
        color=alt.Color('Family', scale=alt.Scale(
            domain=list(FAMILY_COLORS.keys()),
            range=list(FAMILY_COLORS.values())
        ), legend=alt.Legend(title="Famílias", orient="bottom")),
        tooltip=['Time', 'Family', 'Intensity']
    ).properties(
        height=300,
        title="4D SCENT EVOLUTION • DOMINANCE OVER TIME"
    ).interactive()
    
    return chart

def render_olfactory_pyramid(formula):
    """
    Renderiza uma pirâmide olfativa 'High-End'.
    """
    if not formula: return None

    LUXE_PALETTE = {
        "Top":   "rgba(197, 160, 89, 0.9)",
        "Heart": "rgba(212, 175, 55, 0.6)",
        "Base":  "rgba(40, 40, 40, 0.85)"
    }

    categories = {"Top": [], "Heart": [], "Base": []}
    for mol in formula:
        cat = mol.get('category', 'Heart')
        if cat not in categories: cat = "Heart"
        categories[cat].append(mol)

    fig = go.Figure()

    fig.update_layout(
        shapes=[
            dict(type="path", path="M 0,100 L 50,0 L -50,0 Z", 
                 fillcolor="rgba(0,0,0,0)",
                 line=dict(color="#E0E0E0", width=1, dash="solid")),
            dict(type="line", x0=-15, y0=70, x1=15, y1=70, line=dict(color="#EEE", width=1, dash="dot")),
            dict(type="line", x0=-35, y0=35, x1=35, y1=35, line=dict(color="#EEE", width=1, dash="dot")),
        ],
        xaxis=dict(visible=False, range=[-60, 60], fixedrange=True),
        yaxis=dict(visible=False, range=[-10, 110], fixedrange=True),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=550,
        showlegend=False,
        margin=dict(l=0, r=0, t=20, b=0)
    )

    y_ranges = {"Top": (75, 95), "Heart": (40, 65), "Base": (5, 30)}
    
    for cat_name, mols in categories.items():
        x_vals, y_vals, sizes, hover_texts, colors = [], [], [], [], []
        
        for m in mols:
            y = random.uniform(*y_ranges[cat_name])
            width_factor = (100 - y) / 100 * 45 
            x = random.uniform(-width_factor + 3, width_factor - 3)
            
            x_vals.append(x)
            y_vals.append(y)
            
            w = m.get('weight_factor', 1.0)
            sizes.append(10 + (w * 15))
            
            colors.append(LUXE_PALETTE.get(cat_name))
            
            hover_texts.append(
                f"<span style='font-size:14px; color:#C5A059'><b>{m.get('name')}</b></span><br>" +
                f"<span style='color:gray'>Fam: {m.get('olfactive_family', 'N/A')}</span><br>" +
                f"<i>Conc: {w:.2f}</i>"
            )

        fig.add_trace(go.Scatter(
            x=x_vals, y=y_vals,
            mode='markers',
            marker=dict(
                size=sizes,
                color=colors,
                line=dict(width=1, color='white'),
                opacity=1.0
            ),
            hoverinfo='text',
            hovertext=hover_texts,
            hoverlabel=dict(bgcolor="white", font_size=12, font_family="Inter")
        ))

    for y_pos, label in zip([85, 50, 15], ["TOP NOTES", "HEART NOTES", "BASE NOTES"]):
        fig.add_annotation(
            x=55, y=y_pos, text=label, showarrow=False,
            font=dict(family="Inter", size=10, color="#999"), xanchor="left"
        )

    return fig

def render_evaporation_curve(temporal_curve):
    """
    Renderiza o gráfico de projeção vs tempo (Dry-down).
    """
    if not temporal_curve: return None
    
    time_labels = ["0.1h", "0.5h", "1h", "3h", "6h", "10h"]
    
    # Criando o DataFrame para o Altair
    df_curve = pd.DataFrame({
        "Time": time_labels,
        "Projection": temporal_curve,
        "Order": range(len(time_labels))
    })

    # Gráfico de Linha com Área (Estilo Luxo)
    line = alt.Chart(df_curve).mark_area(
        line={'color': LOREAL_GOLD},
        color=alt.Gradient(
            gradient='linear',
            stops=[alt.GradientStop(color='white', offset=0),
                   alt.GradientStop(color=LOREAL_GOLD, offset=1)],
            x1=1, x2=1, y1=1, y2=0
        ),
        opacity=0.3,
        interpolate='monotone'
    ).encode(
        x=alt.X('Time', sort=alt.EncodingSortField(field="Order", order="ascending"), title="Tempo após aplicação"),
        y=alt.Y('Projection', title="Intensidade da Projeção", scale=alt.Scale(domain=[0, 10])),
        tooltip=['Time', 'Projection']
    ).properties(height=250)

    points = line.mark_point(color=LOREAL_BLACK, size=50).encode(
        opacity=alt.value(1)
    )

    return line + points

def get_engine(model):
    try:
        llm_client = GeminiClient()
        strategy_agent = StrategyAgent(llm_client)
    except: strategy_agent = None
    # Garante que o engine carregue os dados novos
    return DiscoveryEngine(model=model, strategy_agent=strategy_agent, csv_path="insumos.csv")

model = load_model()
if 'engine' not in st.session_state:
    st.session_state.engine = get_engine(model)
engine = st.session_state.engine

if 'current_formula' not in st.session_state: st.session_state.current_formula = None
if 'history' not in st.session_state: st.session_state.history = []
if 'round_count' not in st.session_state: st.session_state.round_count = 0

if 'compliance_engine' not in st.session_state:
    st.session_state.compliance_engine = ComplianceEngine()
comp_engine = st.session_state.compliance_engine

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

    st.markdown("<br>", unsafe_allow_html=True)
    
    # --- NOVO: CONTROLES DE REFORMULAÇÃO GREEN ---
    if st.session_state.current_formula:
        st.markdown(f"""<h1 style="color:{LOREAL_GREEN}; font-size:14px; letter-spacing:2px;">♻️ SUSTAINABILITY</h1>""", unsafe_allow_html=True)
        if st.button("REFORMULATE GREEN", help="Substitui ingredientes mantendo o cheiro alvo"):
            with st.spinner("SEARCHING FOR BIO-ALTERNATIVES..."):
                current_mols = st.session_state.current_formula['molecules']
                # Chama o novo método "Replacer"
                results = engine.reformulate_green(current_mols, rounds=30)
                
                if results:
                    best_green = max(results, key=lambda x: x['fitness'])
                    st.session_state.current_formula = best_green
                    st.session_state.round_count += 1
                    st.toast("Formula Reformulated for Eco-Compliance!", icon="🌿")
                    st.rerun()
                else:
                    st.error("Could not improve Eco-Score.")

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
    
    # Recupera Eco-Stats e Score
    eco_score = data.get('eco_score', 0.0)
    eco_stats = chem.get('eco_stats', {"biodegradable_pct": 0, "renewable_pct": 0, "avg_carbon_footprint": 10})

    # [ATUALIZADO] 5 KPIs (incluindo Eco-Score)
    k1, k2, k3, k4, k5 = st.columns(5)
    
    # Formata KPIs
    metrics = [
        ("Longevity", f"{chem.get('longevity',0):.1f}h"),
        ("Sillage", f"{chem.get('projection',0):.1f}/10"),
        ("Uniqueness", f"{chem.get('complexity',0):.1f}/10"),
        ("Harmony", f"{chem.get('evolution',0):.1f}/10"),
        ("Eco-Score", f"{eco_score:.2f}") # Novo KPI
    ]
    
    cols = [k1, k2, k3, k4, k5]
    for col, (lab, val) in zip(cols, metrics):
        # Destaca o Eco-Score em Verde se for alto
        color = LOREAL_GREEN if lab == "Eco-Score" and float(val) > 0.7 else LOREAL_BLACK
        col.markdown(f'<div class="kpi-card"><div class="kpi-lab">{lab}</div><div class="kpi-val" style="color:{color}">{val}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_left, col_right = st.columns([1.8, 1], gap="large")

    with col_left:
        st.markdown("### ⚗️ Formula Analysis")
        df_mols = pd.DataFrame(mols)
        
        if 'biodegradability' not in df_mols.columns: df_mols['biodegradability'] = False
        if 'renewable_source' not in df_mols.columns: df_mols['renewable_source'] = False
        
        st.dataframe(df_mols[['name', 'category', 'weight_factor', 'biodegradability', 'renewable_source']], 
                     column_config={
                         "weight_factor": st.column_config.ProgressColumn("Conc.", format="%.2f", min_value=0, max_value=5),
                         "name": "Ingredient", 
                         "category": "Family",
                         "biodegradability": st.column_config.CheckboxColumn("Bio?", width="small"),
                         "renewable_source": st.column_config.CheckboxColumn("Renew?", width="small")
                     }, use_container_width=True, hide_index=True)

        if st.session_state.current_formula:
            formula = st.session_state.current_formula['molecules']
    
    st.markdown("### 👃 Olfactory Structure")
    
    col_graph, col_data = st.columns([2, 1])
    
    with col_graph:
        fig = render_olfactory_pyramid(formula)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
            
    st.markdown("#### ⏳ 4D Temporal Simulation")
    
    tab1, tab2 = st.tabs(["📉 Intensity Decay", "🌊 Family Dominance"])
    
    with tab1:
        # Gráfico antigo de linha
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
            st.caption("Gere uma nova fórmula para visualizar a evolução por famílias.")

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
        
        # --- NOVO: PAINEL GREEN CHEMISTRY ---
        st.markdown(f"""
            <div class="green-card">
                <h3 style="color:{LOREAL_GREEN}; font-size:16px; margin-top:0;">🌿 Green Chemistry</h3>
                <div style="display:flex; justify-content:space-between; margin-bottom:5px;">
                    <span>Biodegradable</span>
                    <b>{eco_stats.get('biodegradable_pct', 0)}%</b>
                </div>
                <div style="display:flex; justify-content:space-between; margin-bottom:5px;">
                    <span>Renewable Source</span>
                    <b>{eco_stats.get('renewable_pct', 0)}%</b>
                </div>
                <div style="display:flex; justify-content:space-between; margin-bottom:5px;">
                    <span>Carbon Footprint</span>
                    <b>{eco_stats.get('avg_carbon_footprint', 10):.1f}</b> <span style="font-size:10px">(Low=Good)</span>
                </div>
            </div>
        """, unsafe_allow_html=True)
        # -------------------------------------
        
        is_safe, report, stats = comp_engine.check_safety(mols)
    
        if not is_safe:
            st.error("⚠️ IFRA COMPLIANCE VIOLATION")
            for r in report:
                st.warning(r)
            fixes = comp_engine.suggest_fix(report, mols)
            for f in fixes:
                st.info(f"💡 {f}")
        else:
            st.success("✨ IFRA COMPLIANT")
            

        st.markdown("---")
        st.markdown("### 🧠 Sensory Training Loop")
        st.caption("Ajude a IA a refinar seus critérios:")

        c1, c2, c3 = st.columns(3)
        
        with c1:
            f_hedonic = st.slider("💖 Hedonic", 0, 10, 5, help="O quanto você gostou do cheiro? (Subjetivo)")
        with c2:
            f_tech = st.slider("🛠️ Technical", 0, 10, 5, help="Performance: 0=Pífio, 10=Excelente Projeção/Fixação")
        with c3:
            f_creative = st.slider("🎨 Creative", 0, 10, 5, help="Originalidade: 0=Clichê, 10=Inovador")

        if st.button("🧬 TRAIN NEURAL NETWORK", type="primary", use_container_width=True):
            if st.session_state.current_formula:
                current_data = st.session_state.current_formula
                feedback_vector = {
                    "hedonic": f_hedonic,
                    "technical": f_tech,
                    "creative": f_creative
                }
                engine.register_human_feedback(current_data.get("id"), feedback_vector)
        
            restricted_chems = list(comp_engine.IFRA_LIMITS.keys())
        
            for _ in range(2):
                corrupted_mols = [m.copy() for m in current_data['molecules']]
            
                target_bad_chem = random.choice(restricted_chems)
            
                found = False
                for m in corrupted_mols:
                    if target_bad_chem in m['name']:
                        m['weight_factor'] *= 20.0 
                        found = True
            
                if not found:
                    corrupted_mols.append({'name': target_bad_chem, 'weight_factor': 0.8, 'category': 'Base'})

                engine.register_human_feedback(
                    None,
                    {"hedonic": 0.1, "technical": 0.0, "creative": 0.0}, 
                    custom_mols=corrupted_mols 
                )

                engine.trainer.train_step(engine.buffer) #
        
                st.success("Neuro-weights updated with Safety Examples!")
                time.sleep(1)
                
                generate_next() 
                
                st.rerun()

if st.session_state.history:
    st.markdown("---")
    with st.expander("VIEW PREVIOUS ITERATIONS"):
        st.table(pd.DataFrame(st.session_state.history))