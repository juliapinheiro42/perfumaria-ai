import streamlit as st
import pandas as pd
import time
import os
import torch

from infra.gemini_client import GeminiClient
from core.strategy import StrategyAgent
from core.discovery import DiscoveryEngine
from core.model import MoleculeGNN
from core.encoder import FeatureEncoder

st.set_page_config(
    page_title="Laboratório de Evolução IA",
    page_icon="🧬",
    layout="wide"
)

st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
    }
    .ingredient-tag {
        display: inline-block;
        background-color: #e8f5e9;
        color: #2e7d32;
        padding: 5px 10px;
        margin: 2px;
        border-radius: 15px;
        font-size: 0.9em;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# 1. CACHE DE RECURSOS
# =========================================================
@st.cache_resource
def load_engine():
    print("🔄 [SYSTEM] Carregando Engine e Modelo...")
    
    model = MoleculeGNN(num_node_features=5)
    try:
        if hasattr(model, 'load'):
            if model.load():
                print("✅ Pesos carregados.")
            else:
                print("🆕 Pesos novos iniciados.")
    except:
        pass

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
    st.error(f"Erro ao carregar o sistema: {e}")
    st.stop()

# =========================================================
# 2. GERENCIAMENTO DE ESTADO (Session State)
# =========================================================
if 'current_formula' not in st.session_state:
    st.session_state.current_formula = None
if 'history' not in st.session_state:
    st.session_state.history = []
if 'round_count' not in st.session_state:
    st.session_state.round_count = 0
if 'last_feedback' not in st.session_state:
    st.session_state.last_feedback = 0.0

# =========================================================
# 3. SIDEBAR - CONTROLES
# =========================================================
with st.sidebar:
    st.header("🎛️ Configuração")
    
    all_ingredients = sorted(engine.insumos_dict.keys())
    anchors = st.multiselect(
        "⚓ Âncoras (Obrigatórios)", 
        options=all_ingredients,
        help="A IA será obrigada a usar estes ingredientes."
    )
    
    if anchors != engine.anchors:
        engine.anchors = anchors
        st.toast(f"Âncoras atualizadas: {len(anchors)} itens", icon="⚓")

    st.divider()
    
    if st.button("🗑️ Limpar Histórico / Reset"):
        st.session_state.history = []
        st.session_state.round_count = 0
        st.session_state.current_formula = None
        st.rerun()

# =========================================================
# 4. FUNÇÕES LÓGICAS
# =========================================================
def generate_next():
    with st.spinner("⚗️ A IA está criando uma nova fórmula..."):
        discoveries = engine.discover(rounds=1)
        
        if discoveries:
            new_result = discoveries[-1]
            st.session_state.current_formula = new_result
            st.session_state.round_count += 1
        else:
            st.error("A IA não conseguiu gerar uma fórmula válida (verifique as âncoras).")

def submit_feedback():
    score = st.session_state.feedback_slider
    
    if st.session_state.current_formula:
        engine.register_human_feedback(-1, score)
        
        formula_data = st.session_state.current_formula
        st.session_state.history.insert(0, {
            "Rodada": st.session_state.round_count,
            "Nota": score,
            "AI Score": formula_data.get('ai_score', 0),
            "Ingredientes": ", ".join([m['name'] for m in formula_data['molecules']])
        })
        
        st.session_state.last_feedback = score
        st.toast(f"Nota {score} registrada! O modelo aprendeu.", icon="🧠")
        
        generate_next()

# =========================================================
# 5. ÁREA PRINCIPAL
# =========================================================
st.title("🧬 Laboratório de Evolução Genética")
st.markdown("Use este painel para treinar a IA. Dê notas às fórmulas para guiar a evolução.")

if st.session_state.current_formula is None:
    st.info("O sistema está pronto. Clique abaixo para começar o ciclo de evolução.")
    if st.button("🚀 Iniciar Ciclo de Criação", type="primary"):
        generate_next()
        st.rerun()

else:
    data = st.session_state.current_formula
    mols = data['molecules']
    chem = data['chemistry']
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(f"🧪 Fórmula da Rodada #{st.session_state.round_count}")
        
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Fixação Estimada", f"{chem.get('longevity', 0):.1f}h")
        c2.metric("Projeção", f"{chem.get('projection', 0):.1f}/10")
        complexity_score = chem.get('complexity', 0)
        c3.metric("Anti-Dupe", f"{complexity_score:.1f}/10", 
            help="Calcula o quão difícil é fazer engenharia reversa baseado em Naturais Complexos e Moléculas High-Tech.")
        c4.metric("Confiança da IA", f"{data.get('ai_score', 0):.3f}")

        st.markdown("### 🥗 Ingredientes")
        
        html_tags = ""
        for m in mols:
            is_anchor = m['name'] in engine.anchors
            style = "border: 2px solid #FFD700;" if is_anchor else ""
            icon = "⚓ " if is_anchor else ""
            html_tags += f"<span class='ingredient-tag' style='{style}'>{icon}<b>{m['name']}</b> ({m.get('category')})</span>"
        
        st.markdown(html_tags, unsafe_allow_html=True)
        with st.expander("Ver detalhes técnicos (Tabela)"):
            df_display = pd.DataFrame(mols)
            cols_to_show = ['name', 'category', 'complexity_tier', 'price_per_kg', 'molecular_weight']
            available_cols = [c for c in cols_to_show if c in df_display.columns]
            st.dataframe(df_display[available_cols], hide_index=True)
    
        st.markdown("### 🛡️ Proteção Industrial")
    if complexity_score >= 7.0:
        st.success("🔥 **Alta Complexidade:** Esta fórmula possui 'ruído' químico elevado. Extremamente difícil de copiar via GC/MS.")
    elif complexity_score >= 4.0:
        st.warning("⚖️ **Complexidade Média:** Requer equipamentos avançados para duplicação. Considerada segura para nicho.")
    else:
        st.error("⚠️ **Baixa Complexidade:** Risco de engenharia reversa simplificada. Considere adicionar Naturais do Tier 3.")

    with col2:
        st.markdown("### 👃 Sua Avaliação")
        st.markdown("""
        <div class="metric-card">
            Como você avalia o cheiro/composição desta fórmula?
        </div>
        """, unsafe_allow_html=True)
        
        st.slider("Nota (0 = Horrível, 10 = Perfeito)", 0.0, 10.0, 5.0, 0.5, key="feedback_slider")
        
        st.button("✅ Enviar Avaliação & Gerar Próxima", 
                  type="primary", 
                  on_click=submit_feedback, 
                  use_container_width=True)
        
        st.markdown(f"**Status:** Última nota dada foi **{st.session_state.last_feedback}**")
        
        if st.session_state.last_feedback >= 6.0:
            st.caption("📈 Modo: Evolução (Refinando a fórmula anterior)")
        else:
            st.caption("🎲 Modo: Exploração (Tentando algo novo)")

    st.divider()

    st.subheader("📜 Histórico de Aprendizado")
    if st.session_state.history:
        df_hist = pd.DataFrame(st.session_state.history)
        st.dataframe(
            df_hist, 
            column_config={
                "Nota": st.column_config.ProgressColumn(format="%.1f", min_value=0, max_value=10, width="small"),
                "Ingredientes": st.column_config.TextColumn(width="large")
            },
            hide_index=True,
            use_container_width=True
        )
    else:
        st.text("Nenhuma avaliação feita ainda.")