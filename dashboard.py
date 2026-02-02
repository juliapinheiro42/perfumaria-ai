import streamlit as st
import pandas as pd
import time
import os
import torch

# Importações do Backend
from infra.gemini_client import GeminiClient
from core.strategy import StrategyAgent
from core.discovery import DiscoveryEngine
from core.model import MoleculeGNN
from core.market import PerfumeBusinessEngine

# Configuração da Página
st.set_page_config(
    page_title="L'Oréal AI Lab - Evolution Dashboard",
    page_icon="🧬",
    layout="wide"
)

# Estilização CSS Personalizada
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
        padding: 5px 10px;
        margin: 2px;
        border-radius: 15px;
        font-size: 0.85em;
        font-weight: 500;
        color: #1e1e1e;
    }
    .stProgress > div > div > div > div {
        background-color: #4CAF50;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# 1. CACHE DE RECURSOS (SINGLETON)
# =========================================================
@st.cache_resource
def load_engine():
    print("🔄 [SYSTEM] Inicializando Cérebro Digital...")
    
    # 1. Carrega GNN
    model = MoleculeGNN(num_node_features=5)
    try:
        if hasattr(model, 'load'):
            if model.load():
                print("✅ Pesos neurais carregados.")
            else:
                print("🆕 Iniciando novos pesos neurais.")
    except Exception as e:
        print(f"⚠️ Aviso de Modelo: {e}")

    # 2. Carrega Agente Estratégico (Gemini)
    try:
        llm_client = GeminiClient()
        strategy_agent = StrategyAgent(llm_client)
    except:
        strategy_agent = None
        print("⚠️ Modo Offline (Sem LLM)")

    # 3. Inicializa Engine de Descoberta
    engine = DiscoveryEngine(
        model=model,
        strategy_agent=strategy_agent,
        csv_path="insumos.csv"
    )
    
    return engine

# Inicialização Segura
try:
    engine = load_engine()
except Exception as e:
    st.error(f"Erro crítico ao carregar sistema: {e}")
    st.stop()

# =========================================================
# 2. GERENCIAMENTO DE ESTADO (SESSION STATE)
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
    st.image("https://upload.wikimedia.org/wikipedia/commons/9/9d/L%27Or%C3%A9al_logo.svg", width=150)
    st.header("🎛️ Parâmetros de Evolução")
    
    # Controle de Âncoras
    all_ingredients = sorted(engine.insumos_dict.keys())
    anchors = st.multiselect(
        "⚓ Âncoras (Obrigatórios)", 
        options=all_ingredients,
        help="A IA será forçada a incluir estes ingredientes na estrutura."
    )
    
    if anchors != engine.anchors:
        engine.anchors = anchors
        st.toast(f"Âncoras atualizadas: {len(anchors)} itens", icon="⚓")

    st.divider()
    
    # Controles de Sessão
    if st.button("🗑️ Resetar Experimento"):
        st.session_state.history = []
        st.session_state.round_count = 0
        st.session_state.current_formula = None
        st.rerun()

# =========================================================
# 4. LÓGICA DE NEGÓCIO E EVOLUÇÃO
# =========================================================
def generate_next():
    with st.spinner("🧬 Sintetizando nova linhagem molecular..."):
        # Executa o ciclo de descoberta da IA
        discoveries = engine.discover(rounds=1)
        
        if discoveries:
            new_result = discoveries[-1]
            st.session_state.current_formula = new_result
            st.session_state.round_count += 1
        else:
            st.error("Falha na convergência genética. Tente remover âncoras conflitantes.")

def submit_feedback():
    score = st.session_state.feedback_slider
    
    if st.session_state.current_formula:
        # 1. Registra aprendizado no cérebro da IA
        engine.register_human_feedback(-1, score)
        
        # 2. Salva no histórico visual
        data = st.session_state.current_formula
        st.session_state.history.insert(0, {
            "Rodada": st.session_state.round_count,
            "Score Humano": score,
            "Anti-Dupe": data['chemistry'].get('complexity', 0),
            "Evolução": data['chemistry'].get('evolution', 0),
            "Ingredientes": ", ".join([m['name'] for m in data['molecules']])
        })
        
        st.session_state.last_feedback = score
        st.toast(f"Feedback {score} registrado! Otimizando pesos...", icon="🧠")
        
        # 3. Gera a próxima geração imediatamente
        generate_next()

# =========================================================
# 5. DASHBOARD PRINCIPAL
# =========================================================
st.title("🧪 Laboratório de Evolução Genética")
st.markdown("**Objetivo:** Criar fragrâncias de luxo à prova de cópias e otimizadas por neurociência.")

# Tela Inicial (Sem Fórmula)
if st.session_state.current_formula is None:
    st.info("O sistema neural está pronto. Inicie o processo criativo.")
    if st.button("🚀 Iniciar Ciclo de Criação", type="primary"):
        generate_next()
        st.rerun()

# Tela de Análise (Com Fórmula)
else:
    # Recupera dados da sessão
    data = st.session_state.current_formula
    mols = data['molecules']
    chem = data['chemistry']
    
    # Instancia motor de negócios para calcular KPIs financeiros e de mercado
    biz_engine = PerfumeBusinessEngine()
    market_analysis = biz_engine.calculate_global_fit(mols)
    financials = biz_engine.estimate_financials(mols, data.get('market_tier', 'Luxury'))

    # Layout de Colunas
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(f"Formulação Genética #{st.session_state.round_count}")
        
        # --- LINHA 1: KPIs TÉCNICOS ---
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Fixação", f"{chem.get('longevity', 0):.1f}h", help="Longevidade estimada na pele")
        c2.metric("Projeção", f"{chem.get('projection', 0):.1f}/10", help="Rastro (Sillage)")
        c3.metric("Anti-Dupe", f"{chem.get('complexity', 0):.1f}/10", help="Nível de caos químico (Dificuldade de GC/MS)")
        c4.metric("Evolução", f"{chem.get('evolution', 0):.1f}/10", help="Jornada olfativa (Topo vs Fundo)")

        # --- LINHA 2: NEURO-TARGETING (5 MERCADOS) ---
        st.markdown("### 🌍 Análise de Oportunidade Global")
        
        # Agora exibimos 5 colunas para cobrir todo o globo
        regions_order = ["EUA", "Europa", "América Latina", "Ásia", "Oriente Médio"]
        m_cols = st.columns(5)
        
        rankings = market_analysis.get('rankings', {})
        
        for i, reg in enumerate(regions_order):
            score = rankings.get(reg, 0.0)
            # Formatação curta para caber na coluna
            short_name = reg.replace("América Latina", "LatAm").replace("Oriente Médio", "Or. Médio")
            m_cols[i].metric(short_name, f"{score:.1f}")
            m_cols[i].progress(min(score / 10.0, 1.0))
        
        # Destaque do Vencedor com Estratégia Específica
        best_market = market_analysis.get('best', 'Indefinido')
        market_label = market_analysis.get('label', '')
        
        if "Ásia" in best_market:
            st.info(f"📍 **Estratégia:** {best_market} ({market_label}). Foco em pureza 'Zen' e bem-estar.")
        elif "América Latina" in best_market:
            st.success(f"📍 **Estratégia:** {best_market} ({market_label}). Foco em vibração solar e sedução.")
        elif "Oriente Médio" in best_market:
            st.warning(f"📍 **Estratégia:** {best_market} ({market_label}). Foco em mistério, resinas e status.")
        elif "Europa" in best_market:
            st.info(f"📍 **Estratégia:** {best_market} ({market_label}). Foco em 'Clean Beauty' e elegância minimalista.")
        elif "EUA" in best_market:
            st.success(f"📍 **Estratégia:** {best_market} ({market_label}). Foco em impacto, projeção e 'Sex Appeal'.")
        else:
            st.caption(f"📍 Estratégia Sugerida: {best_market} ({market_label})")

        # --- LINHA 3: COMPOSIÇÃO VISUAL ---
        st.markdown("### 🥗 Estrutura Molecular")
        
        html_tags = ""
        for m in mols:
            # Recupera dados enriquecidos
            info = engine.insumos_dict.get(m['name'], {})
            tier = info.get('complexity_tier', 1)
            is_anchor = m['name'] in engine.anchors
            
            # Lógica de Cores e Ícones
            if tier == 3: # Natural Complexo
                bg_color = "#d1e9ff" # Azul Claro
                border = "2px solid #007bff"
                icon = "💎"
            elif tier == 2: # High-Tech
                bg_color = "#e2d1ff" # Roxo Claro
                border = "1px solid #6f42c1"
                icon = "🧬"
            else: # Sintético Comum
                bg_color = "#e8f5e9" # Verde Claro
                border = "1px solid #c3e6cb"
                icon = "🌿"
                
            if is_anchor:
                border = "2px solid #FFD700" # Ouro para âncoras
                icon = "⚓"

            html_tags += f"""
            <span class='ingredient-tag' style='background-color:{bg_color}; border:{border};'>
                {icon} <b>{m['name']}</b> <small>({m.get('category')})</small>
            </span>
            """
        st.markdown(html_tags, unsafe_allow_html=True)

        with st.expander("📊 Ver Tabela Técnica Detalhada"):
            df_display = pd.DataFrame(mols)
            
            # --- CORREÇÃO DE SEGURANÇA ---
            if 'formula_pct' not in df_display.columns:
                df_display['formula_pct'] = 1.0 / len(df_display) if len(df_display) > 0 else 0.0

            def get_insumo_data(name, field, default):
                return engine.insumos_dict.get(name, {}).get(field, default)

            df_display['price_per_kg'] = df_display['name'].apply(lambda x: get_insumo_data(x, 'price_per_kg', 0.0))
            df_display['Tier'] = df_display['name'].apply(lambda x: get_insumo_data(x, 'complexity_tier', 1))
            df_display['Neuro Target'] = df_display['name'].apply(lambda x: get_insumo_data(x, 'neuro_target', '-'))
            
            column_config = {
                "name": "Ingrediente",
                "category": "Nota",
                "formula_pct": st.column_config.NumberColumn("Conc. (%)", format="%.2f%%"),
                "price_per_kg": st.column_config.NumberColumn("Preço ($/kg)", format="$%.2f"),
                "Tier": "Anti-Dupe Lvl",
                "Neuro Target": "Efeito Funcional"
            }

            st.dataframe(
                df_display[['name', 'category', 'formula_pct', 'Tier', 'Neuro Target', 'price_per_kg']],
                use_container_width=True,
                hide_index=True,
                column_config=column_config
            )

    with col2:
        # --- PAINEL LATERAL: VIABILIDADE ---
        st.markdown("### 💰 Business Case")
        st.write("Análise financeira preliminar para escala industrial.")
        
        res_col1, res_col2 = st.columns(2)
        res_col1.metric("Custo (100ml)", f"${financials.get('cost', 0):.2f}")
        res_col2.metric("Margem", f"{financials.get('margin', 0)*100:.0f}%")
        
        st.write(f"**Preço Sugerido (Varejo):** ${financials.get('price', 0):.2f}")
        
        st.divider()
        
        # --- FEEDBACK LOOP ---
        st.markdown("### 👃 Avaliação Sensorial")
        st.markdown("""
        <div class="metric-card">
            O quão alinhada esta fórmula está com o brief?
        </div>
        """, unsafe_allow_html=True)
        
        st.slider("Nota (0-10)", 0.0, 10.0, 5.0, 0.5, key="feedback_slider")
        
        st.button(
            "✅ Aprovar & Evoluir", 
            type="primary", 
            on_click=submit_feedback, 
            use_container_width=True,
            help="Envia esta nota para o Agente de Estratégia e gera uma nova variação."
        )

        if st.session_state.last_feedback > 0:
            st.caption(f"Última Nota: {st.session_state.last_feedback}")

    # Histórico no Fundo
    st.divider()
    st.subheader("📜 Linhagem Evolutiva")
    if st.session_state.history:
        st.dataframe(
            pd.DataFrame(st.session_state.history),
            use_container_width=True,
            hide_index=True
        )