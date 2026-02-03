import time
import torch
import os
import sys
from infra.gemini_client import GeminiClient
from core.strategy import StrategyAgent
from core.discovery import DiscoveryEngine
from core.model import MoleculeGNN

# --- IMPORTAÇÃO DO DASHBOARD ---
try:
    from dashboard import render_dashboard
except ImportError:
    print("⚠️ Dashboard não encontrado. Rodando em modo texto.")
    render_dashboard = None

# =========================================================
# CONFIGURAÇÃO GLOBAL
# =========================================================

NODE_FEATURES = 5 
ROUNDS = int(os.getenv("ROUNDS", 30))
# Reduzi o sleep para 0 porque o input já pausa o programa
SLEEP_BETWEEN_ROUNDS = 0 
CHECKPOINT_INTERVAL = int(os.getenv("CHECKPOINT_INTERVAL", 5))
CSV_PATH = os.getenv("CSV_PATH", "insumos.csv")

# =========================================================
# 1. INICIALIZAÇÃO
# =========================================================

print("\n🧪 [SYSTEM] Inicializando Graph Neural Network (GNN)...")
model = MoleculeGNN(num_node_features=NODE_FEATURES)

loaded = False
try:
    if hasattr(model, 'load'):
        loaded = model.load()
except Exception as e:
    print(f"⚠️ Erro ao carregar pesos: {e}. Iniciando do zero.")

# =========================================================
# 2. AGENTES E ENGINE
# =========================================================

print("🤖 [SYSTEM] Inicializando Agentes...")
try:
    llm_client = GeminiClient()
    strategy_agent = StrategyAgent(llm_client)
except:
    print("⚠️ Aviso: GeminiClient offline.")
    strategy_agent = None

engine = DiscoveryEngine(
    model=model,
    strategy_agent=strategy_agent,
    csv_path=CSV_PATH
)

if not loaded:
    print("\n🔥 [SYSTEM] Iniciando Warmup...")
    engine.warmup(n_samples=20)

# =========================================================
# 3. LOOP PRINCIPAL (AGORA INTERATIVO)
# =========================================================

print(f"\n🚀 Ciclo Hard Science iniciado ({ROUNDS} rodadas)...")

try:
    for i in range(1, ROUNDS + 1):
        try:
            # 1. Gera e Evolui
            engine.discover(rounds=1)

            # 2. Pega o Melhor Resultado
            if engine.discoveries:
                best = engine.discoveries[-1]
                molecules = best['molecules']
                
                # 3. Prepara Dados para o Dashboard
                stats = best['chemistry']
                # Injeta o score final para aparecer no painel
                stats['final_score'] = best.get('fitness', 0.0)
                # Garante que os vetores neurais estejam acessíveis
                if 'neuro_vectors' not in stats and 'neuro_vectors' in best.get('chemistry', {}):
                     stats['neuro_vectors'] = best['chemistry']['neuro_vectors']
                
                risks, _ = engine.chemistry._detect_chemical_risks(molecules)
                
                # 4. RENDERIZA O DASHBOARD
                if render_dashboard:
                    # Limpa a tela para dar destaque ao painel
                    os.system('cls' if os.name == 'nt' else 'clear')
                    
                    render_dashboard(
                        formula_name=f"Gen {i} - {stats.get('olfactive_profile', 'Blend')}",
                        molecules=molecules,
                        stats=stats,
                        risks=risks
                    )
                else:
                    # Fallback Texto Simples (caso o dashboard falhe)
                    print(f"\n--- RODADA {i} ---")
                    print(f"Fórmula: {', '.join([m['name'] for m in molecules])}")
                    print(f"Score: {best['fitness']:.3f} | Fixação: {stats['longevity']:.1f}h")

                # 5. INTERAÇÃO HUMANA (RESTAURADA!)
                print("\n" + "-"*60)
                print(" 📝 Dê sua nota para ensinar a IA (ou Enter para pular)")
                print("-"*60)
                
                while True:
                    user_input = input(" > Nota (0-10): ")
                    
                    if not user_input.strip():
                        print(" ⏩ Pulando feedback...")
                        break
                    
                    if user_input.lower() in ['q', 's', 'exit']:
                        raise KeyboardInterrupt # Permite sair digitando 'q'
                    
                    try:
                        score = float(user_input)
                        if 0 <= score <= 10:
                            # O PULO DO GATO: Registra o feedback e treina a GNN na hora
                            discovery_id = best.get("id", -1)
                            engine.register_human_feedback(discovery_id, score)
                            print(f" ✅ Aprendizado registrado! A IA ajustará os pesos.")
                            time.sleep(1) # Breve pausa para ler a confirmação
                            break
                        else:
                            print(" ⚠️  A nota deve ser entre 0 e 10.")
                    except ValueError:
                        print(" ⚠️  Entrada inválida. Digite um número.")

            # Checkpoint Automático
            if i % CHECKPOINT_INTERVAL == 0:
                print(f" 💾 Salvando progresso...")
                engine.save_results()
                if hasattr(model, 'save'): model.save()

        except Exception as e:
            print(f"❌ Erro na rodada {i}: {e}")
            time.sleep(2)

except KeyboardInterrupt:
    print("\n🛑 Encerrando e salvando memórias...")

finally:
    engine.save_results()
    if hasattr(model, 'save'): model.save()
    print("👋 Bye.")