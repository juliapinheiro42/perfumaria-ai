import time
import torch
import os
from infra.gemini_client import GeminiClient
from core.strategy import StrategyAgent
from core.discovery import DiscoveryEngine
from core.model import MoleculeGNN

# =========================================================
# CONFIGURAÇÃO GLOBAL
# =========================================================

NODE_FEATURES = 5 
ROUNDS = int(os.getenv("ROUNDS", 30))
SLEEP_BETWEEN_ROUNDS = int(os.getenv("SLEEP_BETWEEN_ROUNDS", 1))
CHECKPOINT_INTERVAL = int(os.getenv("CHECKPOINT_INTERVAL", 5))
CSV_PATH = os.getenv("CSV_PATH", "insumos.csv")

# =========================================================
# 1. INICIALIZAÇÃO DO MODELO
# =========================================================

print("\n🧪 [SYSTEM] Inicializando Graph Neural Network (GNN)...")
model = MoleculeGNN(num_node_features=NODE_FEATURES)

loaded = False
try:
    if hasattr(model, 'load'):
        loaded = model.load()
except Exception as e:
    print(f"⚠️ Erro ao carregar pesos: {e}. Iniciando do zero.")

if loaded:
    print("✅ Pesos da GNN carregados com sucesso.")
else:
    print("🆕 Iniciando nova rede neural do zero.")

# =========================================================
# 2. INICIALIZAÇÃO DOS AGENTES
# =========================================================

print("🤖 [SYSTEM] Inicializando Agentes...")
try:
    llm_client = GeminiClient()
    strategy_agent = StrategyAgent(llm_client)
except Exception as e:
    print(f"⚠️ Aviso: GeminiClient não configurado ({e}). IA rodará sem estratégia textual.")
    strategy_agent = None

engine = DiscoveryEngine(
    model=model,
    strategy_agent=strategy_agent,
    csv_path=CSV_PATH
)

# =========================================================
# 4. WARMUP (AQUECIMENTO)
# =========================================================
# Só roda se o modelo for novo para popular o buffer de memória
if not loaded:
    print("\n🔥 [SYSTEM] Iniciando Warmup (Aquecimento)...")
    engine.warmup(n_samples=50)

# =========================================================
# 5. LOOP PRINCIPAL
# =========================================================

print(f"\n🚀 Ciclo de descobertas iniciado ({ROUNDS} rodadas)...")
print("   Dica: Dê notas altas (>6) para manter a fórmula, notas baixas para mudar.")

try:
    for i in range(1, ROUNDS + 1):
        try:
            print(f"\n--- ⚗️  RODADA {i}/{ROUNDS} ---")
            
            engine.discover(rounds=1)

            if engine.discoveries:
                best = engine.discoveries[-1]
                
                print("\n" + "="*60)
                print(f"👃 AVALIAÇÃO OLFATIVA")
                print("="*60)
                
                molecules = best['molecules']
                names = [m['name'] for m in molecules]
                
                formula_str = ", ".join(names)
                if len(formula_str) > 80:
                    print(f"FÓRMULA:\n{formula_str}")
                else:
                    print(f"FÓRMULA: {formula_str}")
                
                print("-" * 60)
                
                chem = best.get('chemistry', {})
                longevity = chem.get('longevity', 0)
                projection = chem.get('projection', 0)
                ai_score = best.get('ai_score', 0)
                sim = best.get('similarity_to_target', 0)
                
                stats = f"⏳ Fixação: {longevity:.1f}h  |  📢 Projeção: {projection:.1f}/10  |  🤖 IA Score: {ai_score:.3f}"
                if sim > 0:
                    stats += f"  |  🎯 Similaridade: {sim:.2f}"
                print(stats)
                print("-" * 60)
                
                while True:
                    user_input = input(" > Nota (0-10) [Enter = Pular]: ")
                    
                    if not user_input.strip():
                        print(" ⏩ Avaliação pulada.")
                        break
                    
                    try:
                        score = float(user_input)
                        if 0 <= score <= 10:
                            # Use ID if available, else fallback to -1
                            discovery_id = best.get("id", -1)
                            engine.register_human_feedback(discovery_id, score)
                            break
                        else:
                            print(" ⚠️  Por favor, entre uma nota entre 0 e 10.")
                    except ValueError:
                        print(" ⚠️  Entrada inválida. Digite um número.")

                print("="*60)
            
            if i % CHECKPOINT_INTERVAL == 0:
                print(f" 💾 [CHECKPOINT] Salvando dados...")
                engine.save_results()
                if hasattr(model, 'save'): model.save()

            if i < ROUNDS:
                time.sleep(SLEEP_BETWEEN_ROUNDS)

        except Exception as e:
            print(f"❌ Erro na rodada {i}: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(5)

except KeyboardInterrupt:
    print("\n\n🛑 Interrupção pelo usuário.")

finally:
    print("\n💾 Salvando estado final...")
    engine.save_results()
    if hasattr(model, 'save'): model.save()
    print("👋 Encerrado.")