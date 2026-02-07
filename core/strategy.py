from email.mime import text
import json
import re
import random
from core.presets import ACORDES_LIB, PERFUME_SKELETONS


class StrategyAgent:
    def __init__(self, llm_client):
        self.llm = llm_client

    def propose_strategy(self, discoveries, goal):
        try:
            history = sorted(
                discoveries, key=lambda d: d['fitness'], reverse=True)[:5]

            prompt = (
                f"You are a Master Perfumer AI. Goal: {goal}.\n\n"
                f"Available Accords Library (Building Blocks):\n{json.dumps(ACORDES_LIB, indent=2)}\n\n"
                f"Analyze the history and propose a new search strategy. "
                f"Choose the most suitable accord from the library above to serve as the core of this goal.\n\n"
                f"RETURN ONLY RAW JSON. NO MARKDOWN. NO COMMENTS.\n"
                f"Required JSON Schema:\n"
                f"{{\n"
                f'  "recommended_accord": "Name_of_Accord_or_None",\n'
                f'  "num_molecules": [min_int, max_int],\n'
                f'  "volatility_range": [min_float, max_float],\n'
                f'  "molecular_weight_range": [min_float, max_float],\n'
                f'  "exploration_bias": float_0_to_1\n'
                f"}}\n\n"
                f"Recent Best Discoveries:\n{json.dumps(history, indent=2)}"
            )

            response = self.llm.generate(prompt)
            strategy = self._parse_response(response)

            if strategy is None:
                print(" [IA] Failed to parse strategy JSON. Using default.")
                return self._get_default_strategy()

            print(
                f" [IA] Estratégia gerada. Acorde sugerido: {strategy.get('recommended_accord')}")

            if random.random() < strategy.get("exploration_bias", 0.3):
                print(" [EPSILON-GREEDY] Forçando exploração de novas moléculas!")
                strategy["num_molecules"] = [4, 6]
                strategy["molecular_weight_range"] = [100, 450]

            return strategy

        except Exception as e:
            print(f"\n Falha na IA (StrategyAgent): {str(e)}")
            print("   -> Usando estratégia padrão de fallback.")
            return self._get_default_strategy()

    def _parse_response(self, text):
        try:
            import re
            start_idx = text.find('{')
            end_idx = text.rfind('}')

            if start_idx == -1 or end_idx == -1:
                raise ValueError("No JSON brackets found")

            json_str = text[start_idx: end_idx + 1]
            return json.loads(json_str)
        except Exception:
            pass

    def _get_default_strategy(self):
        return {
            "recommended_accord": None,
            "num_molecules": [3, 5],
            "volatility_range": [0.2, 0.8],
            "molecular_weight_range": [150, 350],
            "exploration_bias": 0.3
        }

    def mutate(self, molecules):
        is_green_mode = getattr(
            self.discovery, 'target_vector', None) is not None

        for i, m in enumerate(molecules):
            if random.random() < self.mutation_rate:

                if m.get("accord_id") and random.random() < 0.5:
                    continue

                swap_chance = 0.6 if is_green_mode else 0.3

                if random.random() < swap_chance:
                    if is_green_mode:
                        print(
                            f" [EVO] Buscando alternativa GREEN para {m.get('name')}...")
                        new_mol = self.discovery.get_green_replacement(m)

                        new_mol['weight_factor'] = m.get('weight_factor', 1.0)
                        molecules[i] = new_mol
                    else:
                        molecules[i] = self.discovery._random_molecule()
                    continue

                mw = m.get("molecular_weight", 150.0)
                mw *= random.uniform(1 - self.mutation_strength,
                                     1 + self.mutation_strength)
                m["molecular_weight"] = float(np.clip(mw, 80, 600))

                pol = m.get("polarity", 2.0)
                pol *= random.uniform(1 - self.mutation_strength,
                                      1 + self.mutation_strength)
                m["polarity"] = float(np.clip(pol, 0.1, 6.0))
