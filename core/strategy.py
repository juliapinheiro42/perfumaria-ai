import json
import re
import random
from core.presets import ACORDES_LIB, PERFUME_SKELETONS

class StrategyAgent:
    def __init__(self, llm_client):
        self.llm = llm_client

    def propose_strategy(self, discoveries, goal):
        try:
            history = sorted(discoveries, key=lambda d: d['fitness'], reverse=True)[:5]


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
            
            print(f" [IA] Estratégia gerada. Acorde sugerido: {strategy.get('recommended_accord')}")

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
            match = re.search(r'(\{[\s\S]*\})', text)
            if not match: 
                raise ValueError("No JSON found in response")
            
            json_str = match.group(1)
            data = json.loads(json_str)
            
            defaults = self._get_default_strategy()
            
            for key, default_val in defaults.items():
                if key not in data:
                    data[key] = default_val
            
            if not isinstance(data["num_molecules"], list): data["num_molecules"] = [3, 5]
            if len(data["num_molecules"]) != 2: data["num_molecules"] = [3, 5]
            
            return data

        except json.JSONDecodeError:
            raise ValueError("Invalid JSON format")

    def _get_default_strategy(self):
        return {
            "recommended_accord": None,
            "num_molecules": [3, 5], 
            "volatility_range": [0.2, 0.8], 
            "molecular_weight_range": [150, 350], 
            "exploration_bias": 0.3
        }