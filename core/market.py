import pandas as pd
from typing import List, Dict, Any

try:
    df_insumos = pd.read_csv("insumos.csv")
    df_insumos = df_insumos.drop_duplicates(subset=['name'])
    df_insumos = df_insumos.set_index("name")
    DADOS_INSUMOS = df_insumos.to_dict('index')
except Exception as e:
    print(f"⚠️ [MARKET] Erro ao carregar insumos.csv: {e}")
    DADOS_INSUMOS = {}

class PerfumeBusinessEngine:
    def __init__(self, dilution: float = 0.20, bottle_ml: int = 100):
        self.dilution = dilution 
        self.bottle_ml = bottle_ml
        
        self.MARKETS = {
            "Ásia": {"target": "GABA Agonist", "label": "Zen & Wellness", "weight": 1.5},
            "América Latina": {"target": "Dopamine Boost", "label": "Energia & Vibração", "weight": 1.5},
            "Oriente Médio": {"target": "Cortisol Balance", "label": "Opulência & Foco", "weight": 2.0},
            "Europa": {"target": "GABA Agonist", "label": "Clean Beauty", "weight": 1.0},
            "EUA": {"target": "Dopamine Boost", "label": "Power & Sexy", "weight": 1.0}
        }

    def calculate_global_fit(self, molecules: List[Dict]) -> Dict:
        scores = {k: 0.0 for k in self.MARKETS.keys()}
        
        for m in molecules:
            # Tenta pegar do objeto ou do dicionário global
            target = m.get('neuro_target')
            if not target or str(target) == 'nan':
                target = DADOS_INSUMOS.get(m['name'], {}).get('neuro_target', 'Neutral')
            
            for region, criteria in self.MARKETS.items():
                if target == criteria['target']:
                    scores[region] += 2.0 * criteria.get('weight', 1.0)
        
        final_scores = {k: min(10.0, v) for k, v in scores.items()}
        best_market = max(final_scores, key=final_scores.get) if final_scores else "Global"
        
        return {
            "rankings": final_scores,
            "best": best_market,
            "label": self.MARKETS.get(best_market, {}).get('label', 'General')
        }

    def estimate_financials(self, molecules: List[Dict], tech_score: float, neuro_score: float) -> Dict:
        total_essence_g = self.bottle_ml * self.dilution * 0.95
        total_cost = 0.0
        
        for m in molecules:
            pct = m.get("formula_pct", 1.0/len(molecules) if molecules else 0)
            price_kg = DADOS_INSUMOS.get(m['name'], {}).get("price_per_kg", 100.0)
            total_cost += (total_essence_g * pct) * (price_kg / 1000.0)
            
        # Precificação Dinâmica L'Oréal Luxe
        multiplier = 5.0 # Mass Market
        tier = "Mass Market"
        
        if tech_score > 7.0: multiplier = 10.0; tier = "Luxury"
        elif tech_score > 9.0: multiplier = 15.0; tier = "Niche"
            
        if neuro_score > 5.0: multiplier += 2.0 
            
        suggested_price = total_cost * multiplier
        profit = suggested_price - total_cost
        margin = (profit / suggested_price) if suggested_price > 0 else 0.0
        
        return {
            "cost": round(total_cost, 2),
            "price": round(suggested_price, 2),
            "margin_pct": round(margin * 100, 1),
            "market_tier": tier,
            "applied_multiplier": multiplier
        }

    def validate_ifra(self, molecules: List[Dict]) -> Dict:
        logs = []
        is_legal = True
        
        for m in molecules:
            pct = m.get("formula_pct", 1.0/len(molecules) if molecules else 0)
            skin_exposure = pct * self.dilution
            
            # Busca limite, garantindo float e padrão seguro (100% ou 1.0)
            try:
                limit_raw = DADOS_INSUMOS.get(m['name'], {}).get("ifra_limit", 1.0)
                limit = float(limit_raw)
            except (ValueError, TypeError):
                limit = 1.0

            if skin_exposure > limit:
                is_legal = False
                
                # --- CORREÇÃO DO ZERO DIVISION ERROR ---
                if limit > 0.00001:
                    severity = (skin_exposure - limit) / limit
                else:
                    # Se limite for 0 (banido), a severidade é máxima
                    severity = 100.0 
                
                logs.append(f"IFRA Violation: {m['name']} ({skin_exposure:.1%} > {limit:.1%})")
                
        return {"is_legal": is_legal, "logs": logs}

def business_evaluation(molecules: List[Dict], tech_score: float, neuro_score: float) -> Dict:
    engine = PerfumeBusinessEngine() 
    
    # Normalização de pesos
    total_w = sum(m.get('weight_factor', 1.0) for m in molecules)
    for m in molecules:
        m['formula_pct'] = m.get('weight_factor', 1.0) / total_w if total_w > 0 else 0

    fin = engine.estimate_financials(molecules, tech_score, neuro_score)
    market = engine.calculate_global_fit(molecules)
    ifra = engine.validate_ifra(molecules)
    
    return {
        "financials": fin,
        "market_strategy": market,
        "compliance": ifra,
        "market_tier": fin['market_tier']
    }