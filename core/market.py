import pandas as pd
from typing import List, Dict, Any

# =========================================================
# CARREGAMENTO DE DADOS GLOBAIS
# =========================================================
try:
    # Carrega o banco de dados de insumos na memória
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
        
        # Mapeamento Estratégico: Região -> Neuro-Target (Biologia do Consumidor)
        self.MARKETS = {
            "Ásia": {
                "target": "GABA Agonist", 
                "label": "Zen & Wellness", 
                "weight": 1.5
            },
            "América Latina": {
                "target": "Dopamine Boost", 
                "label": "Energia & Vibração", 
                "weight": 1.5
            },
            "Oriente Médio": {
                "target": "Cortisol Balance", 
                "label": "Opulência & Foco", 
                "weight": 2.0
            },
            "Europa": {
                "target": "GABA Agonist", 
                "label": "Clean Beauty", 
                "weight": 1.0
            },
            "EUA": {
                "target": "Dopamine Boost", 
                "label": "Power & Sexy", 
                "weight": 1.0
            }
        }

    def calculate_global_fit(self, molecules: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analisa a fórmula e retorna o ranking de mercados onde ela performaria melhor
        baseado na concentração de Neuro-Targets (funcionalidade biológica).
        """
        scores = {k: 0.0 for k in self.MARKETS.keys()}
        total_mols = len(molecules) if molecules else 1
        
        for m in molecules:
            # Recupera dados do insumo do CSV carregado
            m_data = DADOS_INSUMOS.get(m['name'], {})
            target = m_data.get('neuro_target', 'Neutral')
            
            # Pontuação baseada no Fit com o Driver do Mercado
            for region, criteria in self.MARKETS.items():
                if target == criteria['target']:
                    scores[region] += 2.0 * criteria.get('weight', 1.0)
        
        # Normalização para escala 0-10
        # Fator de ajuste para garantir que boas fórmulas cheguem perto de 10
        max_possible = total_mols * 2.5 
        normalized = {k: min(10.0, (v / max_possible) * 10) for k, v in scores.items()}
        
        # Identifica o vencedor
        best_market = max(normalized, key=normalized.get)
        
        return {
            "rankings": normalized,
            "best": best_market,
            "label": self.MARKETS[best_market]['label']
        }

    def estimate_financials(self, molecules: List[Dict[str, Any]], market_tier: str = "Luxury") -> Dict[str, float]:
        """
        Calcula o COGS (Custo do Produto Vendido), Preço Sugerido e Margem.
        """
        total_essence_g = self.bottle_ml * self.dilution * 0.95 # Densidade média 0.95
        total_cost = 0.0
        
        for m in molecules:
            name = m.get("name")
            # Se não houver percentual definido, assume divisão igualitária
            pct = m.get("formula_pct", 1.0 / len(molecules) if molecules else 0)
            
            # Preço por kg do CSV (default 100 se não achar)
            price_kg = DADOS_INSUMOS.get(name, {}).get("price_per_kg", 100.0)
            
            # Custo = (Gramas usadas) * (Preço por grama)
            total_cost += (total_essence_g * pct) * (price_kg / 1000.0)
            
        # Multiplicadores de Markup por Tier de Mercado
        multipliers = {
            "Experimental": 12.0, 
            "Niche": 10.0, 
            "Luxury": 8.0, 
            "Premium": 5.0, 
            "Mass Market": 3.0
        }
        multiplier = multipliers.get(market_tier, 5.0)
        
        suggested_price = total_cost * multiplier
        profit = suggested_price - total_cost
        margin = (profit / suggested_price) if suggested_price > 0 else 0.0
        
        return {
            "cost": round(total_cost, 2),
            "price": round(suggested_price, 2),
            "margin": round(margin, 2),
            "profit": round(profit, 2)
        }

    def validate_ifra(self, molecules: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Verifica conformidade com limites de segurança (IFRA).
        """
        penalty = 0.0
        logs = []
        is_legal = True
        
        for m in molecules:
            name = m.get("name")
            pct_in_oil = m.get("formula_pct", 1.0 / len(molecules) if molecules else 0)
            skin_exposure = pct_in_oil * self.dilution
            
            limit = DADOS_INSUMOS.get(name, {}).get("ifra_limit", 1.0)
            
            if skin_exposure > limit:
                is_legal = False
                severity = (skin_exposure - limit) / limit
                penalty += 0.2 * (1 + severity)
                logs.append(f"IFRA Violation: {name} ({skin_exposure:.1%} > {limit:.1%})")
                
        return {
            "is_legal": is_legal, 
            "penalty": min(penalty, 1.0), 
            "logs": logs
        }

    def classify_tier(self, tech_score: float, cost: float, molecules: List[Dict]) -> str:
        """
        Define o Tier do produto (Nicho, Luxo, Mass) baseado na complexidade e custo.
        """
        # Verifica se usa ingredientes "Nobres" (Tier 3 - Naturais Complexos)
        has_noble = any(DADOS_INSUMOS.get(m['name'], {}).get('complexity_tier', 1) >= 3 for m in molecules)
        
        if tech_score > 0.85 and has_noble: return "Niche"
        if cost > 40: return "Luxury" # Custo alto de essência
        if tech_score > 0.60: return "Premium"
        return "Mass Market"

# Wrapper para compatibilidade caso rode isoladamente
def business_evaluation(molecules: List[Dict], longevity: float, projection: float, tech_score: float) -> Dict:
    engine = PerfumeBusinessEngine() 
    
    # 1. Estimativa de Custo Inicial
    temp_cost = engine.estimate_financials(molecules, "Mass Market")['cost']
    
    # 2. Classificação do Tier
    tier = engine.classify_tier(tech_score, temp_cost, molecules)
    
    # 3. Financials Reais (com markup correto)
    fin = engine.estimate_financials(molecules, tier)
    
    # 4. IFRA & Market Fit
    ifra = engine.validate_ifra(molecules)
    market_fit = engine.calculate_global_fit(molecules)
    
    return {
        "cost": fin['cost'],
        "price": fin['price'],
        "margin": fin['margin'],
        "market_tier": tier,
        "best_region": market_fit['best'],
        "ifra_legal": ifra["is_legal"],
        "logs": ifra["logs"]
    }