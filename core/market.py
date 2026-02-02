try:
    df_insumos = pd.read_csv("insumos.csv")
    df_insumos = df_insumos.drop_duplicates(subset=['name'])
    df_insumos = df_insumos.set_index("name")
    DADOS_INSUMOS = df_insumos.to_dict('index')
except Exception as e:
    print(f"Erro ao carregar insumos.csv: {e}")
    DADOS_INSUMOS = {}

class PerfumeBusinessEngine:
    def __init__(self, dilution: float = 0.20, bottle_ml: int = 100):
        self.dilution = dilution 
        self.bottle_ml = bottle_ml

    def estimate_cogs(self, molecules: List[Dict[str, Any]]) -> float:
        total_essence_g = self.bottle_ml * self.dilution * 0.95
        total_cost = 0
        for m in molecules:
            name = m.get("name")
            pct = m.get("formula_pct", 1.0 / len(molecules) if molecules else 0)
            price_kg = DADOS_INSUMOS.get(name, {}).get("price_per_kg", 100.0)
            total_cost += (total_essence_g * pct) * (price_kg / 1000)
        return float(total_cost)

    def validate_ifra(self, molecules: List[Dict[str, Any]]) -> Dict[str, Any]:
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
                logs.append(f"IFRA: {name} excesso")
        return {"is_legal": is_legal, "penalty": min(penalty, 1.0), "logs": logs}

    def classify_market(self, tech_score: float, cost: float, molecules: List[Dict]) -> str:
        has_noble = any(DADOS_INSUMOS.get(m['name'], {}).get('is_noble', False) for m in molecules)
        if tech_score > 0.45 and cost > 60: return "Experimental"
        if tech_score > 0.85 and has_noble: return "Niche"
        if cost > 40: return "Luxury"
        if tech_score > 0.38: return "Premium"
        return "Mass Market"

def business_evaluation(molecules: List[Dict], longevity: float, projection: float, tech_score: float) -> Dict:
    engine = PerfumeBusinessEngine() 
    
    cost = engine.estimate_cogs(molecules)
    ifra = engine.validate_ifra(molecules)
    market = engine.classify_market(tech_score, cost, molecules)
    
    multipliers = {
        "Experimental": 12.0, "Niche": 10.0, "Luxury": 8.0, 
        "Premium": 5.0, "Mass Market": 3.0
    }
    price = cost * multipliers.get(market, 5.0)
    profit = price - cost
    
    return {
        "cost": float(cost),
        "price": float(price),
        "profit": float(profit),
        "margin": profit / price if price > 0 else 0,
        "market": market,
        "ifra_legal": ifra["is_legal"],
        "adjusted_score": max(0, tech_score - ifra["penalty"])
    }

if __name__ == "__main__":
    minha_formula = [
        {"name": "Oakmoss Absolute", "formula_pct": 0.05}, 
        {"name": "Iso E Super", "formula_pct": 0.50}
    ]
    res = business_evaluation(minha_formula, 1.4, 0.9, 0.88)
    print(f"Resultado do Teste: {res}")