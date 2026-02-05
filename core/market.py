import pandas as pd
from typing import List, Dict, Any
import numpy as np


try:
    from infra.database import load_insumos_from_db
except ImportError:
    def load_insumos_from_db(): return {}

DADOS_INSUMOS = {}

try:
    print(" Conectando ao PostgreSQL...")
    DADOS_INSUMOS = load_insumos_from_db()

    if DADOS_INSUMOS:
        print(f" {len(DADOS_INSUMOS)} insumos carregados do Banco de Dados.")
    else:
        raise Exception("Retorno vazio do Banco de Dados")

except Exception as e_db:
    print(f" [MARKET] Falha no Banco de Dados ({e_db}). Tentando CSV local...")

    try:
        df_insumos = pd.read_csv("insumos.csv")
        df_insumos = df_insumos.drop_duplicates(subset=['name'])
        df_insumos = df_insumos.set_index("name")
        DADOS_INSUMOS = df_insumos.to_dict('index')
        print(f"✅ {len(DADOS_INSUMOS)} insumos carregados do CSV.")
    except Exception as e_csv:
        print(f"❌ [MARKET] Erro crítico: Nem DB nem CSV disponíveis. {e_csv}")
        DADOS_INSUMOS = {}


class PerfumeBusinessEngine:
    def __init__(self, dilution: float = 0.20, bottle_ml: int = 100):
        self.dilution = dilution
        self.bottle_ml = bottle_ml

        self.MARKETS = {
            "Ásia": {
                "keywords": ["Relaxamento", "Calma", "Zen", "Frescor", "Natureza"],
                "ideal_coords": (0.6, -0.6),
                "label": "Zen & Wellness",
                "weight": 1.5
            },
            "América Latina": {
                "keywords": ["Energia", "Tropical", "Alegria", "Radiância", "Festa"],
                "ideal_coords": (0.7, 0.7),
                "label": "Energia & Vibração",
                "weight": 1.5
            },
            "Oriente Médio": {
                "keywords": ["Oud", "Especiado", "Madeira", "Intenso", "Mistério"],
                "ideal_coords": (0.5, 0.3),
                "label": "Opulência & Foco",
                "weight": 2.0
            },
            "Europa": {
                "keywords": ["Limpeza", "Clean", "Foco", "Sofisticação"],
                "ideal_coords": (0.4, 0.0),
                "label": "Clean Beauty",
                "weight": 1.0
            },
            "EUA": {
                "keywords": ["Atração", "Sexy", "Confiança", "Power", "Doçura"],
                "ideal_coords": (0.8, 0.5),
                "label": "Power & Sexy",
                "weight": 1.0
            }
        }

    def calculate_global_fit(self, molecules: List[Dict]) -> Dict:
        scores = {k: 0.0 for k in self.MARKETS.keys()}

        total_weight = sum(m.get('weight_factor', 1.0) for m in molecules)
        formula_valence = 0
        formula_arousal = 0

        text_features = []

        for m in molecules:
            db_data = DADOS_INSUMOS.get(m['name'], {})

            val = m.get('russell_valence', db_data.get('russell_valence', 0))
            aro = m.get('russell_arousal', db_data.get('russell_arousal', 0))

            try:
                val = float(val)
                aro = float(aro)
            except:
                val, aro = 0.0, 0.0

            weight = m.get('weight_factor', 1.0)
            formula_valence += val * weight
            formula_arousal += aro * weight

            trad_use = m.get('traditional_use',
                             db_data.get('traditional_use', ''))
            if trad_use:
                text_features.append(str(trad_use).lower())

        if total_weight > 0:
            formula_valence /= total_weight
            formula_arousal /= total_weight

        full_text = " ".join(text_features)

        for region, criteria in self.MARKETS.items():
            score = 0.0

            dist = np.sqrt((formula_valence - criteria['ideal_coords'][0])**2 +
                           (formula_arousal - criteria['ideal_coords'][1])**2)
            vector_score = max(0, 1.0 - dist) * 10.0
            score += vector_score * 0.6

            keyword_matches = sum(
                1 for kw in criteria['keywords'] if kw.lower() in full_text)
            text_score = min(10.0, keyword_matches * 2.0)
            score += text_score * 0.4

            scores[region] = round(score * criteria.get('weight', 1.0), 2)

        final_scores = {k: min(10.0, v) for k, v in scores.items()}
        best_market = max(
            final_scores, key=final_scores.get) if final_scores else "Global"

        return {
            "rankings": final_scores,
            "best": best_market,
            "label": self.MARKETS.get(best_market, {}).get('label', 'General'),
            "coords": {"valence": round(formula_valence, 2), "arousal": round(formula_arousal, 2)}
        }

    def estimate_financials(self, molecules: List[Dict], tech_score: float, neuro_score: float) -> Dict:
        total_essence_g = self.bottle_ml * self.dilution * 0.95
        total_cost = 0.0

        for m in molecules:
            price_kg = DADOS_INSUMOS.get(
                m['name'], {}).get("price_per_kg", 100.0)

            weight = m.get('weight_factor', 1.0)
            total_w = sum(x.get('weight_factor', 1.0) for x in molecules)
            pct = weight / total_w if total_w > 0 else 0

            total_cost += (total_essence_g * pct) * (price_kg / 1000.0)

        multiplier = 5.0
        tier = "Mass Market"

        if tech_score > 7.0:
            multiplier = 10.0
            tier = "Luxury"
        elif tech_score > 9.0:
            multiplier = 15.0
            tier = "Niche"

        if neuro_score > 0.7:
            multiplier += 3.0

        suggested_price = total_cost * multiplier
        profit = suggested_price - total_cost
        margin = (profit / suggested_price) if suggested_price > 0 else 0.0

        return {
            "cost": round(total_cost, 2),
            "price": round(suggested_price, 2),
            "margin_pct": round(margin * 100, 1),
            "market_tier": tier,
            "applied_multiplier": round(multiplier, 1)
        }

    def validate_ifra(self, molecules: List[Dict]) -> Dict:
        logs = []
        is_legal = True
        total_w = sum(m.get('weight_factor', 1.0) for m in molecules)

        for m in molecules:
            weight = m.get('weight_factor', 1.0)
            pct = weight / total_w if total_w > 0 else 0
            skin_exposure = pct * self.dilution

            try:
                limit_raw = DADOS_INSUMOS.get(
                    m['name'], {}).get("ifra_limit", 1.0)
                if isinstance(limit_raw, str) and "%" in limit_raw:
                    limit = float(limit_raw.replace("%", "")) / 100.0
                else:
                    limit = float(limit_raw)
            except (ValueError, TypeError):
                limit = 1.0

            if skin_exposure > limit:
                is_legal = False
                severity = (skin_exposure - limit) / \
                    limit if limit > 0 else 100.0
                logs.append(
                    f"IFRA Violation: {m['name']} ({skin_exposure:.1%} > Limite {limit:.1%})")

        return {"is_legal": is_legal, "logs": logs}


def business_evaluation(molecules: List[Dict], tech_score: float, neuro_score: float) -> Dict:
    engine = PerfumeBusinessEngine()

    market = engine.calculate_global_fit(molecules)

    fin = engine.estimate_financials(molecules, tech_score, neuro_score)

    ifra = engine.validate_ifra(molecules)

    return {
        "financials": fin,
        "market_strategy": market,
        "compliance": ifra,
        "market_tier": fin['market_tier']
    }
