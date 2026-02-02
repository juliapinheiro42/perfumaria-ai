import pandas as pd
from typing import List, Dict, Union

class IFRAValidator:
    def __init__(self, insumos_data: Union[Dict, pd.DataFrame]):
        """
        insumos_data: Pode ser um DataFrame ou Dicionário mapeando o nome 
        da molécula para seus dados técnicos.
        """
        if isinstance(insumos_data, pd.DataFrame):
            self.insumos_data = insumos_data.to_dict('index')
        else:
            self.insumos_data = insumos_data

    def validate_formula(self, formula_composition: List[Dict], dilution: float = 0.20) -> Dict:
        """
        formula_composition: Lista de dicts [{'name': 'MolX', 'concentration': 0.10}, ...]
        dilution: Concentração da essência no álcool (0.20 = Eau de Parfum)
        """
        warnings = []
        total_penalty = 0.0
        is_legal = True

        for item in formula_composition:
            mol_id = item.get('name')
            # Concentração da molécula dentro da essência pura (0.0 a 1.0)
            pct_in_oil = item.get('concentration', 0.0)
            
            # Recupera o limite IFRA (Default 100% se não listado)
            molecule_spec = self.insumos_data.get(mol_id, {})
            limit = molecule_spec.get("ifra_limit", 1.0)
            
            # O cálculo real: Concentração na essência * Diluição final
            # Ex: 10% de Oakmoss na essência * 20% de diluição = 2% na pele.
            final_skin_concentration = pct_in_oil * dilution

            if final_skin_concentration > limit:
                is_legal = False
                # Calcula o quão acima do limite a fórmula está
                severity = (final_skin_concentration - limit) / limit
                
                # Penalidade dinâmica para o AI Score (min 0.1, max 1.0)
                total_penalty += min(1.0, 0.2 + (severity * 0.1))
                
                warnings.append({
                    "molecule": mol_id,
                    "ifra_limit_skin": f"{limit * 100:.4f}%",
                    "current_skin_exposure": f"{final_skin_concentration * 100:.4f}%",
                    "max_allowed_in_essence": f"{(limit / dilution) * 100:.2f}%",
                    "severity_index": round(severity, 2),
                    "status": "ILLEGAL"
                })

        return {
            "is_legal": is_legal,
            "ai_score_penalty": round(min(1.0, total_penalty), 4),
            "dilution_used": f"{dilution * 100}%",
            "warnings": warnings
        }

# --- Exemplo de Uso Prático ---
if __name__ == "__main__":
    # Simulando sua base de insumos
    base_insumos = {
        "Oakmoss Absolute": {"ifra_limit": 0.001},  # Muito restrito
        "Rose Oxide": {"ifra_limit": 0.02},
        "Iso E Super": {"ifra_limit": 0.21}
    }

    validator = IFRAValidator(base_insumos)

    # Simulando uma fórmula gótica pesada
    minha_formula = [
        {"name": "Oakmoss Absolute", "concentration": 0.05}, # 5% na essência
        {"name": "Iso E Super", "concentration": 0.30}      # 30% na essência
    ]

    resultado = validator.validate_formula(minha_formula, dilution=0.20)
    
    print(f"Status Legal: {resultado['is_legal']}")
    print(f"Penalidade IA: {resultado['ai_score_penalty']}")
    for w in resultado['warnings']:
        print(f" -> {w['molecule']}: Severidade {w['severity_index']}")