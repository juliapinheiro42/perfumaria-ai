# core/compliance.py
import pandas as pd

class ComplianceEngine:
    def __init__(self):
        self.IFRA_LIMITS = {
            "Citral": 0.6,
            "Isoeugenol": 0.11,
            "Eugenol": 2.5,
            "Cinnamal": 0.05,
            "Coumarin": 1.6,
            "Geraniol": 4.7,
            "Benzyl Benzoate": 4.8,
            "Oakmoss Absolute": 0.1
        }

        self.NATURALS_COMPOSITION = {
            "Lemon Oil": {"Citral": 0.03, "Limonene": 0.65, "Geraniol": 0.01},
            "Bergamot Oil": {"Limonene": 0.40, "Linalyl Acetate": 0.30, "Citral": 0.01, "Bergapten": 0.005},
            "Orange Sweet Oil": {"Limonene": 0.95, "Citral": 0.005},
            "Lavender Oil": {"Linalool": 0.35, "Linalyl Acetate": 0.40, "Geraniol": 0.03, "Coumarin": 0.001},
            "Rose Oil": {"Citronellol": 0.40, "Geraniol": 0.20, "Eugenol": 0.02, "Methyleugenol": 0.01},
            "Ylang Ylang Oil": {"Benzyl Benzoate": 0.08, "Isoeugenol": 0.01, "Geraniol": 0.02},
            "Clove Oil": {"Eugenol": 0.85, "Isoeugenol": 0.01},
            "Cinnamon Bark": {"Cinnamal": 0.70, "Eugenol": 0.05},
            "Oakmoss Absolute": {"Atranol": 0.01, "Oakmoss Absolute": 1.0}, # O próprio material é restrito
            "Jasmine Absolute": {"Benzyl Benzoate": 0.15, "Indole": 0.02, "Eugenol": 0.01},
            "Tonka Bean Abs": {"Coumarin": 0.90}
        }

    def check_safety(self, formula_molecules):
        """
        Analisa a fórmula completa, explode os naturais e valida limites acumulados.
        Retorna: (is_safe: bool, report: list, stats: dict)
        """
        total_weight = sum(m.get('weight_factor', 1.0) for m in formula_molecules)
        if total_weight == 0: return True, [], {}

        chemical_totals = {chem: 0.0 for chem in self.IFRA_LIMITS.keys()}
        
        for ingredient in formula_molecules:
            name = ingredient.get('name', '').strip()
            weight = ingredient.get('weight_factor', 1.0)
            
            if name in chemical_totals:
                chemical_totals[name] += weight

            matched_natural = self._find_natural_composition(name)
            
            if matched_natural:
                for chem_name, fraction in matched_natural.items():
                    if chem_name in chemical_totals:
                        chemical_totals[chem_name] += weight * fraction

        violations = []
        stats = {}
        is_safe = True

        for chem, total_amount in chemical_totals.items():
            concentration_pct = (total_amount / total_weight) * 100
            limit = self.IFRA_LIMITS[chem]
            
            stats[chem] = concentration_pct
            
            if concentration_pct > limit:
                is_safe = False
                violations.append(
                    f"⛔ {chem}: Found {concentration_pct:.3f}% (Limit: {limit}%)"
                )

        if is_safe:
            return True, ["✅ IFRA Compliant"], stats
        else:
            return False, violations, stats

    def _find_natural_composition(self, ingredient_name):
        """Tenta encontrar a composição química buscando no dicionário."""
        if ingredient_name in self.NATURALS_COMPOSITION:
            return self.NATURALS_COMPOSITION[ingredient_name]
        
        for key, comp in self.NATURALS_COMPOSITION.items():
            if key in ingredient_name:
                return comp
        return None

    def suggest_fix(self, violations, formula_molecules):
        """(Opcional) Sugere onde reduzir para corrigir."""
        suggestions = []
        for v in violations:
            chem_name = v.split(":")[0].replace("⛔", "").strip()
            culprits = []
            for m in formula_molecules:
                name = m.get('name')
                if name == chem_name:
                    culprits.append(name)
                nat_data = self._find_natural_composition(name)
                if nat_data and chem_name in nat_data:
                    culprits.append(f"{name} (contains {chem_name})")
            
            if culprits:
                suggestions.append(f"To fix {chem_name}, reduce: {', '.join(culprits)}")
        return suggestions