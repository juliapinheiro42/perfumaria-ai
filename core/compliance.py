import pandas as pd
import os

class ComplianceEngine:
    def __init__(self, insumos_path=None):
        if insumos_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            insumos_path = os.path.join(base_dir, 'insumos.csv')

        self.db = self._load_database(insumos_path)
        self.dynamic_limits = {}

        if not self.db.empty:
            valid_limits = self.db[self.db['ifra_limit'].notna()].copy()

            def clean_limit(val):
                try:
                    if isinstance(val, (int, float)):
                        return float(val)
                    return float(str(val).replace('%', '').strip())
                except Exception:
                    return 100.0

            for _, row in valid_limits.iterrows():
                limit_val = clean_limit(row['ifra_limit'])
                if limit_val < 100.0:
                    self.dynamic_limits[row['name'].strip()] = limit_val

        # IFRA defaults (keys stored lowercase for consistent comparison)
        self.IFRA_LIMITS = {
            "citral": 0.6, "isoeugenol": 0.11, "eugenol": 2.5, "cinnamal": 0.05,
            "coumarin": 1.6, "geraniol": 4.7, "benzyl benzoate": 4.8, "oakmoss absolute": 0.1
        }

        # unify all limits as lowercase keys
        self.ALL_LIMITS = {**{k.lower(): v for k, v in self.IFRA_LIMITS.items()},
                           **{k.strip().lower(): v for k, v in self.dynamic_limits.items()}}

        self.POTENCY_MASS_LIMITS = {
            "ultra": 0.01, "high": 0.10, "medium": 1.0, "low": 1.0, "weak": 1.0
        }

        self.NATURALS_COMPOSITION = {
            "lemon oil": {"citral": 0.03, "limonene": 0.65, "geraniol": 0.01},
            "bergamot oil": {"limonene": 0.40, "linalyl acetate": 0.30, "citral": 0.01, "bergapten": 0.005},
            "orange sweet oil": {"limonene": 0.95, "citral": 0.005},
            "lavender oil": {"linalool": 0.35, "linalyl acetate": 0.40, "geraniol": 0.03, "coumarin": 0.001},
            "rose oil": {"citronellol": 0.40, "geraniol": 0.20, "eugenol": 0.02, "methyleugenol": 0.01},
            "ylang ylang oil": {"benzyl benzoate": 0.08, "isoeugenol": 0.01, "geraniol": 0.02},
            "clove oil": {"eugenol": 0.85, "isoeugenol": 0.01},
            "cinnamon bark": {"cinnamal": 0.70, "eugenol": 0.05},
            "oakmoss absolute": {"atranol": 0.01, "oakmoss absolute": 1.0},
            "jasmine absolute": {"benzyl benzoate": 0.15, "indole": 0.02, "eugenol": 0.01},
            "tonka bean abs": {"coumarin": 0.90}
        }

    def _load_database(self, path):
        """Carrega o CSV de insumos e prepara para busca."""
        try:
            if not os.path.exists(path):
                print(f"Aviso: Banco de insumos não encontrado em {path}")
                return pd.DataFrame()
            
            df = pd.read_csv(path)
            df['name'] = df['name'].astype(str).str.strip()
            df['olfactive_notes'] = df['olfactive_notes'].astype(str).str.lower()
            df['olfactive_family'] = df['olfactive_family'].astype(str).str.strip()
            return df
        except Exception as e:
            print(f"Erro ao carregar insumos: {e}")
            return pd.DataFrame()

    def check_safety(self, formula_molecules):
        """
        Analisa segurança e sugere correções inteligentes.
        Retorna: (is_safe, report_list, stats_dict)
        """
        total_weight = sum(m.get('weight_factor', 1.0) for m in formula_molecules)
        if total_weight == 0: return True, [], {}

        violations = []
        stats = {}
        is_safe = True

        chemical_totals = {chem: 0.0 for chem in self.ALL_LIMITS.keys()} 
        
        for ingredient in formula_molecules:
            name = ingredient.get('name', '').strip().lower()
            weight = ingredient.get('weight_factor', 1.0)
            
            if name in chemical_totals:
                chemical_totals[name] += weight

            matched_natural = self._find_natural_composition(ingredient.get('name', ''))
            if matched_natural:
                for chem_name, fraction in matched_natural.items():
                    if chem_name in chemical_totals:
                        chemical_totals[chem_name] += weight * fraction

        for chem, total_amount in chemical_totals.items():
            if total_amount == 0:
                continue
            concentration_pct = (total_amount / total_weight) * 100
            limit = self.ALL_LIMITS[chem]
            
            if concentration_pct > limit:
                is_safe = False
                msg = f"IFRA Violation - {chem}: Found {concentration_pct:.3f}% (Limit: {limit}%)"
                violations.append(msg)
                
                culprits = [m.get('name', '') for m in formula_molecules if m.get('name', '').strip().lower() == chem or (self._find_natural_composition(m.get('name', '')) and chem in self._find_natural_composition(m.get('name', '')))]
                for culprit in culprits:
                    subs = self.find_substitutes(culprit)
                    if subs:
                        formatted_subs = "\n   ".join([f"-> {s['name']} (Score: {s['score']:.1f}) {s['tags']}" for s in subs])
                        violations.append(f"   Smart Swap for {culprit}:\n   {formatted_subs}")

        for m in formula_molecules:
            potency = m.get('odor_potency', 'medium').lower() 
            limit_pct = self.POTENCY_MASS_LIMITS.get(potency, 1.0) * 100 
            
            actual_pct = (m.get('weight_factor', 1.0) / total_weight) * 100
            
            if actual_pct > limit_pct:
                is_safe = False
                violations.append(
                    f"Potency Overdose: {m.get('name')} ({potency}) is {actual_pct:.2f}% (Max: {limit_pct:.2f}%)"
                )

        if is_safe:
            return True, ["Compliant & Balanced"], stats
        else:
            return False, violations, stats

    def find_substitutes(self, target_name, top_n=3):
        """
        Encontra substitutos baseados em Família Olfativa e Similaridade de Notas.
        Dá bônus para ingredientes biodegradáveis e renováveis.
        """
        if self.db.empty:
            return []

        target_row = self.db[self.db['name'].str.lower() == target_name.lower()]
        if target_row.empty:
            return []
        
        target_family = target_row.iloc[0]['olfactive_family']
        target_notes = set(target_row.iloc[0]['olfactive_notes'].split())
        
        candidates = self.db[
            (self.db['olfactive_family'] == target_family) & 
            (self.db['name'].str.lower() != target_name.lower())
        ].copy()

        if candidates.empty:
            return []

        results = []
        for _, row in candidates.iterrows():
            cand_notes = set(str(row['olfactive_notes']).split())
            intersection = len(target_notes.intersection(cand_notes))
            union = len(target_notes.union(cand_notes))
            similarity = (intersection / union) * 100 if union > 0 else 0
            
            green_bonus = 0
            tags = ""
            if row.get('biodegradability') == True or str(row.get('biodegradability')).lower() == 'true':
                green_bonus += 15
                tags += " [bio]"
            if row.get('renewable_source') == True or str(row.get('renewable_source')).lower() == 'true':
                green_bonus += 10
                tags += " [renew]"

            final_score = similarity + green_bonus
            
            results.append({
                'name': row['name'],
                'score': final_score,
                'tags': tags,
                'price': row.get('price_per_kg', 0)
            })

        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_n]

    def _find_natural_composition(self, ingredient_name):
        """Busca composição química de naturais."""
        if not ingredient_name:
            return None
        name_lower = ingredient_name.strip().lower()
        if name_lower in self.NATURALS_COMPOSITION:
            return self.NATURALS_COMPOSITION[name_lower]
        
        for key, comp in self.NATURALS_COMPOSITION.items():
            if key in name_lower:
                return comp
        return None

    def calculate_eco_score(self, formula_molecules):
        """
        Calcula o Eco-Score (0.0 a 1.0).
        Biodegradabilidade (40%) + Renovável (30%) + Baixo Carbono (30%).
        """
        total_weight = sum(m.get('weight_factor', 1.0) for m in formula_molecules)
        if total_weight == 0: return 0.0, {}

        bio_mass = 0.0
        renew_mass = 0.0
        weighted_carbon = 0.0

        for m in formula_molecules:
            w = m.get('weight_factor', 1.0)
            
            db_data = self.db[self.db['name'].str.lower() == m.get('name', '').lower()]
            
            if not db_data.empty:
                is_bio = db_data.iloc[0].get('biodegradability', False)
                is_renew = db_data.iloc[0].get('renewable_source', False)
                carbon = float(db_data.iloc[0].get('carbon_footprint', 10.0))
            else:
                is_bio = m.get('biodegradability', False)
                is_renew = m.get('renewable_source', False)
                carbon = m.get('carbon_footprint', 10.0)

            is_bio = str(is_bio).lower() == 'true' if isinstance(is_bio, str) else bool(is_bio)
            is_renew = str(is_renew).lower() == 'true' if isinstance(is_renew, str) else bool(is_renew)

            if is_bio: bio_mass += w
            if is_renew: renew_mass += w
            weighted_carbon += (w * carbon)

        bio_pct = bio_mass / total_weight
        renew_pct = renew_mass / total_weight
        avg_carbon = weighted_carbon / total_weight

        carbon_score = max(0.0, 1.0 - (avg_carbon / 10.0))
        eco_score = (bio_pct * 0.40) + (renew_pct * 0.30) + (carbon_score * 0.30)

        stats = {
            "eco_score": round(eco_score, 3),
            "biodegradable_pct": round(bio_pct * 100, 1),
            "renewable_pct": round(renew_pct * 100, 1),
            "avg_carbon_footprint": round(avg_carbon, 2)
        }

        return eco_score, stats

    def suggest_fix(self, violations, formula_molecules):
        """Mantido para retrocompatibilidade, mas agora o check_safety já enriquece o report."""
        return [v for v in violations if "Smart Swap" in v]