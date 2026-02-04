import pandas as pd
import os

class ComplianceEngine:
    def __init__(self, insumos_path=None):
        # Tenta localizar o CSV automaticamente se não for passado
        if insumos_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            insumos_path = os.path.join(base_dir, 'insumos.csv')

        self.db = self._load_database(insumos_path)
        
        # Limites Hard-Coded (Fallbacks críticos)
        self.IFRA_LIMITS = {
            "Citral": 0.6, "Isoeugenol": 0.11, "Eugenol": 2.5, "Cinnamal": 0.05,
            "Coumarin": 1.6, "Geraniol": 4.7, "Benzyl Benzoate": 4.8, "Oakmoss Absolute": 0.1
        }
        
        self.POTENCY_MASS_LIMITS = {
            "ultra": 0.01, "high": 0.10, "medium": 1.0, "low": 1.0, "weak": 1.0
        }

        # Composição de Naturais para verificação profunda
        self.NATURALS_COMPOSITION = {
            "Lemon Oil": {"Citral": 0.03, "Limonene": 0.65, "Geraniol": 0.01},
            "Bergamot Oil": {"Limonene": 0.40, "Linalyl Acetate": 0.30, "Citral": 0.01, "Bergapten": 0.005},
            "Orange Sweet Oil": {"Limonene": 0.95, "Citral": 0.005},
            "Lavender Oil": {"Linalool": 0.35, "Linalyl Acetate": 0.40, "Geraniol": 0.03, "Coumarin": 0.001},
            "Rose Oil": {"Citronellol": 0.40, "Geraniol": 0.20, "Eugenol": 0.02, "Methyleugenol": 0.01},
            "Ylang Ylang Oil": {"Benzyl Benzoate": 0.08, "Isoeugenol": 0.01, "Geraniol": 0.02},
            "Clove Oil": {"Eugenol": 0.85, "Isoeugenol": 0.01},
            "Cinnamon Bark": {"Cinnamal": 0.70, "Eugenol": 0.05},
            "Oakmoss Absolute": {"Atranol": 0.01, "Oakmoss Absolute": 1.0},
            "Jasmine Absolute": {"Benzyl Benzoate": 0.15, "Indole": 0.02, "Eugenol": 0.01},
            "Tonka Bean Abs": {"Coumarin": 0.90}
        }

    def _load_database(self, path):
        """Carrega o CSV de insumos e prepara para busca."""
        try:
            if not os.path.exists(path):
                print(f"⚠️ Aviso: Banco de insumos não encontrado em {path}")
                return pd.DataFrame()
            
            df = pd.read_csv(path)
            # Normalização básica
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
        
        # 1. Acumulação Química (Explosão de Naturais)
        chemical_totals = {chem: 0.0 for chem in self.IFRA_LIMITS.keys()}
        
        for ingredient in formula_molecules:
            name = ingredient.get('name', '').strip()
            weight = ingredient.get('weight_factor', 1.0)
            
            # Soma direta se for a molécula isolada
            if name in chemical_totals:
                chemical_totals[name] += weight

            # Soma indireta vinda de naturais
            matched_natural = self._find_natural_composition(name)
            if matched_natural:
                for chem_name, fraction in matched_natural.items():
                    if chem_name in chemical_totals:
                        chemical_totals[chem_name] += weight * fraction

        # 2. Verificação IFRA
        for chem, total_amount in chemical_totals.items():
            concentration_pct = (total_amount / total_weight) * 100
            limit = self.IFRA_LIMITS[chem]
            stats[chem] = concentration_pct
            
            if concentration_pct > limit:
                is_safe = False
                msg = f"⛔ IFRA Violation - {chem}: Found {concentration_pct:.3f}% (Limit: {limit}%)"
                violations.append(msg)
                
                # Tenta encontrar quem está causando isso para sugerir troca
                culprits = [m['name'] for m in formula_molecules if m['name'] == chem or (self._find_natural_composition(m['name']) and chem in self._find_natural_composition(m['name']))]
                for culprit in culprits:
                    subs = self.find_substitutes(culprit)
                    if subs:
                        formatted_subs = "\n   ".join([f"-> {s['name']} (Score: {s['score']:.1f}) {s['tags']}" for s in subs])
                        violations.append(f"   💡 Smart Swap for {culprit}:\n   {formatted_subs}")

        # 3. Verificação de Potência
        for m in formula_molecules:
            potency = m.get('odor_potency', 'medium').lower() 
            limit_pct = self.POTENCY_MASS_LIMITS.get(potency, 1.0) * 100 
            
            actual_pct = (m.get('weight_factor', 1.0) / total_weight) * 100
            
            if actual_pct > limit_pct:
                is_safe = False
                violations.append(
                    f"⛔ Potency Overdose: {m.get('name')} ({potency}) is {actual_pct:.2f}% (Max: {limit_pct:.2f}%)"
                )

        if is_safe:
            return True, ["✅ Compliant & Balanced"], stats
        else:
            return False, violations, stats

    def find_substitutes(self, target_name, top_n=3):
        """
        Encontra substitutos baseados em Família Olfativa e Similaridade de Notas.
        Dá bônus para ingredientes biodegradáveis e renováveis.
        """
        if self.db.empty:
            return []

        # Encontra o ingrediente alvo no DB
        target_row = self.db[self.db['name'].str.lower() == target_name.lower()]
        if target_row.empty:
            return []
        
        target_family = target_row.iloc[0]['olfactive_family']
        target_notes = set(target_row.iloc[0]['olfactive_notes'].split())
        
        # Filtra candidatos da mesma família (excluindo o próprio)
        candidates = self.db[
            (self.db['olfactive_family'] == target_family) & 
            (self.db['name'].str.lower() != target_name.lower())
        ].copy()

        if candidates.empty:
            return []

        results = []
        for _, row in candidates.iterrows():
            # 1. Similaridade de Jaccard nas notas
            cand_notes = set(str(row['olfactive_notes']).split())
            intersection = len(target_notes.intersection(cand_notes))
            union = len(target_notes.union(cand_notes))
            similarity = (intersection / union) * 100 if union > 0 else 0
            
            # 2. Bônus Green
            green_bonus = 0
            tags = ""
            if row.get('biodegradability') == True or str(row.get('biodegradability')).lower() == 'true':
                green_bonus += 15
                tags += "🌿"
            if row.get('renewable_source') == True or str(row.get('renewable_source')).lower() == 'true':
                green_bonus += 10
                tags += "♻️"

            # 3. Score Final
            final_score = similarity + green_bonus
            
            results.append({
                'name': row['name'],
                'score': final_score,
                'tags': tags,
                'price': row.get('price_per_kg', 0)
            })

        # Ordena por score e retorna os top N
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_n]

    def _find_natural_composition(self, ingredient_name):
        """Busca composição química de naturais."""
        if ingredient_name in self.NATURALS_COMPOSITION:
            return self.NATURALS_COMPOSITION[ingredient_name]
        
        for key, comp in self.NATURALS_COMPOSITION.items():
            if key in ingredient_name:
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
            
            # Tenta pegar dados do DB se não estiverem na molécula
            db_data = self.db[self.db['name'].str.lower() == m.get('name', '').lower()]
            
            if not db_data.empty:
                is_bio = db_data.iloc[0].get('biodegradability', False)
                is_renew = db_data.iloc[0].get('renewable_source', False)
                carbon = float(db_data.iloc[0].get('carbon_footprint', 10.0))
            else:
                is_bio = m.get('biodegradability', False)
                is_renew = m.get('renewable_source', False)
                carbon = m.get('carbon_footprint', 10.0)

            # Converte strings "True"/"False" do CSV se necessário
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