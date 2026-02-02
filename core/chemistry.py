import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors


class ChemistryEngine:
    def __init__(self):
        self.SUPER_FIXADORES = [
            "Ambroxan", "Cetalox", "Timberol", "Evernyl", "Galaxolide",
            "Musk Ketone", "Iso E Super", "Civet", "Castoreum", "Olibanum Oil",
            "Patchouli Oil", "Vetiver Oil", "Ebanol", "Javanol", "Vertofix"
        ]

    # ======================================================================
    # MAIN EVALUATION
    # ======================================================================

    def evaluate_blend(self, molecules, target_mood=None):
        if not molecules:
            return {
                "longevity": 0.0,
                "projection": 0.0,
                "stability": 0.0,
                "complexity": 0.0,
                "technology_viability": 0.0,
                "olfactive_profile": "Vazio",
                "neuro_score": self._calculate_neuro_score(molecules, target_mood)
            }

        bps = []
        logps = []
        mol_weights = []

        fixers_count = 0

        for m in molecules:
            name = m.get("name", "")

            if name in self.SUPER_FIXADORES:
                fixers_count += 1
                val_bp = 380.0
                val_logp = 4.5
            else:
                val_bp = m.get("boiling_point", 0)
                if val_bp <= 0:
                    val_bp = self._estimate_bp(m)

                val_logp = 3.0
                if "rdkit" in m and "LogP" in m["rdkit"]:
                    val_logp = m["rdkit"]["LogP"]

            bps.append(val_bp)
            logps.append(val_logp)
            mol_weights.append(m.get("molecular_weight", 200))

        # =========================================================
        # 1. LONGEVITY (FIXATION)
        # =========================================================
        avg_bp = np.mean(bps)

        base_score = (avg_bp - 200) / 15.0

        elite_bonus = 0.0
        if fixers_count >= 2:
            elite_bonus = 3.5
        elif fixers_count == 1:
            elite_bonus = 1.5

        longevity = base_score + elite_bonus
        longevity = float(np.clip(longevity, 1.0, 10.0))

        # =========================================================
        # 2. PROJECTION (SILLAGE) - REALISTIC WEIGHTED MODEL
        # =========================================================
        # Projeção é dominada pelas notas de topo (alta volatilidade) e boosters.
        # Notas de base pesadas em excesso "seguram" a fragrância rente à pele (matam a projeção).

        total_weight = sum(m.get("weight_factor", 1.0) for m in molecules)
        weighted_volatility = 0.0

        for m in molecules:
            w = m.get("weight_factor", 1.0) / total_weight
            bp = m.get("boiling_point", 250.0)
            # Volatilidade inversa ao Ponto de Ebulição.
            # BP 100C (Citrus) -> Alta Volatilidade
            # BP 300C (Musk) -> Baixa Volatilidade
            vol = 1000.0 / (bp + 1.0)
            weighted_volatility += vol * w

        # Penalidade por excesso de base ("Muddy")
        heavy_molecules = [m for m in molecules if m.get("boiling_point", 0) > 300]
        muddy_penalty = 0.0
        if len(heavy_molecules) > len(molecules) * 0.5:
            muddy_penalty = 2.0

        # Boosters (Difusivos)
        diffusive_boost = 0.0
        names = [m.get("name", "") for m in molecules]

        boosters = {
            "Hedione": 2.0, "Ambroxan": 1.5, "Iso E Super": 1.2,
            "Ethyl Maltol": 1.0, "Calone": 1.5, "Aldehyde C12": 1.8
        }

        for name, boost in boosters.items():
            if name in names:
                diffusive_boost += boost

        # Cálculo Final: Volatilidade Ponderada * Escala + Boost - Penalidade
        projection = (weighted_volatility * 2.0) + diffusive_boost - muddy_penalty

        # Ajuste Fino para garantir realismo (0-10)
        projection = float(np.clip(projection, 0.5, 10.0))

        # =========================================================
        # 3. FINAL PACKAGE
        # =========================================================
        return {
            "longevity": longevity,
            "projection": projection,
            "stability": self._calculate_stability(molecules),
            "complexity": self._calculate_complexity(molecules),
            "evolution": self._calculate_evolution_score(molecules),
            "technology_viability": 1.0,
            "olfactive_profile": self._determine_profile(molecules)
        }

    # ======================================================================
    # COMPLEXITY
    # ======================================================================

    def _calculate_complexity(self, molecules):
        score = 0.0
        
        for m in molecules:
            tier = int(m.get("complexity_tier", 1))
            
            if tier == 3:
                score += 3.0
            elif tier == 2:
                score += 2.0
            else:
                score += 0.5

        # Bônus de Diversidade (Mantido, pois é excelente)
        unique_ingredients = len(set(m.get("name", "") for m in molecules))
        diversity_bonus = unique_ingredients * 0.3 # Aumentei o peso da diversidade

        return float(np.clip(score + diversity_bonus, 0.0, 10.0))
    
    # ======================================================================
    # BOILING POINT ESTIMATION
    # ======================================================================

    def _estimate_bp(self, molecule):
        mw = molecule.get("molecular_weight", 200)
        logp = 3.0

        if "rdkit" in molecule:
            logp = molecule["rdkit"].get("LogP", 3.0)

        estimated = 100 + (mw * 0.8) + (logp * 10)
        return float(np.clip(estimated, 80, 450))

    # ======================================================================
    # STABILITY
    # ======================================================================

    def _calculate_stability(self, molecules):
        unstable_count = 0
        risks = ["Aldehyde", "Citral", "Limonene", "Orange", "Lemon"]

        for m in molecules:
            name = m.get("name", "")
            if any(r in name for r in risks):
                unstable_count += 1

        risk_ratio = unstable_count / max(len(molecules), 1)
        return float(np.clip(1.0 - (risk_ratio * 0.4), 0.4, 1.0))

    # ======================================================================
    # OLFACTIVE PROFILE
    # ======================================================================

    def _determine_profile(self, molecules):
        cats = [m.get("category", "Heart") for m in molecules]

        if cats.count("Base") > len(cats) * 0.4:
            return "Oriental/Amadeirado"
        if cats.count("Top") > len(cats) * 0.4:
            return "Cítrico/Fresco"
        return "Floral/Equilibrado"

    def _calculate_evolution_score(self, molecules):
        """
        Mede a 'distância' entre a saída (Top) e o fundo (Base).
        Perfumes lineares = Score baixo (Fácil de copiar).
        Perfumes dinâmicos = Score alto (Dificulta o teste de 1 hora).
        """
        top_mols = [m for m in molecules if m.get("category") == "Top"]
        base_mols = [m for m in molecules if m.get("category") == "Base"]

        if not top_mols or not base_mols:
            return 1.0

        avg_mw_top = np.mean([m.get("molecular_weight", 120) for m in top_mols])
        avg_mw_base = np.mean([m.get("molecular_weight", 280) for m in base_mols])
        
        volatility_gap = (avg_mw_base - avg_mw_top) / 25.0

        top_families = set([m.get("olfactive_family", "") for m in top_mols])
        base_families = set([m.get("olfactive_family", "") for m in base_mols])
        
        contrast_bonus = len(top_families.symmetric_difference(base_families)) * 1.2

        final_score = volatility_gap + contrast_bonus
        return float(np.clip(final_score, 1.0, 10.0))
    
    def _calculate_neuro_score(self, molecules, target_mood):
        """
        Calcula se o perfume cumpre a promessa emocional (ex: "Relaxamento").
        Baseado nas colunas 'functional_effect' do CSV.
        """
        if not target_mood or not molecules:
            return 0.0

        hit_count = 0
        synergy_bonus = 0.0
        
        keywords = {
            "calming": ["Relaxation", "GABA", "Alpha Waves", "Meditative"],
            "energy": ["Energy", "Dopamine", "Vitality", "Refreshing"],
            "seduction": ["Sexy", "Oxytocin", "Attraction", "Aphrodisiac"],
            "confidence": ["Confidence", "Serotonin", "Power"]
        }
        
        target_keys = keywords.get(target_mood.lower(), [])
        
        neuro_mechanisms = set()

        for m in molecules:
            effect = m.get("functional_effect", "")
            mechanism = m.get("neuro_target", "")
            
            if any(k in effect for k in target_keys) or any(k in mechanism for k in target_keys):
                hit_count += 1
                neuro_mechanisms.add(mechanism)
                
                if m.get("complexity_tier", 1) == 3:
                    hit_count += 0.5

        synergy_bonus = len(neuro_mechanisms) * 1.0
        
        total_score = hit_count + synergy_bonus
        return float(np.clip(total_score, 0.0, 10.0))