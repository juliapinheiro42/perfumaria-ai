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

    def evaluate_blend(self, molecules):
        if not molecules:
            return {
                "longevity": 0.0,
                "projection": 0.0,
                "stability": 0.0,
                "complexity": 0.0,
                "technology_viability": 0.0,
                "olfactive_profile": "Vazio"
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
        # 2. PROJECTION (SILLAGE)
        # =========================================================
        volatility_score = 1000.0 / avg_bp

        diffusive_boost = 0.0
        names = [m.get("name", "") for m in molecules]

        if "Hedione" in names:
            diffusive_boost += 1.5
        if "Ambroxan" in names:
            diffusive_boost += 1.5
        if "Iso E Super" in names:
            diffusive_boost += 1.0
        if "Ethyl Maltol" in names:
            diffusive_boost += 0.8

        projection = (volatility_score * 2.5) + diffusive_boost

        avg_logp = np.mean(logps)
        if avg_logp > 5.0 and diffusive_boost < 1.0:
            projection -= 2.0

        projection = float(np.clip(projection, 1.0, 10.0))

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

        complex_materials = [
            "Bergamot Oil FCF", "Orange Sweet Oil", "Lemon Oil",
            "Petitgrain Oil", "Lavandin Oil", "Ylang Ylang Extra",
            "Olibanum Oil", "Patchouli Oil", "Oakmoss Absolute",
            "Blackcurrant Bud Absolute", "Labdanum Resinoid"
        ]

        tech_materials = [
            "Ambrocenide", "Javanol", "Spirambrene", "Akigalawood"
        ]

        for m in molecules:
            name = m.get("name", "")
            if name in complex_materials:
                score += 3.0
            elif name in tech_materials:
                score += 2.0
            else:
                score += 0.1

        unique_ingredients = len(set(m.get("name", "") for m in molecules))
        diversity_bonus = unique_ingredients * 0.2

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

        top_families = set([m.get("sub_category", "") for m in top_mols])
        base_families = set([m.get("sub_category", "") for m in base_mols])
        
        contrast_bonus = len(top_families.symmetric_difference(base_families)) * 1.2

        final_score = volatility_gap + contrast_bonus
        return float(np.clip(final_score, 1.0, 10.0))