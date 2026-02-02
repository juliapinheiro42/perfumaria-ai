import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors

class ChemistryEngine:
    def __init__(self):
        # VIP LIST: Ingredients that defy standard physics rules in perfumery.
        # If the AI uses these, we force high longevity/projection stats.
        self.SUPER_FIXADORES = [
            "Ambroxan", "Cetalox", "Timberol", "Evernyl", "Galaxolide", 
            "Musk Ketone", "Iso E Super", "Civet", "Castoreum", "Olibanum Oil",
            "Patchouli Oil", "Vetiver Oil", "Ebanol", "Javanol", "Vertofix"
        ]

    def evaluate_blend(self, molecules):
        if not molecules:
            return {
                "longevity": 0.0, "projection": 0.0, "stability": 0.0, 
                "technology_viability": 0.0, "olfactive_profile": "Vazio"
            }

        bps = []    # Boiling Points
        logps = []  # Hydrophobicity
        mol_weights = []
        
        fixers_count = 0

        for m in molecules:
            name = m.get("name", "")
            
            # --- CRITICAL CORRECTION: VIP CHECK ---
            if name in self.SUPER_FIXADORES:
                fixers_count += 1
                val_bp = 380.0  # Force high BP
                val_logp = 4.5  # Force high hydrophobicity
            else:
                # Normal logic for other ingredients
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
        # 1. LONGEVITY CALCULATION (Fixation)
        # =========================================================
        avg_bp = np.mean(bps)
        
        # Base score derived from average Boiling Point
        # 200C = 0 pts | 350C = 10 pts
        base_score = (avg_bp - 200) / 15.0 
        
        # Bonus for having VIP Fixers
        # If you have 2+ super fixers, you get a massive boost
        elite_bonus = 0.0
        if fixers_count >= 2:
            elite_bonus = 3.5 
        elif fixers_count == 1:
            elite_bonus = 1.5

        # Calculate final longevity
        longevity = base_score + elite_bonus
        
        # Cap at 10.0 but allow "beast mode" detection internally
        longevity = float(np.clip(longevity, 1.0, 10.0))

        # =========================================================
        # 2. PROJECTION CALCULATION (Sillage)
        # =========================================================
        # Complex logic: 
        # Light molecules project fast (Top notes).
        # Diffusive heavy molecules (Ambroxan/Hedione) ALSO project well.
        
        # Volatility factor (Standard projection)
        volatility_score = 1000.0 / avg_bp 
        
        # Diffusive Booster (The "Baccarat" Effect)
        # Some molecules are heavy but fill the room (Diffusive)
        diffusive_boost = 0.0
        names = [m.get("name", "") for m in molecules]
        
        if "Hedione" in names: diffusive_boost += 1.5
        if "Ambroxan" in names: diffusive_boost += 1.5
        if "Iso E Super" in names: diffusive_boost += 1.0
        if "Ethyl Maltol" in names: diffusive_boost += 0.8 # Sugary cloud

        projection = (volatility_score * 2.5) + diffusive_boost
        
        # Penalty if it's too oily/dense without diffusive materials
        avg_logp = np.mean(logps)
        if avg_logp > 5.0 and diffusive_boost < 1.0:
            projection -= 2.0 # Skin scent

        projection = float(np.clip(projection, 1.0, 10.0))

        # =========================================================
        # 3. STABILITY & PROFILE
        # =========================================================
        return {
            "longevity": longevity,
            "projection": projection,
            "stability": self._calculate_stability(molecules),
            "technology_viability": 1.0,
            "olfactive_profile": self._determine_profile(molecules)
        }

    def _estimate_bp(self, molecule):
        mw = molecule.get("molecular_weight", 200)
        logp = 3.0
        if "rdkit" in molecule:
            logp = molecule["rdkit"].get("LogP", 3.0)
        # Estimation formula
        estimated = 100 + (mw * 0.8) + (logp * 10)
        return float(np.clip(estimated, 80, 450))

    def _calculate_stability(self, molecules):
        # Simplified stability check
        unstable_count = 0
        risks = ["Aldehyde", "Citral", "Limonene", "Orange", "Lemon"]
        for m in molecules:
            name = m.get("name", "")
            if any(r in name for r in risks):
                unstable_count += 1
        
        risk_ratio = unstable_count / len(molecules)
        return float(np.clip(1.0 - (risk_ratio * 0.4), 0.4, 1.0))

    def _determine_profile(self, molecules):
        cats = [m.get("category", "Heart") for m in molecules]
        if cats.count("Base") > len(cats) * 0.4: return "Oriental/Amadeirado"
        if cats.count("Top") > len(cats) * 0.4: return "Cítrico/Fresco"
        return "Floral/Equilibrado"