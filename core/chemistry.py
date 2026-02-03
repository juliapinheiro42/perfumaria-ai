import math
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors

class ChemistryEngine:
    
    # ======================================================================
    # CONSTANTES FÍSICO-QUÍMICAS & NEUROCIÊNCIA
    # ======================================================================
    
    SUPER_FIXADORES = [
        "Ambroxan", "Cetalox", "Timberol", "Evernyl", "Galaxolide",
        "Musk Ketone", "Iso E Super", "Civet", "Castoreum", "Olibanum Oil",
        "Patchouli Oil", "Vetiver Oil", "Ebanol", "Javanol", "Vertofix",
        "Ambermax", "Ambrocenide" # Adicionados super ambers modernos
    ]

    POTENCY_MAP = {
        "ultra": 0.005,
        "high": 1.0,
        "medium": 10.0,
        "low": 100.0,
        "weak": 500.0,
        "mute": 1000.0
    }

    RUSSELL_COORDINATES = {
        "Uplifting / Energy": (0.8, 0.7),
        "Calming / Relaxation": (0.6, -0.6),
        "Mood Balance / Joy": (0.8, 0.3),
        "Refreshing / Clarity": (0.4, 0.6),
        "Balancing": (0.5, 0.0),
        "Grounding / Focus": (0.4, -0.4),
        "Indulgence / Comfort": (0.9, -0.2),
        "Attraction / Radiance": (0.7, 0.4),
        "Comfort / Safety": (0.8, -0.3),
        "Meditative / Grounding": (0.5, -0.6),
        "Stimulating / Warmth": (0.7, 0.7),
        "Clean / Focus": (0.5, 0.5),
        "Confidence / Mystery": (0.6, 0.5),
        "Confidence Boost": (0.7, 0.6),
        "Arousing / Sexy": (0.8, 0.4),
        "Natureza / Água": (0.7, 0.1),
        "Fome / Aconchego": (0.8, -0.1),
        "Clareza / Foco": (0.4, 0.7),
        "Relaxamento / Frescor": (0.7, -0.5)
    }

    def __init__(self):
        pass

    # ======================================================================
    # MAIN EVALUATION
    # ======================================================================

    def evaluate_blend(self, molecules, target_mood=None):
        if not molecules:
            return self._empty_result()

        vapor_pressures = []
        logps = []
        
        for m in molecules:
            bp = m.get("boiling_point", 0)
            if bp <= 0: bp = self._estimate_bp(m)
            vp = self._estimate_vapor_pressure(bp)
            vapor_pressures.append(vp)
            
            logp = self._get_logp(m)
            logps.append(logp)

        longevity = self._calculate_longevity(molecules, vapor_pressures, logps)

        projection = self._calculate_projection(molecules, vapor_pressures, logps)

        neuro_score, neuro_vectors = self._calculate_neuro_impact(molecules)

        stability = self._calculate_stability(molecules)

        return {
            "longevity": longevity,
            "projection": projection,
            "stability": stability,
            "complexity": self._calculate_complexity(molecules),
            "evolution": self._calculate_evolution_score(molecules),
            "technology_viability": 1.0,
            "olfactive_profile": self._determine_profile(molecules),
            "neuro_score": neuro_score,
            "neuro_vectors": neuro_vectors # Essencial para o Dashboard
        }

    # ======================================================================
    # FÍSICA: PROJEÇÃO & OAV
    # ======================================================================

    def _calculate_projection(self, molecules, vapor_pressures, logps):
        total_mols = sum(m.get("weight_factor", 1.0) / m.get("molecular_weight", 200) for m in molecules)
        total_oav = 0.0
        stevens_exponent = 0.5 

        for i, m in enumerate(molecules):
            mol_weight = m.get("molecular_weight", 200)
            mols = m.get("weight_factor", 1.0) / mol_weight
            mol_fraction = mols / total_mols if total_mols > 0 else 0
            
            vp = vapor_pressures[i] 
            
            raw_threshold = m.get("odor_threshold_ppb")
            threshold = 50.0 
            
            try:
                if raw_threshold is not None and str(raw_threshold).strip() != "":
                    threshold = float(raw_threshold)
                else:
                    potency_class = str(m.get("odor_potency", "medium")).lower().strip()
                    threshold = self.POTENCY_MAP.get(potency_class, 50.0)
            except:
                threshold = 50.0
            
            if threshold <= 1e-6: threshold = 1e-6
            
            concentration_in_air = mol_fraction * vp
            oav = concentration_in_air / threshold
            
            m['vp'] = vp
            m['logp'] = logps[i]
            m['oav'] = oav

            total_oav += math.pow(oav, stevens_exponent)

        projection = math.log10(total_oav + 1.0) * 3.5 
        
        max_vp = max(vapor_pressures) if vapor_pressures else 0
        if max_vp < 5.0: projection *= 0.7

        return float(np.clip(projection, 1.0, 10.0))

    # ======================================================================
    # FÍSICA: LONGEVIDADE (VP + LogP)
    # ======================================================================

    def _calculate_longevity(self, molecules, vapor_pressures, logps):
        avg_vp = np.mean(vapor_pressures) if vapor_pressures else 100.0
        avg_logp = np.mean(logps) if logps else 2.5
        
        volatility_score = 12.0 / (np.log10(avg_vp + 0.01) + 3.0)
        
        skin_factor = 1.0 + (max(0, avg_logp - 2.5) * 0.4)
        
        fixer_bonus = 0.0
        fixers = [m for m in molecules if m.get("name") in self.SUPER_FIXADORES]
        if len(fixers) >= 2: fixer_bonus = 2.0
        elif len(fixers) == 1: fixer_bonus = 1.0

        raw_longevity = (volatility_score * skin_factor) + fixer_bonus
        return float(np.clip(raw_longevity, 1.0, 10.0))

    # ======================================================================
    # NEUROCIÊNCIA (VETORES)
    # ======================================================================

    def _calculate_neuro_impact(self, molecules):
        v_total = 0
        a_total = 0
        weights = 0
    
        for m in molecules:
            weight = m.get("weight_factor", 1.0)
            
            r_val = m.get("russell_valence")
            r_aro = m.get("russell_arousal")
            
            if r_val is not None and r_aro is not None:
                try:
                    valence = float(r_val)
                    arousal = float(r_aro)
                except ValueError:
                    valence, arousal = 0, 0
            else:
                effect = m.get("traditional_use") or m.get("functional_effect", "")
                coords = self.RUSSELL_COORDINATES.get(effect, (0, 0))
                valence, arousal = coords

            v_total += valence * weight
            a_total += arousal * weight
            weights += weight

        if weights == 0:
            return 0.5, {"valence": 0.0, "arousal": 0.0}

        final_valence = round(v_total / weights, 2)
        final_arousal = round(a_total / weights, 2)
        
        intensity = (final_valence**2 + final_arousal**2)**0.5
        score = float(np.clip(0.5 + (intensity * 0.4), 0.2, 1.0))
        
        return score, {"valence": final_valence, "arousal": final_arousal}
    
    
    # ======================================================================
    # QUÍMICA: REAÇÕES & ESTABILIDADE
    # ======================================================================

    def _detect_chemical_risks(self, molecules):
        risks = []
        penalty = 0.0
        
        aldehyde_patt = Chem.MolFromSmarts("[CX3H1](=O)[#6]") 
        amine_patt = Chem.MolFromSmarts("[NX3H2]")
        
        has_aldehyde = False
        has_amine = False
        aldehydes_found = []
        amines_found = []

        for m in molecules:
            smiles = m.get("smiles", "")
            if not smiles: continue
            try:
                mol = Chem.MolFromSmiles(smiles)
                if not mol: continue
                
                if mol.HasSubstructMatch(aldehyde_patt):
                    has_aldehyde = True
                    aldehydes_found.append(m.get("name"))
                
                if mol.HasSubstructMatch(amine_patt) and "Nitro" not in m.get("olfactive_family", ""):
                    has_amine = True
                    amines_found.append(m.get("name"))
            except: pass

        if has_aldehyde and has_amine:
            risks.append(f"⚠️ BASE DE SCHIFF: Reação entre {aldehydes_found[:1]} e {amines_found[:1]}.")
            penalty += 0.4

        citrus_count = sum(1 for m in molecules if m.get("olfactive_family") == "Cítrico")
        if citrus_count >= 3:
            risks.append("⚠️ RISCO DE OXIDAÇÃO: Alto teor cítrico sem antioxidante.")
            penalty += 0.1

        return risks, penalty

    def _calculate_stability(self, molecules):
        base_stability = 1.0
        risks, penalty = self._detect_chemical_risks(molecules)
        return float(np.clip(base_stability - penalty, 0.2, 1.0))

    # ======================================================================
    # UTILITÁRIOS FÍSICOS
    # ======================================================================

    def _estimate_vapor_pressure(self, bp_celsius):
        if bp_celsius <= 25: return 100000.0
        Tb = bp_celsius + 273.15
        T = 298.15 
        delta_H = 88.0 * Tb
        exponent = - (delta_H / 8.314) * (1/T - 1/Tb)
        return 101325.0 * math.exp(exponent)

    def _estimate_bp(self, molecule):
        mw = molecule.get("molecular_weight", 200)
        logp = self._get_logp(molecule)
        estimated = 100 + (mw * 0.8) + (logp * 10)
        return float(np.clip(estimated, 80, 450))

    def _get_logp(self, molecule):
        if "rdkit" in molecule and "LogP" in molecule["rdkit"]:
            return molecule["rdkit"]["LogP"]
        return molecule.get("polarity", 2.5)

    def _calculate_complexity(self, molecules):
        score = sum(m.get("complexity_tier", 1) for m in molecules) * 0.5
        unique = len(set(m.get("name") for m in molecules))
        return float(np.clip(score + (unique * 0.2), 0.0, 10.0))

    def _determine_profile(self, molecules):
        cats = [m.get("category", "Heart") for m in molecules]
        if cats.count("Base") > len(cats)*0.4: return "Oriental/Amadeirado"
        if cats.count("Top") > len(cats)*0.4: return "Cítrico/Fresco"
        return "Floral/Equilibrado"

    def _calculate_evolution_score(self, molecules):
        return 5.0 

    def _empty_result(self):
        return {
            "longevity": 0.0, "projection": 0.0, "stability": 0.0,
            "complexity": 0.0, "neuro_score": 0.0, "neuro_vectors": {}
        }