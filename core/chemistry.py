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

    # Tabela de Potência Olfativa (Proxies para OAV)
    POTENCY_MAP = {
        "ultra": 0.005,   # Bombas (Geosmin, Pirazinas)
        "high": 1.0,      # Fortes (Aldeídos, Musks nobres)
        "medium": 10.0,   # Médios (Linalool, Florais)
        "low": 100.0,     # Transparentes (Iso E, Salicilatos)
        "weak": 500.0,    # Fracos (Terpenos cítricos)
        "mute": 1000.0    # Solventes
    }

    NEURO_VECTORS = {
        # --- ESTIMULANTES (ENERGY) ---
        "dopamine": 1.0,       "uplifting": 1.0,    "energy": 1.0,
        "noradrenaline": 1.0,  "norepinephrine": 1.0, "focus": 1.0,
        "alertness": 1.0,      "acetylcholine": 0.8,  "learning": 0.8,
        "stimulant": 1.0,      "awake": 1.0,          "vitality": 1.0,

        # --- CALMANTES (CALM/GABA) ---
        "gaba": -1.0,          "relax": -1.0,         "calm": -1.0,
        "sedative": -1.0,      "sleep": -1.0,         "anxiolytic": -1.0,
        "meditative": -0.9,    "alpha waves": -0.9,   "peace": -0.8,
        "serenity": -0.8,      "zen": -0.8,

        # --- MODULADORES DE HUMOR (MOOD) ---
        "serotonin": -0.5,     "happiness": -0.5,     "joy": -0.5,
        "mood": -0.4,          "balance": -0.3,       "well-being": -0.4,
        "antidepressant": -0.5,
        
        # --- SOCIAIS / CONFORTO (OXYTOCIN/ENDORPHIN) ---
        "oxytocin": 0.1,       "bonding": 0.1,        "trust": 0.1,
        "comfort": 0.2,        "cozy": 0.2,           "hug": 0.2,
        "endorphin": 0.2,      "pleasure": 0.2,       "euphoria": 0.3,
        "sensual": 0.2,        "aphrodisiac": 0.3,    "sexy": 0.3
    }

    def __init__(self):
        pass

    # ======================================================================
    # MAIN EVALUATION
    # ======================================================================

    def evaluate_blend(self, molecules, target_mood=None):
        if not molecules:
            return self._empty_result()

        # 1. Cálculos Físicos Preliminares (VP, LogP)
        vapor_pressures = []
        logps = []
        
        for m in molecules:
            # Pressão de Vapor (Física)
            bp = m.get("boiling_point", 0)
            if bp <= 0: bp = self._estimate_bp(m)
            vp = self._estimate_vapor_pressure(bp)
            vapor_pressures.append(vp)
            
            # LogP (Química/Pele)
            logp = self._get_logp(m)
            logps.append(logp)

        # 2. Longevidade (Modelo Híbrido VP + LogP)
        longevity = self._calculate_longevity(molecules, vapor_pressures, logps)

        # 3. Projeção (Lei de Stevens + OAV)
        projection = self._calculate_projection(molecules, vapor_pressures, logps)

        # 4. Neurociência (Vetores)
        neuro_score, neuro_vectors = self._calculate_neuro_impact(molecules)

        # 5. Estabilidade (Reações Químicas)
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
            # A. Fração Molar
            mol_weight = m.get("molecular_weight", 200)
            mols = m.get("weight_factor", 1.0) / mol_weight
            mol_fraction = mols / total_mols if total_mols > 0 else 0
            
            # B. Pressão de Vapor
            vp = vapor_pressures[i] 
            
            # C. Limiar de Odor (Hard Science vs Knowledge Base)
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
            
            # D. Cálculo OAV
            concentration_in_air = mol_fraction * vp
            oav = concentration_in_air / threshold
            
            # Salvar dados no objeto para o Dashboard usar depois
            m['vp'] = vp
            m['logp'] = logps[i]
            m['oav'] = oav

            total_oav += math.pow(oav, stevens_exponent)

        # Normalização Logarítmica (ajustado para escala 0-10)
        projection = math.log10(total_oav + 1.0) * 3.5 
        
        # Penalidade "Fundo Mudo"
        max_vp = max(vapor_pressures) if vapor_pressures else 0
        if max_vp < 5.0: projection *= 0.7

        return float(np.clip(projection, 1.0, 10.0))

    # ======================================================================
    # FÍSICA: LONGEVIDADE (VP + LogP)
    # ======================================================================

    def _calculate_longevity(self, molecules, vapor_pressures, logps):
        avg_vp = np.mean(vapor_pressures) if vapor_pressures else 100.0
        avg_logp = np.mean(logps) if logps else 2.5
        
        # Volatilidade Inversa (quanto menor VP, maior duração)
        volatility_score = 12.0 / (np.log10(avg_vp + 0.01) + 3.0)
        
        # Fator Pele (LogP > 3.0 "gruda")
        skin_factor = 1.0 + (max(0, avg_logp - 2.5) * 0.4)
        
        # Super Fixadores (Ancoragem Química)
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
        vectors = {'energy': 0.0, 'calm': 0.0, 'mood': 0.0}
        total_weight = sum(m.get("weight_factor", 1.0) for m in molecules)
        if total_weight == 0: return 0.5, vectors

        for m in molecules:
            raw_target = str(m.get("neuro_target", "")).lower()
            weight = m.get("weight_factor", 1.0) / total_weight
            
            target_key = None
            for key in self.NEURO_VECTORS:
                if key in raw_target:
                    target_key = key
                    break
            
            if target_key:
                direction = self.NEURO_VECTORS[target_key]
                if direction > 0.5: vectors['energy'] += weight * direction
                elif direction < -0.3: vectors['calm'] += weight * abs(direction)
                else: vectors['mood'] += weight

        # Sinergia vs Cancelamento
        total_intensity = sum(vectors.values())
        clash_penalty = 0.0
        
        # Conflito: Muita Energia E Muita Calma
        if vectors['energy'] > 0.3 and vectors['calm'] > 0.3:
            overlap = min(vectors['energy'], vectors['calm'])
            mitigation = vectors['mood'] * 1.5 # Moduladores suavizam
            clash_penalty = max(0, overlap - mitigation) * 2.0

        score = (total_intensity * 5.0) - (clash_penalty * 10.0)
        final_score = float(np.clip(0.5 + (score * 0.1), 0.2, 1.0))
        
        return final_score, vectors

    # ======================================================================
    # QUÍMICA: REAÇÕES & ESTABILIDADE
    # ======================================================================

    def _detect_chemical_risks(self, molecules):
        risks = []
        penalty = 0.0
        
        aldehyde_patt = Chem.MolFromSmarts("[CX3H1](=O)[#6]") 
        amine_patt = Chem.MolFromSmarts("[NX3H2]") # Amina primária
        
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
                
                # Exceção técnica: Musk Ketone tem Nitrogênio mas não é Amina Primária reativa
                if mol.HasSubstructMatch(amine_patt) and "Nitro" not in m.get("olfactive_family", ""):
                    has_amine = True
                    amines_found.append(m.get("name"))
            except: pass

        # Regra Base de Schiff
        if has_aldehyde and has_amine:
            risks.append(f"⚠️ BASE DE SCHIFF: Reação entre {aldehydes_found[:1]} e {amines_found[:1]}.")
            penalty += 0.4

        # Regra Oxidação Cítrica
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
        delta_H = 88.0 * Tb # Regra de Trouton
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
        # Simplificado para manter o foco nas novidades
        return 5.0 

    def _empty_result(self):
        return {
            "longevity": 0.0, "projection": 0.0, "stability": 0.0,
            "complexity": 0.0, "neuro_score": 0.0, "neuro_vectors": {}
        }