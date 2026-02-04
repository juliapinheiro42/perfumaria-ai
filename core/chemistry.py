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
        "Ambermax", "Ambrocenide"
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

    TIME_STEPS = [0.1, 0.5, 1.0, 3.0, 6.0, 10.0]

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
            if bp <= 0:
                bp = self._estimate_bp(m)

            vp = self._estimate_vapor_pressure(bp)
            vapor_pressures.append(vp)

            logp = self._get_logp(m)
            logps.append(logp)

        longevity = self._calculate_longevity(molecules, vapor_pressures, logps)
        projection = self._calculate_projection(molecules, vapor_pressures, logps)
        neuro_score, neuro_vectors = self._calculate_neuro_impact(molecules)
        stability = self._calculate_stability(molecules)

        temporal_projection, family_evolution_data = (
            self._calculate_temporal_evaporation(molecules, vapor_pressures)
        )

        evolution = self._calculate_evolution_score(
            molecules, temporal_projection
        )

        return {
            "longevity": longevity,
            "projection": projection,
            "stability": stability,
            "complexity": self._calculate_complexity(molecules),
            "evolution": evolution,
            "temporal_curve": temporal_projection,
            "temporal_families": family_evolution_data,
            "technology_viability": 1.0,
            "olfactive_profile": self._determine_profile(molecules),
            "neuro_score": neuro_score,
            "neuro_vectors": neuro_vectors,
            "eco_stats": {
                "biodegradable_pct": 0,
                "renewable_pct": 0,
                "avg_carbon_footprint": 10
            }
        }

    # ======================================================================
    # FÍSICA: PROJEÇÃO & OAV
    # ======================================================================

    def _calculate_projection(self, molecules, vapor_pressures, logps):
        total_mols = sum(
            m.get("weight_factor", 1.0) / max(m.get("molecular_weight", 200), 1)
            for m in molecules
        )

        total_oav = 0.0
        stevens_exponent = 0.5

        for i, m in enumerate(molecules):
            mol_weight = max(m.get("molecular_weight", 200), 1)
            mols = m.get("weight_factor", 1.0) / mol_weight
            mol_fraction = mols / total_mols if total_mols > 0 else 0.0

            vp = vapor_pressures[i]

            raw_threshold = m.get("odor_threshold_ppb")
            threshold = 50.0

            try:
                if raw_threshold not in (None, ""):
                    threshold = float(raw_threshold)
                else:
                    potency_class = str(
                        m.get("odor_potency", "medium")
                    ).lower().strip()
                    threshold = self.POTENCY_MAP.get(
                        potency_class, 50.0
                    )
            except Exception:
                threshold = 50.0

            threshold = max(threshold, 1e-6)

            concentration_in_air = mol_fraction * vp
            oav = concentration_in_air / threshold

            m["vp"] = vp
            m["logp"] = logps[i]
            m["oav"] = oav

            total_oav += math.pow(oav, stevens_exponent)

        projection = math.log10(total_oav + 1.0) * 3.5

        if vapor_pressures and max(vapor_pressures) < 5.0:
            projection *= 0.7

        return float(np.clip(projection, 1.0, 10.0))

    # ======================================================================
    # FÍSICA: LONGEVIDADE (VP + LogP)
    # ======================================================================

    def _calculate_longevity(self, molecules, vapor_pressures, logps):
        avg_vp = np.mean(vapor_pressures) if vapor_pressures else 100.0
        avg_logp = np.mean(logps) if logps else 2.5

        volatility_score = 12.0 / (np.log10(avg_vp + 0.01) + 3.0)
        skin_factor = 1.0 + max(0.0, avg_logp - 2.5) * 0.4

        fixers = [
            m for m in molecules
            if m.get("name") in self.SUPER_FIXADORES
        ]

        fixer_bonus = 2.0 if len(fixers) >= 2 else 1.0 if len(fixers) == 1 else 0.0

        raw_longevity = (volatility_score * skin_factor) + fixer_bonus
        return float(np.clip(raw_longevity, 1.0, 10.0))

    # ======================================================================
    # NEUROCIÊNCIA (VETORES)
    # ======================================================================

    def _calculate_neuro_impact(self, molecules):
        v_total = 0.0
        a_total = 0.0
        weights = 0.0

        for m in molecules:
            weight = m.get("weight_factor", 1.0)

            r_val = m.get("russell_valence")
            r_aro = m.get("russell_arousal")

            try:
                valence = float(r_val)
                arousal = float(r_aro)
            except (TypeError, ValueError):
                effect = (
                    m.get("traditional_use")
                    or m.get("functional_effect", "")
                )
                valence, arousal = self.RUSSELL_COORDINATES.get(
                    effect, (0.0, 0.0)
                )

            v_total += valence * weight
            a_total += arousal * weight
            weights += weight

        if weights == 0:
            return 0.5, {"valence": 0.0, "arousal": 0.0}

        final_valence = round(v_total / weights, 2)
        final_arousal = round(a_total / weights, 2)

        intensity = math.sqrt(
            final_valence ** 2 + final_arousal ** 2
        )

        score = float(np.clip(0.5 + intensity * 0.4, 0.2, 1.0))

        return score, {
            "valence": final_valence,
            "arousal": final_arousal
        }

    # ======================================================================
    # QUÍMICA: REAÇÕES & ESTABILIDADE
    # ======================================================================

    def _detect_chemical_risks(self, molecules):
        risks = []
        penalty = 0.0

        aldehyde_patt = Chem.MolFromSmarts("[CX3H1](=O)[#6]")
        amine_patt = Chem.MolFromSmarts("[NX3H2]")

        aldehydes_found = []
        amines_found = []

        for m in molecules:
            smiles = m.get("smiles")
            if not smiles:
                continue

            mol = Chem.MolFromSmiles(smiles)
            if not mol:
                continue

            if mol.HasSubstructMatch(aldehyde_patt):
                aldehydes_found.append(m.get("name"))

            if mol.HasSubstructMatch(amine_patt) and "Nitro" not in str(
                m.get("olfactive_family", "")
            ):
                amines_found.append(m.get("name"))

        if aldehydes_found and amines_found:
            risks.append(
                f"⚠️ BASE DE SCHIFF: Reação entre "
                f"{aldehydes_found[:1]} e {amines_found[:1]}"
            )
            penalty += 0.4

        citrus_count = sum(
            1 for m in molecules
            if m.get("olfactive_family") == "Cítrico"
        )

        if citrus_count >= 3:
            risks.append(
                "⚠️ RISCO DE OXIDAÇÃO: Alto teor cítrico sem antioxidante."
            )
            penalty += 0.1

        return risks, penalty

    def _calculate_stability(self, molecules):
        base_stability = 1.0
        _, penalty = self._detect_chemical_risks(molecules)
        return float(np.clip(base_stability - penalty, 0.2, 1.0))

    # ======================================================================
    # UTILITÁRIOS FÍSICOS
    # ======================================================================

    def _estimate_vapor_pressure(self, bp_celsius):
        if bp_celsius <= 25:
            return 100000.0

        Tb = bp_celsius + 273.15
        T = 298.15
        delta_H = 88.0 * Tb

        exponent = -(
            delta_H / 8.314
        ) * (1.0 / T - 1.0 / Tb)

        return 101325.0 * math.exp(exponent)

    def _estimate_bp(self, molecule):
        mw = molecule.get("molecular_weight", 200)
        logp = self._get_logp(molecule)

        estimated = 100.0 + (mw * 0.8) + (logp * 10.0)
        return float(np.clip(estimated, 80.0, 450.0))

    def _get_logp(self, molecule):
        if "rdkit" in molecule and "LogP" in molecule["rdkit"]:
            return molecule["rdkit"]["LogP"]
        return molecule.get("polarity", 2.5)

    def _calculate_complexity(self, molecules):
        score = sum(
            m.get("complexity_tier", 1) for m in molecules
        ) * 0.5

        unique = len({m.get("name") for m in molecules})
        return float(np.clip(score + unique * 0.2, 0.0, 10.0))

    def _determine_profile(self, molecules):
        cats = [m.get("category", "Heart") for m in molecules]

        if cats.count("Base") > len(cats) * 0.4:
            return "Oriental/Amadeirado"

        if cats.count("Top") > len(cats) * 0.4:
            return "Cítrico/Fresco"

        return "Floral/Equilibrado"

    # ======================================================================
    # EVOLUÇÃO TEMPORAL
    # ======================================================================

    def _calculate_temporal_evaporation(self, molecules, vapor_pressures):
        temporal_projection = []
        family_evolution_data = []

        fixative_mass = sum(
            m.get("weight_factor", 0.0)
            for m in molecules
            if m.get("name") in self.SUPER_FIXADORES
        )

        retention_factor = 1.0 / (1.0 + fixative_mass * 2.0)

        for t in self.TIME_STEPS:
            total_oav_at_t = 0.0
            family_oavs = {}

            for i, m in enumerate(molecules):
                vp = vapor_pressures[i]
                family = m.get("olfactive_family", "Outros")

                effective_vp = (
                    vp * max(0.4, retention_factor)
                    if vp > 10.0 else vp
                )

                remaining_pct = math.exp(
                    -(effective_vp * 0.05) * t
                )

                threshold = self.POTENCY_MAP.get(
                    str(m.get("odor_potency", "medium")).lower(),
                    50.0
                )

                conc_t = m.get("weight_factor", 1.0) * remaining_pct
                oav_t = (conc_t * effective_vp) / max(threshold, 1e-6)

                intensity = math.log10(oav_t + 1.0)
                family_oavs[family] = (
                    family_oavs.get(family, 0.0) + intensity
                )

                total_oav_at_t += oav_t

            for fam, intensity in family_oavs.items():
                if intensity > 0.1:
                    family_evolution_data.append({
                        "Time": t,
                        "Family": fam,
                        "Intensity": round(intensity, 2)
                    })

            proj_t = math.log10(total_oav_at_t + 1.0) * 3.5
            temporal_projection.append(round(float(proj_t), 2))

        return temporal_projection, family_evolution_data

    def _calculate_evolution_score(self, molecules, temporal_projection):
        if not temporal_projection:
            return 5.0

        initial = temporal_projection[0]
        mid = temporal_projection[2] if len(temporal_projection) > 2 else initial

        drop_off = initial - mid
        stability_score = 10.0 - (drop_off * 1.5)

        return float(np.clip(stability_score, 1.0, 10.0))

    def _empty_result(self):
        return {
            "longevity": 0.0,
            "projection": 0.0,
            "stability": 0.0,
            "complexity": 0.0,
            "neuro_score": 0.0,
            "neuro_vectors": {}
        }
