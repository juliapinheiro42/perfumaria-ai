import numpy as np
import random
import os
import json
import uuid
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors


from core.encoder import FeatureEncoder


try:
    from torch_geometric.loader import DataLoader
except ImportError:
    DataLoader = None

from core.evolution import EvolutionEngine
from core.surrogate import BayesianSurrogate
from core.encoder import FeatureEncoder
from core.chemistry import ChemistryEngine
from core.market import business_evaluation
from core.replay_buffer import ReplayBuffer
from core.trainer import ModelTrainer
from core.presets import ACORDES_LIB, PERFUME_SKELETONS
from core.compliance import ComplianceEngine

class DiscoveryEngine:

    def __init__(self, model, strategy_agent=None, csv_path="insumos.csv"):
        self.model = model
        self.strategy_agent = strategy_agent

        self.chemistry = ChemistryEngine()
        self.buffer = ReplayBuffer()
        self.trainer = ModelTrainer(model)
        self.surrogate = BayesianSurrogate()

        self.discoveries = []
        self.anchors = []

        self.target_vector = None
        self.target_price = 100.0
        self.target_complexity_score = 0.0
        self.compliance = ComplianceEngine()

        self.last_human_score = 0.0

        print(f"[INIT] Carregando {csv_path}...")
        try:
            self.df_insumos = pd.read_csv(csv_path)
            self.df_insumos.columns = self.df_insumos.columns.str.strip()

            initial_len = len(self.df_insumos)
            self.df_insumos['name'] = self.df_insumos['name'].astype(str).str.strip()
            self.df_insumos.drop_duplicates(subset=["name"], keep="first", inplace=True)

            if len(self.df_insumos) < initial_len:
                print(f"[INFO] Removidas {initial_len - len(self.df_insumos)} duplicatas.")

            self.insumos_dict = self.df_insumos.set_index("name").to_dict("index")

        except Exception as e:
            print(f"[CRITICAL ERROR] Falha ao ler CSV: {e}")
            self.df_insumos = pd.DataFrame()
            self.insumos_dict = {}

        self._validate_and_clean_dataset()
        self.evolution = EvolutionEngine(self)

    def _validate_and_clean_dataset(self):
        if self.df_insumos.empty: return
        if "smiles" not in self.df_insumos.columns:
            return

        valid_mask = []
        for _, row in self.df_insumos.iterrows():
            try:
                mol = Chem.MolFromSmiles(row["smiles"])
                valid_mask.append(mol is not None)
            except Exception:
                valid_mask.append(False)

        self.df_insumos = (
            self.df_insumos[valid_mask].copy().reset_index(drop=True)
        )
        print(f"[INIT] {len(self.df_insumos)} insumos válidos carregados.")
        
    def _generate_structured_formula(self):
        """Gera um perfume baseado em arquitetura (Skeletons) e não aleatoriamente."""
        if self.df_insumos.empty: return []

        molecules = []
        added_names = set()

        skeleton_name = random.choice(list(PERFUME_SKELETONS.keys()))
        accord_names = PERFUME_SKELETONS[skeleton_name]
        
        print(f"   >>> Construindo estrutura: {skeleton_name}")

        for acc_name in accord_names:
            accord_data = ACORDES_LIB.get(acc_name)
            if not accord_data: continue

            mols = accord_data["molecules"]
            ratios = accord_data.get("ratios", [1.0]*len(mols))

            for i, mol_name in enumerate(mols):
                found_row = self._fuzzy_find_ingredient(mol_name)
                
                if found_row is not None and found_row['name'] not in added_names:
                    mol_obj = self._row_to_molecule(found_row)
                    
                    base_weight = ratios[i] if i < len(ratios) else 1.0
                    mol_obj['weight_factor'] = base_weight * random.uniform(2.0, 4.0) 
                    
                    mol_obj['accord_origin'] = acc_name 
                    
                    molecules.append(mol_obj)
                    added_names.add(found_row['name'])

        target_size = random.randint(10, 15)
        while len(molecules) < target_size:
            if self.df_insumos.empty: break
            row = self.df_insumos.sample(1).iloc[0]
            if row['name'] not in added_names:
                molecules.append(self._row_to_molecule(row))
                added_names.add(row['name'])

        return molecules

    def _fuzzy_find_ingredient(self, target_name):
        """Tenta achar o ingrediente no dicionário, tolerando pequenas diferenças."""
        target_clean = str(target_name).strip()
        
        if target_clean in self.insumos_dict:
            data = self.insumos_dict[target_clean].copy()
            data['name'] = target_clean
            return pd.Series(data)
            
        target_lower = target_clean.lower()
        
        best_match = None
        shortest_len = 999
        
        for name_key, data_values in self.insumos_dict.items():
            name_str = str(name_key)
            name_lower = name_str.lower().strip()
            
            if name_lower == target_lower:
                data = data_values.copy()
                data['name'] = name_str
                return pd.Series(data)
            
            if target_lower in name_lower:
                if len(name_lower) < shortest_len:
                    shortest_len = len(name_lower)
                    best_match = data_values.copy()
                    best_match['name'] = name_str

        if best_match:
            return pd.Series(best_match)
                
        return None

    def warmup(self, n_samples=200):
        if self.df_insumos.empty: return

        print(f"[WARMUP] Gerando {n_samples} amostras iniciais...")
        count = 0
        attempts = 0

        while count < n_samples and attempts < n_samples * 10:
            attempts += 1

            raw_mols = self._generate_molecules()
            if not raw_mols: continue

            molecules = [self._enrich_with_rdkit(m) for m in raw_mols]

            if len(molecules) < 3: continue
            if not self._validate_chemical_synergy(molecules): continue

            result = self.evaluate(molecules)
            if result["fitness"] <= 0.01: continue

            feature_vec = FeatureEncoder.encode_blend(molecules)
            self.surrogate.add_observation(feature_vec, result["fitness"])

            graphs = FeatureEncoder.encode_graphs(molecules)
            if graphs:
                self.buffer.add(graphs, result["fitness"])

            count += 1
            if count % 50 == 0:
                print(f"  > Gerado {count}/{n_samples}")

        if self.buffer.size() > 0:
            self.trainer.retrain(self.buffer)
            print("[WARMUP] Concluído.")
        else:
            print("[WARMUP] Falhou: nenhum dado válido.")

    # ======================================================================
    # DISCOVER
    # ======================================================================

    def discover(self, rounds=50, goal="Discover perfumes", threshold=0.1):
        if self.df_insumos.empty:
            print("[ERROR] Dataset vazio. Abortando descoberta.")
            return []

        self.discoveries = [] 
        best_fitness = 0.0

        for i in range(rounds):

            if self.discoveries and self.last_human_score >= 6.0:
                last_best = self.discoveries[-1]["molecules"]
                raw_mols = self._mutate_formula(last_best)
            else:
                if random.random() < 0.8:
                    raw_mols = self._generate_structured_formula()
                else:
                    raw_mols = self._generate_molecules()

            if not raw_mols: continue

            molecules = [self._enrich_with_rdkit(m) for m in raw_mols]
            
            if len(molecules) < 3: continue

            feature_vec = FeatureEncoder.encode_blend(molecules)
            result = self.evaluate(molecules)

            if result["fitness"] <= 0.01: continue

            self.surrogate.add_observation(feature_vec, result["fitness"])

            graphs = FeatureEncoder.encode_graphs(molecules)
            if graphs:
                self.buffer.add(graphs, result["fitness"], weight=1.0)

            if self.trainer.maybe_retrain(self.buffer):
                print(f"{i:02d} | 🔁 GNN re-treinada automaticamente")

            best_fitness = max(best_fitness, result["fitness"])
            self.discoveries.append(result)

        return self.discoveries

    # ======================================================================
    # GERAÇÃO DE MOLÉCULAS (LEGADO / FALLBACK)
    # ======================================================================

    def _generate_molecules(self, strategy=None):
        if self.df_insumos.empty: return []
        
        molecules = []
        current_names = set()
        
        if self.anchors:
            for anchor_name in self.anchors:
                clean_name = str(anchor_name).strip().lower()
                match_key = None
                
                for key in self.insumos_dict.keys():
                    if str(key).strip().lower() == clean_name:
                        match_key = key
                        break
                
                if match_key:
                    row = self.insumos_dict[match_key].copy()
                    row['name'] = match_key 
                    molecules.append(self._row_to_molecule(pd.Series(row)))
                    current_names.add(match_key)
                else:
                    print(f"[AVISO] Âncora '{anchor_name}' não encontrada no estoque.")

        target_size = random.randint(7, 12)
        max_price = 800.0 if self.target_vector is not None else 9999.0
        
        while len(molecules) < target_size:
            required = ["Top", "Heart", "Base"]
            cat = random.choices(required, weights=[0.3, 0.3, 0.4], k=1)[0]
            
            pool = self.df_insumos[
                (self.df_insumos["category"] == cat) & 
                (self.df_insumos["price_per_kg"] <= max_price)
            ]
            
            if not pool.empty:
                valid_pool = pool[~pool['name'].isin(current_names)]
                if valid_pool.empty: valid_pool = pool
                
                selected = valid_pool.sample(1).iloc[0]
                molecules.append(self._row_to_molecule(selected))
                current_names.add(selected['name'])
            else:
                break 
                
        return molecules

    # ======================================================================
    # MUTAÇÃO GENÉTICA
    # ======================================================================

    def _mutate_formula(self, parent_molecules):
        child = [m.copy() for m in parent_molecules]

        mutation_type = random.choice(["swap", "swap", "add", "remove"])

        anchor_clean = [str(a).lower().strip() for a in self.anchors]
        safe_indices = []
        for i, m in enumerate(child):
            m_name = str(m.get('name', '')).lower().strip()
            if m_name not in anchor_clean:
                safe_indices.append(i)

        if mutation_type == "swap" and safe_indices:
            idx = random.choice(safe_indices)
            old_mol = child[idx]
            category = old_mol.get("category", "Heart")

            pool = self.df_insumos[self.df_insumos["category"] == category]
            if not pool.empty:
                new_row = pool.sample(1).iloc[0]
                child[idx] = self._row_to_molecule(new_row)

        elif mutation_type == "add" and len(child) < 14:
            new_mols = self._generate_molecules()
            if new_mols:
                existing = {m['name'] for m in child}
                for m in new_mols:
                    if m['name'] not in existing:
                        child.append(m)
                        break

        elif mutation_type == "remove" and safe_indices and len(child) > 4:
            idx = random.choice(safe_indices)
            child.pop(idx)

        return child

    # ======================================================================
    # AVALIAÇÃO
    # ======================================================================

    def evaluate(self, molecules):
        if not molecules or not isinstance(molecules, list):
            return self._invalid_result(molecules)
            
        if not all("rdkit" in m for m in molecules):
            return self._invalid_result(molecules)

        if len(molecules) < 2:
            return self._invalid_result(molecules)

        if self.anchors:
            names_in_blend = [str(m.get('name', '')).strip().lower() for m in molecules]
            for a in self.anchors:
                clean_anchor = str(a).strip().lower()
                if clean_anchor not in names_in_blend:
                    return self._invalid_result(molecules)

        ai_score = 0.5
        if self.model and DataLoader:
            graphs = FeatureEncoder.encode_graphs(molecules)
            if graphs:
                try:
                    loader = DataLoader(graphs, batch_size=len(graphs))
                    batch = next(iter(loader))
                    pred = self.model.predict(batch)
                    ai_score = float(pred[0][0]) if pred is not None else 0.0
                except:
                    ai_score = 0.5

        chem = self.chemistry.evaluate_blend(molecules)
        is_safe, safety_logs, safety_stats = self.compliance.check_safety(molecules)
        safety_penalty = 1.0 if is_safe else 0.00001
        chem["safety_logs"] = safety_logs
        chem["safety_stats"] = safety_stats
        tech_score = chem.get("complexity", 0.0)
        neuro_score = chem.get("neuro_score", 0.0)
        
        market = business_evaluation(
            molecules,
            tech_score,
            neuro_score
        )
        market["compliance"] = {"legal": is_safe, "logs": safety_logs}

        similarity = 0.0
        fitness = 0.0
        fitness = fitness * safety_penalty

        norm_proj = chem.get("projection", 0) / 10.0
        norm_long = chem.get("longevity", 0) / 10.0
        norm_stab = chem.get("stability", 0)
        norm_compl = min(tech_score / 10.0, 1.0)

        financials = market.get("financials", {})
        margin_pct = financials.get("margin_pct", 0.0) / 100.0

        if self.target_vector is not None:
            candidate_vector = FeatureEncoder.encode_blend(molecules)
            dot = np.dot(candidate_vector, self.target_vector)
            na = np.linalg.norm(candidate_vector)
            nb = np.linalg.norm(self.target_vector)
            
            if na > 0 and nb > 0:
                similarity = dot / (na * nb)

            cost_kg = financials.get("cost", 9999.0)
            price_score = 1.0 if cost_kg <= self.target_price else (self.target_price / max(cost_kg, 1.0))

            total_w = sum(m.get('weight_factor', 1.0) for m in molecules)
            candidate_avg_tier = sum(
                m.get('complexity_tier', 1) * m.get('weight_factor', 1.0)
                for m in molecules
            ) / total_w if total_w > 0 else 1.0

            tier_diff = abs(candidate_avg_tier - self.target_complexity_score)
            tier_match_score = max(0.0, 1.0 - (tier_diff / 2.0))

            perf_score = (norm_proj + norm_long + norm_stab) / 3.0

            if similarity < 0.01:
                fitness = (perf_score * 0.5) + (price_score * 0.3) + (tier_match_score * 0.2)
            else:
                fitness = (
                    (similarity * 0.4) +
                    (price_score * 0.2) +
                    (perf_score * 0.2) +
                    (tier_match_score * 0.2)
                )

        else:
            perf_score = (norm_proj * 0.4) + (norm_long * 0.4) + (norm_stab * 0.2)

            fitness = (
                (0.35 * perf_score) +
                (0.25 * margin_pct) +
                (0.25 * norm_compl) +
                (0.15 * ai_score)
            ) * self._diversity_penalty(molecules)

        return {
            "id": str(uuid.uuid4()),
            "fitness": float(np.clip(fitness, 0, 1.5)),
            "ai_score": ai_score,
            "similarity_to_target": float(similarity),
            "chemistry": chem,
            "market": market, 
            "molecules": molecules,
            "market_tier": market.get("market_tier", "Mass Market")
        }


    def _validate_chemical_synergy(self, molecules):
        if not molecules:
            return False
        categories = {m.get("category") for m in molecules}
        return len(categories) >= 2


    def _row_to_molecule(self, row):
        """Mapeia dados do CSV para objeto de molécula com campos estratégicos."""
        return {
            "name": row.get("name"),
            "molecular_weight": row.get("molecular_weight", 150),
            "polarity": row.get("polarity", 2.0),
            "category": row.get("category", "Heart"),
            "smiles": row.get("smiles"),
            "price_per_kg": row.get("price_per_kg", 100),
            "ifra_limit": row.get("ifra_limit", 1.0),
            "is_allergen": row.get("is_allergen", False),
            "complexity_tier": row.get("complexity_tier", 1),
            "olfactive_family": row.get("olfactive_family", "Floral"),
            "traditional_use": row.get("traditional_use", ""),
            "russell_valence": row.get("russell_valence", 0.0),
            "russell_arousal": row.get("russell_arousal", 0.0),
            "evidence_level": row.get("evidence_level", "None"),
            "odor_potency": str(row.get("odor_potency", "medium")).lower().strip(),
            "weight_factor": random.uniform(1.0, 5.0)
        }


    def _enrich_with_rdkit(self, molecule):
        if "rdkit" in molecule or not isinstance(molecule.get("smiles"), str):
            return molecule

        mol = Chem.MolFromSmiles(molecule["smiles"])
        if mol is None:
            return molecule

        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)

        molecule["boiling_point"] = float(
            np.clip(0.5 * mw + 20 * logp + 0.3 * tpsa + 50, 80, 450)
        )

        molecule["rdkit"] = {
            "LogP": logp,
            "TPSA": tpsa,
            "MolMR": Descriptors.MolMR(mol),
            "RotBonds": Descriptors.NumRotatableBonds(mol),
            "HDonors": Descriptors.NumHDonors(mol),
        }

        return molecule

    # ======================================================================
    # FEEDBACK HUMANO
    # ======================================================================

    def register_human_feedback(self, discovery_id, feedback_data, custom_mols=None):
        
        discovery = None
    
        if custom_mols:
            discovery = {
                "molecules": custom_mols,
                "ai_score": 0.5
            }
        else:
            if isinstance(discovery_id, str):
                for d in self.discoveries:
                    if d.get("id") == discovery_id:
                        discovery = d
                        break
            elif isinstance(discovery_id, int):
                try:
                    idx = len(self.discoveries) - 1 if discovery_id == -1 else discovery_id
                    discovery = self.discoveries[idx]
                except IndexError:
                    pass
        if not discovery:
            print(f"   >>> [FEEDBACK ERROR] Discovery não encontrada: {discovery_id}")
            return

        if isinstance(feedback_data, (int, float)):
            hedonic = float(feedback_data)
            technical = 5.0
            creative = 5.0
        else:
            hedonic = float(feedback_data.get("hedonic", 5.0))
            technical = float(feedback_data.get("technical", 5.0))
            creative = float(feedback_data.get("creative", 5.0))

        self.last_human_score = hedonic

        weighted_score = (
            (hedonic / 10.0) * 0.6 + 
            (technical / 10.0) * 0.2 + 
            (creative / 10.0) * 0.2
        )
    
        ai_score = discovery.get("ai_score", 0.5)
        new_fitness = (ai_score * 0.2) + (weighted_score * 0.8)

        if not custom_mols:
            discovery["fitness"] = new_fitness
            discovery["human_feedback_vector"] = feedback_data

        print(f"   >>> [RLHF] Feedback {'(SYNTHETIC)' if custom_mols else ''}: H{hedonic}/T{technical}/C{creative} -> Fitness: {new_fitness:.4f}")

        # --- 1. Treinamento Positivo (O que o humano gostou) ---
        graphs = FeatureEncoder.encode_graphs(discovery["molecules"])
        if graphs:
            prio_weight = 10.0 if custom_mols else 5.0 
            self.buffer.add(graphs, new_fitness, weight=prio_weight)

        # --- 2. [NOVO] Contrastive Learning Injection (Geração de Exemplos Negativos) ---
        # Só geramos negativos se o feedback atual for BOM (ex: > 6.0). 
        # Não adianta gerar "versão piorada" de algo que já é ruim.
        if new_fitness > 0.6: 
            try:
                # Gera variantes 'estragadas' (overdoses) dessa fórmula boa
                negatives = ReplayBuffer.generate_negative_examples(
                    discovery["molecules"], 
                    self.compliance
                )
                
                count_neg = 0
                for bad_formula, bad_fitness in negatives:
                    bad_graphs = FeatureEncoder.encode_graphs(bad_formula)
                    if bad_graphs:
                        # Ensinamos: "Essa variação é PÉSSIMA (fitness ~0)"
                        self.buffer.add(bad_graphs, bad_fitness, weight=2.0)
                        count_neg += 1
                
                if count_neg > 0:
                    print(f"   >>> [TEACHING] Gerados {count_neg} exemplos negativos para contraste.")
            except Exception as e:
                print(f"   >>> [WARNING] Falha ao gerar exemplos negativos: {e}")

        # --- 3. Atualização dos Pesos (Backpropagation) ---
        if self.trainer:
            loss = self.trainer.train_step(self.buffer)
            print(f"   >>> [LEARNING] Pesos neurais ajustados. Loss: {loss:.4f}")

        feature_vec = FeatureEncoder.encode_blend(discovery["molecules"])
        self.surrogate.add_observation(feature_vec, new_fitness)
        
    def save_results(self, folder="results", filename="discoveries.json"):
        def sanitize(obj):
            if isinstance(obj, dict):
                return {str(k): sanitize(v) for k, v in obj.items()}
            
            if isinstance(obj, (list, tuple)):
                return [sanitize(v) for v in obj]
            
            if isinstance(obj, (np.integer, int)):
                return int(obj)
            if isinstance(obj, (np.floating, float)):
                val = float(obj)
                if np.isnan(val) or np.isinf(val):
                    return None 
                return val
                
            if isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            if isinstance(obj, np.ndarray):
                return sanitize(obj.tolist())
            if obj is None:
                return None
                
            return str(obj)

        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, filename)

        try:
            existing_data = []
            if os.path.exists(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        content = f.read().strip()
                        if content:
                            existing_data = json.loads(content)
                            if not isinstance(existing_data, list): existing_data = []
                except Exception:
                    existing_data = []

            clean_current_discoveries = sanitize(self.discoveries)
            
            existing_ids = {d.get("id") for d in existing_data if isinstance(d, dict)}
            count_new = 0
            
            for d in clean_current_discoveries:
                if isinstance(d, dict) and d.get("id") and d.get("id") not in existing_ids:
                    existing_data.append(d)
                    count_new += 1

            with open(path, "w", encoding="utf-8") as f:
                json.dump(existing_data, f, indent=4, ensure_ascii=False)
                
            print(f"\n[IO] Salvos {count_new} novos perfumes em {path}")
            
        except Exception as e:
            print(f"\n[CRITICAL IO ERROR] Detalhe do erro: {type(e).__name__}: {e}")

    def _diversity_penalty(self, molecules):
        names = [m.get("name", "UNK") for m in molecules]
        ratio = len(set(names)) / max(len(names), 1)
        return ratio if ratio == 1 else ratio * 0.5


    def _invalid_result(self, molecules):
        return {
            "fitness": 0.01,
            "ai_score": 0.0,
            "chemistry": {
                "projection": 0.0,
                "longevity": 0.0,
                "stability": 0.0,
                "complexity": 0.0,
                "neuro_score": 0.0,
                "evolution": 0.0,
                "technology_viability": 0.0,
            },
            "market": {
                "financials": {
                    "cost": 0.0, 
                    "price": 0.0, 
                    "margin_pct": 0.0, 
                    "profit": 0.0, 
                    "applied_multiplier": 1.0
                },
                "market_strategy": {"best": "Unknown", "rankings": {}},
                "compliance": {"legal": False, "logs": []},
                "market_tier": "Invalid"
            },
            "molecules": molecules,
            "market_tier": "Invalid"
        }