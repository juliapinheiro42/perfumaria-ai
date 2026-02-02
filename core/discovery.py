import numpy as np
import random
import os
import json
import uuid
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

# Tenta importar torch_geometric, se falhar usa fallback
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


    def set_dupe_target(self, target_molecules, anchors=None):
        print(f"[DUPE MODE] Alvo definido com {len(target_molecules)} moléculas.")

        enriched = [self._enrich_with_rdkit(m) for m in target_molecules]

        self.anchors = [str(a).strip() for a in anchors] if anchors else []
        if self.anchors:
            print(f"[ANCHOR] Ingredientes travados: {self.anchors}")

        self.target_vector = FeatureEncoder.encode_blend(enriched)

        # Calcula complexidade alvo baseada em tiers
        # Se não tiver 'weight_factor', assume 1.0. Tiers variam de 1 a 3.
        total_w = sum(m.get('weight_factor', 1.0) for m in enriched)
        weighted_tier = sum(
            m.get('complexity_tier', 1) * m.get('weight_factor', 1.0)
            for m in enriched
        )
        self.target_complexity_score = weighted_tier / total_w if total_w > 0 else 1.0

        self.target_price = sum(m.get("price_per_kg", 0) for m in target_molecules)
        if self.target_price == 0:
            self.target_price = 150.0

        print(f"[DUPE MODE] Assinatura vetorial gerada. Tier Alvo: {self.target_complexity_score:.2f}")


    def _validate_and_clean_dataset(self):
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

            # Validações básicas
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
                raw_mols = self._generate_molecules()

            if not raw_mols: continue

            molecules = [self._enrich_with_rdkit(m) for m in raw_mols]
            
            # Pré-filtro para não perder tempo
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
    # GERAÇÃO DE MOLÉCULAS
    # ======================================================================

    def _generate_molecules(self, strategy=None):
        if self.df_insumos.empty: return []
        
        molecules = []
        current_names = set()
        
        # 1. Adiciona Âncoras
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

        # 2. Preenche o restante
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
    # AVALIAÇÃO (CORRIGIDA E ATUALIZADA)
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

        # 1. Predição AI (GNN)
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

        # 2. Avaliação Química
        chem = self.chemistry.evaluate_blend(molecules)
        
        # 3. Avaliação de Mercado
        tech_score = chem.get("complexity", 0.0)
        neuro_score = chem.get("neuro_score", 0.0)
        
        market = business_evaluation(
            molecules,
            tech_score,
            neuro_score
        )

        similarity = 0.0
        fitness = 0.0

        # Normalizações Úteis (0-1)
        norm_proj = chem.get("projection", 0) / 10.0
        norm_long = chem.get("longevity", 0) / 10.0
        norm_stab = chem.get("stability", 0)
        norm_compl = min(tech_score / 10.0, 1.0)

        financials = market.get("financials", {})
        margin_pct = financials.get("margin_pct", 0.0) / 100.0

        if self.target_vector is not None:
            # --- MODO DUPE MATCHING ---
            candidate_vector = FeatureEncoder.encode_blend(molecules)
            dot = np.dot(candidate_vector, self.target_vector)
            na = np.linalg.norm(candidate_vector)
            nb = np.linalg.norm(self.target_vector)
            
            if na > 0 and nb > 0:
                similarity = dot / (na * nb)

            # Preço: Quão perto ou abaixo do alvo?
            cost_kg = financials.get("cost", 9999.0)
            price_score = 1.0 if cost_kg <= self.target_price else (self.target_price / max(cost_kg, 1.0))

            # Dificuldade de Dupe por Tier (NEW)
            # Se o alvo é complexo (tier alto) e o candidato é simples, penaliza.
            # tech_score é a soma (0-10), precisamos da média ponderada (1-3) para comparar com target.

            total_w = sum(m.get('weight_factor', 1.0) for m in molecules)
            candidate_avg_tier = sum(
                m.get('complexity_tier', 1) * m.get('weight_factor', 1.0)
                for m in molecules
            ) / total_w if total_w > 0 else 1.0

            # Agora comparamos Banana com Banana (Escala 1.0 a 3.0)
            tier_diff = abs(candidate_avg_tier - self.target_complexity_score)
            
            # Normaliza diferença (Max diff é 2.0 -> abs(3-1))
            tier_match_score = max(0.0, 1.0 - (tier_diff / 2.0))

            # Fórmula Fitness "Tudo Influencia"
            # Similarity (40%) + Price (20%) + Perf (20%) + TierMatch (20%)
            perf_score = (norm_proj + norm_long + norm_stab) / 3.0

            if similarity < 0.01:
                # Fallback se similaridade for nula
                fitness = (perf_score * 0.5) + (price_score * 0.3) + (tier_match_score * 0.2)
            else:
                fitness = (
                    (similarity * 0.4) +
                    (price_score * 0.2) +
                    (perf_score * 0.2) +
                    (tier_match_score * 0.2)
                )

        else:
            # --- MODO DISCOVERY / LUXURY ---
            # Aqui queremos alta performance, alta margem e alta complexidade (Uniqueness)
            
            # Peso Estratégico L'Oréal: Química + Margem + Exclusividade
            # Projection agora é crucial

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
            # Novos Campos L'Oréal
            "complexity_tier": row.get("complexity_tier", 1),
            "neuro_target": row.get("neuro_target", "Neutral"),
            "functional_effect": row.get("functional_effect", ""),
            "olfactive_family": row.get("olfactive_family", "Floral"),
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

    def register_human_feedback(self, discovery_id, human_score_0_10):
        discovery = None

        # Tenta buscar por ID (String UUID)
        if isinstance(discovery_id, str):
            for d in self.discoveries:
                if d.get("id") == discovery_id:
                    discovery = d
                    break

        # Fallback para índice numérico (Legacy Support)
        elif isinstance(discovery_id, int):
            try:
                idx = len(self.discoveries) - 1 if discovery_id == -1 else discovery_id
                discovery = self.discoveries[idx]
            except IndexError:
                pass

        if not discovery:
            print(f"   >>> [FEEDBACK ERROR] Discovery não encontrada: {discovery_id}")
            return

        self.last_human_score = float(human_score_0_10)

        normalized = human_score_0_10 / 10.0
        ai_score = discovery.get("ai_score", 0.5)

        new_fitness = ai_score * 0.3 + normalized * 0.7
        discovery["fitness"] = new_fitness
        discovery["human_score"] = human_score_0_10

        print(f"   >>> [FEEDBACK] Nota {human_score_0_10} registrada.")

        graphs = FeatureEncoder.encode_graphs(discovery["molecules"])
        if graphs:
            self.buffer.add(graphs, new_fitness, weight=5.0)

        if self.trainer:
            loss = self.trainer.train_step(self.buffer)
            print(f"   >>> [LEARNING] Modelo atualizado. Loss: {loss:.4f}")

        feature_vec = FeatureEncoder.encode_blend(discovery["molecules"])
        self.surrogate.add_observation(feature_vec, new_fitness)


    def save_results(self, folder="results", filename="discoveries.json"):
        def sanitize(obj):
            if isinstance(obj, dict):
                return {k: sanitize(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [sanitize(v) for v in obj]
            if isinstance(obj, (np.integer, int)):
                return int(obj)
            if isinstance(obj, (np.floating, float)):
                return float(obj)
            if isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            return obj

        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, filename)

        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(sanitize(self.discoveries), f, indent=4, ensure_ascii=False)
            print(f"\n[IO] Resultados salvos em {path}")
        except Exception as e:
            print(f"\n[IO ERROR] Falha ao salvar JSON: {e}")


    def _diversity_penalty(self, molecules):
        names = [m.get("name", "UNK") for m in molecules]
        ratio = len(set(names)) / max(len(names), 1)
        return ratio if ratio == 1 else ratio * 0.5


    def _invalid_result(self, molecules):
        # Atualizado para bater com a estrutura do novo PerfumeBusinessEngine e Dashboard
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