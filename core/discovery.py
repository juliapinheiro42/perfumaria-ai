import numpy as np
import random
import os
import json
import pandas as pd

from rdkit import Chem
from rdkit.Chem import Descriptors

from torch_geometric.loader import DataLoader

from core.evolution import EvolutionEngine
from core.surrogate import BayesianSurrogate, expected_improvement
from core.encoder import FeatureEncoder
from core.chemistry import ChemistryEngine
from core.market import business_evaluation
from core.replay_buffer import ReplayBuffer
from core.trainer import ModelTrainer
from core.presets import ACORDES_PRESETS


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

        # Modo Dupe
        self.target_vector = None
        self.target_price = 100.0

        # Memória de feedback humano
        self.last_human_score = 0.0

        print(f"[INIT] Carregando {csv_path}...")
        try:
            self.df_insumos = pd.read_csv(csv_path)
            self.df_insumos.columns = self.df_insumos.columns.str.strip()

            initial_len = len(self.df_insumos)
            # Limpeza de nomes para garantir matching
            self.df_insumos['name'] = self.df_insumos['name'].astype(str).str.strip()
            self.df_insumos.drop_duplicates(subset=["name"], keep="first", inplace=True)

            if len(self.df_insumos) < initial_len:
                print(f"[INFO] Removidas {initial_len - len(self.df_insumos)} duplicatas.")

            self.insumos_dict = self.df_insumos.set_index("name").to_dict("index")

        except Exception as e:
            print(f"[CRITICAL ERROR] Falha ao ler CSV: {e}")
            self.df_insumos = pd.DataFrame(
                columns=["name", "smiles", "category", "molecular_weight"]
            )
            self.insumos_dict = {}

        self._validate_and_clean_dataset()
        self.evolution = EvolutionEngine(self)

    # ======================================================================

    def set_dupe_target(self, target_molecules, anchors=None):
        print(f"[DUPE MODE] Alvo definido com {len(target_molecules)} moléculas.")

        enriched = [self._enrich_with_rdkit(m) for m in target_molecules]

        # Limpa nomes das âncoras
        self.anchors = [str(a).strip() for a in anchors] if anchors else []
        if self.anchors:
            print(f"[ANCHOR] Ingredientes travados: {self.anchors}")

        self.target_vector = FeatureEncoder.encode_blend(enriched)

        self.target_price = sum(m.get("price_per_kg", 0) for m in target_molecules)
        if self.target_price == 0:
            self.target_price = 150.0

        print(
            f"[DUPE MODE] Assinatura vetorial gerada. "
            f"Custo Máximo Alvo: €{self.target_price:.2f}/kg"
        )

    # ======================================================================

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

    # ======================================================================

    def warmup(self, n_samples=200):
        if self.df_insumos.empty:
            return

        print(f"[WARMUP] Gerando {n_samples} amostras iniciais...")
        count = 0
        attempts = 0

        while count < n_samples and attempts < n_samples * 10:
            attempts += 1

            raw_mols = self._generate_molecules()
            if not raw_mols:
                continue

            molecules = [self._enrich_with_rdkit(m) for m in raw_mols]

            if not self._validate_chemical_synergy(molecules):
                continue

            result = self.evaluate(molecules)
            if result["fitness"] <= 0.01:
                continue

            feature_vec = FeatureEncoder.encode_blend(molecules)
            self.surrogate.add_observation(feature_vec, result["fitness"])

            graphs = FeatureEncoder.encode_graphs(molecules)
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

        # --- CORREÇÃO DE MEMÓRIA: Limpa resultados anteriores ---
        self.discoveries = [] 
        best_fitness = 0.0

        for i in range(rounds):

            # Evolução guiada por feedback humano
            if self.discoveries and self.last_human_score >= 6.0:
                # print(f"   -> [EVOLUÇÃO] Refinando fórmula anterior...")
                last_best = self.discoveries[-1]["molecules"]
                raw_mols = self._mutate_formula(last_best)
            else:
                raw_mols = self._generate_molecules()

            if not raw_mols:
                continue

            molecules = [self._enrich_with_rdkit(m) for m in raw_mols]

            feature_vec = FeatureEncoder.encode_blend(molecules)
            result = self.evaluate(molecules)

            # Se foi invalidado (ex: falta de âncora), pula
            if result["fitness"] <= 0.01:
                continue

            self.surrogate.add_observation(feature_vec, result["fitness"])

            graphs = FeatureEncoder.encode_graphs(molecules)
            self.buffer.add(graphs, result["fitness"], weight=1.0)

            if self.trainer.maybe_retrain(self.buffer):
                print(f"{i:02d} | 🔁 GNN re-treinada automaticamente")

            best_fitness = max(best_fitness, result["fitness"])
            self.discoveries.append(result)

            sim_msg = ""
            if "similarity_to_target" in result:
                sim_msg = f"| Sim {result['similarity_to_target']:.2f}"

            chem_msg = f"| Fix {result['chemistry']['longevity']:.1f}h"

            # print(f"{i:02d} | Fit {result['fitness']:.3f} | AI {result['ai_score']:.3f} {sim_msg} {chem_msg}")

        return self.discoveries

    # ======================================================================
    # GERAÇÃO DE MOLÉCULAS (CORRIGIDO)
    # ======================================================================

    def _generate_molecules(self, strategy=None):
        if self.df_insumos.empty: return []
        
        molecules = []
        current_names = set()
        
        # 1. INJEÇÃO FORÇADA DE ÂNCORAS
        # Esta função agora SÓ gera, nunca retorna erro.
        if self.anchors:
            for anchor_name in self.anchors:
                clean_name = str(anchor_name).strip().lower()
                match_key = None
                
                # Busca robusta (Case Insensitive)
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
                    # Apenas avisa, não quebra o código
                    print(f"[AVISO] Âncora '{anchor_name}' não encontrada no estoque.")

        # 2. PREENCHIMENTO
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

        # Identifica índices seguros (não-âncoras)
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
                # Evita duplicatas
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
    # AVALIAÇÃO (CORRIGIDO)
    # ======================================================================

    def evaluate(self, molecules):
        # Validação básica de tipo
        if not molecules or not isinstance(molecules, list):
            return self._invalid_result(molecules)
            
        if not all("rdkit" in m for m in molecules):
            return self._invalid_result(molecules)

        if len(molecules) < 2:
            return self._invalid_result(molecules)

        # --- REGRA DE OURO: VALIDAÇÃO DE ÂNCORAS ---
        # Aqui é onde penalizamos se a âncora sumiu
        if self.anchors:
            names_in_blend = [str(m.get('name', '')).strip().lower() for m in molecules]
            for a in self.anchors:
                clean_anchor = str(a).strip().lower()
                if clean_anchor not in names_in_blend:
                    # Faltou âncora = Fórmula Inválida
                    return self._invalid_result(molecules)

        graphs = FeatureEncoder.encode_graphs(molecules)
        if not graphs:
            return self._invalid_result(molecules)

        loader = DataLoader(graphs, batch_size=len(graphs))
        batch = next(iter(loader))

        try:
            pred = self.model.predict(batch)
            ai_score = float(pred[0][0]) if pred is not None else 0.0
        except:
            ai_score = 0.5

        chem = self.chemistry.evaluate_blend(molecules)
        market = business_evaluation(
            molecules,
            chem["longevity"],
            chem["projection"],
            chem["technology_viability"],
        )

        similarity = 0.0
        fitness = 0.0

        if self.target_vector is not None:
            candidate_vector = FeatureEncoder.encode_blend(molecules)

            dot = np.dot(candidate_vector, self.target_vector)
            na = np.linalg.norm(candidate_vector)
            nb = np.linalg.norm(self.target_vector)

            if na > 0 and nb > 0:
                similarity = dot / (na * nb)

            cost_kg = (
                sum(m.get("price_per_kg", 0) for m in molecules)
                / max(len(molecules), 1)
            )

            price_score = (
                1.0
                if cost_kg <= self.target_price
                else (self.target_price / cost_kg)
            )

            if similarity < 0.01:
                # Fallback se não houver similaridade
                fitness = (
                    (chem["longevity"] / 10.0) * 0.4
                    + ai_score * 0.4
                    + price_score * 0.2
                )
            else:
                fitness = (
                    similarity * 0.6
                    + price_score * 0.2
                    + (chem["longevity"] / 10.0) * 0.2
                )

        else:
            balance = (
                np.sqrt(chem["projection"] * chem["longevity"] + 0.1) / 5
            )
            chemistry_score = np.clip(balance, 0, 1)
            fitness = (
                (0.7 * chemistry_score + 0.3 * market["margin"])
                * self._diversity_penalty(molecules)
            )

        return {
            "fitness": float(np.clip(fitness, 0, 1.5)),
            "ai_score": ai_score,
            "similarity_to_target": float(similarity),
            "chemistry": chem,
            "market": {
                **market,
                "cost_per_kg": sum(
                    m.get("price_per_kg", 0) for m in molecules
                ) / len(molecules),
            },
            "molecules": molecules,
        }

    # ======================================================================

    def _validate_chemical_synergy(self, molecules):
        if not molecules:
            return False
        categories = {m.get("category") for m in molecules}
        return len(categories) >= 2

    # ======================================================================

    def _row_to_molecule(self, row):
        return {
            "name": row.get("name"),
            "molecular_weight": row.get("molecular_weight", 150),
            "polarity": row.get("polarity", 2.0),
            "category": row.get("category", "Heart"),
            "smiles": row.get("smiles"),
            "price_per_kg": row.get("price_per_kg", 100),
            "ifra_limit": row.get("ifra_limit", 1.0),
            "is_allergen": row.get("is_allergen", False),
        }

    # ======================================================================

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

    def register_human_feedback(self, discovery_idx, human_score_0_10):
        try:
            if discovery_idx == -1:
                discovery_idx = len(self.discoveries) - 1
            discovery = self.discoveries[discovery_idx]
        except IndexError:
            return

        self.last_human_score = float(human_score_0_10)

        normalized = human_score_0_10 / 10.0
        ai_score = discovery.get("ai_score", 0.5)

        new_fitness = ai_score * 0.3 + normalized * 0.7
        discovery["fitness"] = new_fitness
        discovery["human_score"] = human_score_0_10

        print(
            f"   >>> [FEEDBACK] Nota {human_score_0_10} registrada. "
            f"Treinando IA..."
        )

        graphs = FeatureEncoder.encode_graphs(discovery["molecules"])
        self.buffer.add(graphs, new_fitness, weight=5.0)

        loss = self.trainer.train_step(self.buffer)
        print(f"   >>> [LEARNING] Modelo atualizado. Loss: {loss:.4f}")

        feature_vec = FeatureEncoder.encode_blend(discovery["molecules"])
        self.surrogate.add_observation(feature_vec, new_fitness)

    # ======================================================================

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

    # ======================================================================

    def _diversity_penalty(self, molecules):
        names = [m.get("name", "UNK") for m in molecules]
        ratio = len(set(names)) / max(len(names), 1)
        return ratio if ratio == 1 else ratio * 0.5

    # ======================================================================

    def _invalid_result(self, molecules):
        return {
            "fitness": 0.01,
            "ai_score": 0.0,
            "chemistry": {
                "projection": 0.0,
                "longevity": 0.0,
                "stability": 0.0,
                "technology_viability": 0.0,
            },
            "market": {
                "market": 0.0,
                "margin": 0.0,
                "cost": 0.0,
                "cost_per_kg": 0.0,
            },
            "molecules": molecules,
        }