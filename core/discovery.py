import numpy as np
import random
import os
import json
import uuid
import pandas as pd
import copy
from rdkit import Chem
from rdkit.Chem import Descriptors

from core.encoder import FeatureEncoder

try:
    from torch_geometric.loader import DataLoader
except ImportError:
    DataLoader = None

from core.evolution import EvolutionEngine
from core.surrogate import BayesianSurrogate
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
            # Limpeza de nomes de colunas e dados
            self.df_insumos.columns = self.df_insumos.columns.str.strip()
            self.df_insumos['name'] = self.df_insumos['name'].astype(str).str.strip()
            
            initial_len = len(self.df_insumos)
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
                # Validação básica de SMILES para evitar erros do RDKit depois
                if not isinstance(row["smiles"], str) or len(row["smiles"]) < 2:
                    valid_mask.append(False)
                    continue
                mol = Chem.MolFromSmiles(row["smiles"])
                valid_mask.append(mol is not None)
            except Exception:
                valid_mask.append(False)

        self.df_insumos = (
            self.df_insumos[valid_mask].copy().reset_index(drop=True)
        )
        print(f"[INIT] {len(self.df_insumos)} insumos válidos carregados e verificados.")
        
    def _generate_structured_formula(self):
        """Gera um perfume baseado em arquitetura (Skeletons) e não aleatoriamente."""
        if self.df_insumos.empty: return []

        molecules = []
        added_names = set()

        skeleton_name = random.choice(list(PERFUME_SKELETONS.keys()))
        accord_names = PERFUME_SKELETONS[skeleton_name]

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

        # Preencher o resto para ter volume
        target_size = random.randint(8, 14)
        attempts = 0
        while len(molecules) < target_size and attempts < 50:
            attempts += 1
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
        
        for name_key, data_values in self.insumos_dict.items():
            name_str = str(name_key)
            if name_str.lower().strip() == target_lower:
                data = data_values.copy()
                data['name'] = name_str
                return pd.Series(data)
                
        return None

    def warmup(self, n_samples=50):
        # Warmup reduzido para não travar o start
        if self.df_insumos.empty: return
        print(f"[WARMUP] Gerando {n_samples} amostras iniciais...")
        # ... (lógica de warmup mantida simples) ...

    # ======================================================================
    # DISCOVER (CORE)
    # ======================================================================

    def discover(self, rounds=50, goal="Discover perfumes", threshold=0.1, initial_seed=None):
        if self.df_insumos.empty:
            print("[ERROR] Dataset vazio. Abortando descoberta.")
            return []

        self.discoveries = [] 
        best_fitness = 0.0

        # --- SEEDING (Injeção da fórmula atual) ---
        if initial_seed:
            seed_result = self.evaluate(initial_seed)
            seed_result['fitness'] = max(seed_result['fitness'], 0.8) 
            self.discoveries.append(seed_result)
            self.last_human_score = 10.0 

        print(f"[DISCOVER] Iniciando loop de {rounds} gerações...")

        for i in range(rounds):
            is_reformulation = self.target_vector is not None
            should_mutate = (self.discoveries and (self.last_human_score >= 6.0 or is_reformulation))

            raw_mols = []
            if should_mutate:
                if is_reformulation:
                    self.discoveries.sort(key=lambda x: x['fitness'], reverse=True)
                last_best = self.discoveries[0]["molecules"] 
                raw_mols = self._mutate_formula(last_best)
            else:
                if random.random() < 0.7:
                    raw_mols = self._generate_structured_formula()
                else:
                    raw_mols = self._generate_molecules()

            if not raw_mols: continue

            molecules = [self._enrich_with_rdkit(m) for m in raw_mols]
            
            if len(molecules) < 3: 
                continue

            feature_vec = FeatureEncoder.encode_blend(molecules)
            result = self.evaluate(molecules)

            # Debug Log para ver se está gerando
            if i % 10 == 0:
                print(f"  > Gen {i}: Fitness {result['fitness']:.4f} | Eco {result['eco_score']:.2f} | Safe? {result['market']['compliance']['legal']}")

            # Relaxamento do filtro: aceita fitness > 0.001
            if result["fitness"] <= 0.001: 
                continue

            self.surrogate.add_observation(feature_vec, result["fitness"])

            graphs = FeatureEncoder.encode_graphs(molecules)
            if graphs:
                self.buffer.add(graphs, result["fitness"], weight=1.0)

            if self.trainer.maybe_retrain(self.buffer):
                print(f"{i:02d} | 🔁 GNN re-treinada automaticamente")

            best_fitness = max(best_fitness, result["fitness"])
            self.discoveries.append(result)

        # === REDE DE SEGURANÇA (FALLBACK) ===
        if not self.discoveries:
            print("⚠️ AVISO: Algoritmo Genético não encontrou candidatos perfeitos. Gerando fallback de emergência.")
            for _ in range(5):
                fallback_mols = self._generate_molecules()
                if fallback_mols:
                    molecules = [self._enrich_with_rdkit(m) for m in fallback_mols]
                    res = self.evaluate(molecules)
                    self.discoveries.append(res)
                    print("✅ Fallback gerado com sucesso.")
                    break

        return self.discoveries

    # ======================================================================
    # FLANKERS (Variações)
    # ======================================================================

    def create_flanker(self, parent_formula, flanker_type="Intense"):
        """Gera uma variação (flanker) mantendo o 'esqueleto' do original."""
        if not parent_formula: return self._invalid_result([])
        
        # Pega a lista de moléculas do objeto pai
        mols_source = parent_formula.get('molecules', parent_formula) if isinstance(parent_formula, dict) else parent_formula
        flanker_molecules = copy.deepcopy(mols_source)
        
        target_families = []
        forbidden_families = []
        
        if "Sport" in flanker_type or "Fresh" in flanker_type:
            target_families = ["Citrus", "Green", "Aquatic", "Aromatic", "Herbal"]
            forbidden_families = ["Gourmand", "Oriental", "Vanilla"]
            mutation_intensity = 0.4
            
        elif "Intense" in flanker_type or "Night" in flanker_type:
            target_families = ["Woody", "Spicy", "Amber", "Leather", "Musk"]
            forbidden_families = ["Citrus", "Fruity", "Green"]
            mutation_intensity = 0.3
            
        else: # Generic / Sweet
            target_families = ["Gourmand", "Fruity", "Floral", "Vanilla"]
            mutation_intensity = 0.25

        dna_molecules = [m for m in flanker_molecules if m.get('category') == 'Heart']
        mutable_molecules = [m for m in flanker_molecules if m.get('category') != 'Heart']
        
        if not mutable_molecules:
            split_idx = int(len(flanker_molecules)*0.5)
            mutable_molecules = flanker_molecules[:split_idx]
            dna_molecules = flanker_molecules[split_idx:]

        new_molecules = []
        
        # Boost no DNA para versões Intense
        dna_boost = 1.3 if "Intense" in flanker_type else 0.9
        for m in dna_molecules:
            m['weight_factor'] *= dna_boost
            new_molecules.append(m)
            
        # Busca substitutos
        possible_replacements = self.df_insumos[
            self.df_insumos['olfactive_family'].isin(target_families)
        ].to_dict('records')
        
        for m in mutable_molecules:
            fam = m.get('olfactive_family', '')
            
            # Se for proibida para este estilo, troca ou reduz drasticamente
            if fam in forbidden_families:
                if possible_replacements and random.random() < 0.8:
                    new_mol = self._row_to_molecule(random.choice(possible_replacements))
                    new_mol['weight_factor'] = m.get('weight_factor', 1.0)
                    new_molecules.append(new_mol)
                else:
                    m['weight_factor'] *= 0.2 # Reduz muito
                    new_molecules.append(m)
            
            # Mutação aleatória
            elif random.random() < mutation_intensity and possible_replacements:
                new_mol = self._row_to_molecule(random.choice(possible_replacements))
                new_mol['weight_factor'] = m.get('weight_factor', 1.0) * random.uniform(0.9, 1.2)
                new_molecules.append(new_mol)
            
            else:
                # Rebalanceamento
                if fam in target_families:
                    m['weight_factor'] *= 1.4
                new_molecules.append(m)

        # Enriquece e avalia
        final_mols = [self._enrich_with_rdkit(m) for m in new_molecules]
        result = self.evaluate(final_mols)
        result['market_tier'] = f"Flanker {flanker_type}"
        return result

    # ======================================================================
    # GERAÇÃO DE MOLÉCULAS (LEGADO)
    # ======================================================================

    def _generate_molecules(self, strategy=None):
        if self.df_insumos.empty: return []
        
        molecules = []
        current_names = set()
        
        # Âncoras
        if self.anchors:
            for anchor_name in self.anchors:
                found = self._fuzzy_find_ingredient(anchor_name)
                if found is not None:
                    molecules.append(self._row_to_molecule(found))
                    current_names.add(found['name'])

        target_size = random.randint(6, 12)
        
        attempts = 0
        while len(molecules) < target_size and attempts < 100:
            attempts += 1
            required = ["Top", "Heart", "Base"]
            cat = random.choices(required, weights=[0.3, 0.3, 0.4], k=1)[0]
            
            pool = self.df_insumos[self.df_insumos["category"] == cat]
            
            if not pool.empty:
                selected = pool.sample(1).iloc[0]
                if selected['name'] not in current_names:
                    molecules.append(self._row_to_molecule(selected))
                    current_names.add(selected['name'])
                
        return molecules

    # ======================================================================
    # MUTAÇÃO GENÉTICA
    # ======================================================================

    def _mutate_formula(self, parent_molecules):
        child = [m.copy() for m in parent_molecules]
        if not child: return []

        mutation_type = random.choice(["swap", "swap", "add", "remove", "rebalance"])

        anchor_clean = [str(a).lower().strip() for a in self.anchors]
        safe_indices = []
        for i, m in enumerate(child):
            m_name = str(m.get('name', '')).lower().strip()
            if m_name not in anchor_clean:
                safe_indices.append(i)

        if mutation_type == "rebalance" and child:
            idx = random.randint(0, len(child)-1)
            child[idx]['weight_factor'] *= random.uniform(0.8, 1.2)

        elif mutation_type == "swap" and safe_indices:
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
                for m in new_mols:
                    existing = [x['name'] for x in child]
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
        if not molecules: return self._invalid_result(molecules)
        
        # 1. Calcula Eco Score
        eco_score, eco_stats = self.compliance.calculate_eco_score(molecules)
        
        # 2. IA Score (Mock ou GNN)
        ai_score = 0.5
        if self.model and DataLoader:
            graphs = FeatureEncoder.encode_graphs(molecules)
            if graphs:
                try:
                    loader = DataLoader(graphs, batch_size=len(graphs))
                    batch = next(iter(loader))
                    pred = self.model.predict(batch)
                    ai_score = float(pred[0][0]) if pred is not None else 0.5
                except:
                    ai_score = 0.5

        # 3. Química e Mercado
        chem = self.chemistry.evaluate_blend(molecules)
        chem["eco_stats"] = eco_stats
        
        is_safe, safety_logs, safety_stats = self.compliance.check_safety(molecules)
        
        safety_penalty = 1.0 if is_safe else 0.4 
        
        tech_score = chem.get("complexity", 0.0)
        neuro_score = chem.get("neuro_score", 0.0)
        
        market = business_evaluation(molecules, tech_score, neuro_score)
        market["compliance"] = {"legal": is_safe, "logs": safety_logs}

        # 4. Cálculo de Fitness
        similarity = 0.0
        
        if self.target_vector is not None:
            # Modo Reformulação: Foco em similaridade + Eco
            candidate_vector = FeatureEncoder.encode_blend(molecules)
            dot = np.dot(candidate_vector, self.target_vector)
            na = np.linalg.norm(candidate_vector)
            nb = np.linalg.norm(self.target_vector)
            
            if na > 0 and nb > 0:
                similarity = dot / (na * nb)

            fitness = (
                (similarity * 0.50) +
                (eco_score * 0.40) + 
                (ai_score * 0.10)
            )
        else:
            # Modo Descoberta
            norm_proj = min(chem.get("projection", 0) / 10.0, 1.0)
            norm_long = min(chem.get("longevity", 0) / 10.0, 1.0)
            
            fitness = (
                (0.30 * norm_proj) +
                (0.20 * norm_long) +
                (0.20 * ai_score) +
                (0.30 * eco_score)
            ) * self._diversity_penalty(molecules)

        fitness = fitness * safety_penalty

        return {
            "id": str(uuid.uuid4()),
            "fitness": float(np.clip(fitness, 0.0, 1.5)),
            "ai_score": ai_score,
            "eco_score": eco_score,
            "similarity_to_target": float(similarity),
            "chemistry": chem,
            "market": market, 
            "molecules": molecules,
            "market_tier": market.get("market_tier", "Mass Market")
        }


    def _validate_chemical_synergy(self, molecules):
        if not molecules: return False
        categories = {m.get("category") for m in molecules}
        return len(categories) >= 2


    def _row_to_molecule(self, row):
        """Mapeia dados do CSV para objeto de molécula com campos estratégicos."""
        return {
            "name": row.get("name"),
            "molecular_weight": float(row.get("molecular_weight", 150)),
            "polarity": float(row.get("polarity", 2.0)),
            "category": row.get("category", "Heart"),
            "smiles": row.get("smiles", ""),
            "price_per_kg": float(row.get("price_per_kg", 100)),
            "ifra_limit": row.get("ifra_limit", 1.0),
            "olfactive_family": row.get("olfactive_family", "Floral"),
            "traditional_use": row.get("traditional_use", ""),
            "russell_valence": float(row.get("russell_valence", 0.0)),
            "russell_arousal": float(row.get("russell_arousal", 0.0)),
            "odor_potency": str(row.get("odor_potency", "medium")).lower().strip(),
            "biodegradability": str(row.get("biodegradability", "False")).lower() == "true",
            "renewable_source": str(row.get("renewable_source", "False")).lower() == "true",
            "carbon_footprint": float(row.get("carbon_footprint", 10.0)),
            "weight_factor": random.uniform(1.0, 5.0)
        }


    def _enrich_with_rdkit(self, molecule):
        if "rdkit" in molecule or not isinstance(molecule.get("smiles"), str):
            return molecule
        
        if not molecule["smiles"]: return molecule

        try:
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
        except Exception:
            # Fallback seguro caso RDKit falhe em smiles estranhos
            pass

        return molecule

    # ======================================================================
    # HUMAN FEEDBACK & TOOLS
    # ======================================================================

    def register_human_feedback(self, discovery_id, feedback_data, custom_mols=None):
        """Registra feedback humano para refinar o modelo GNN."""
        rating = feedback_data.get('rating', 5.0)
        # Normaliza 1-10 para 0-1
        normalized_score = rating / 10.0
        self.last_human_score = rating

        print(f"[FEEDBACK] Nota recebida: {rating}/10. Atualizando memória...")

        # Se o usuário passou moléculas customizadas (ex: alterou no slider), aprendemos com isso
        mols_to_learn = custom_mols
        
        # Se não, tenta usar a última descoberta (fallback simplificado)
        if not mols_to_learn and self.discoveries:
            mols_to_learn = self.discoveries[-1].get('molecules', [])

        if mols_to_learn:
            # Adiciona ao Buffer de Replay
            graphs = FeatureEncoder.encode_graphs(mols_to_learn)
            if graphs:
                # Peso 5.0 para feedback humano (muito mais importante que simulação)
                self.buffer.add(graphs, normalized_score, weight=5.0)
                
            # Se a nota for muito alta ou muito baixa, força um re-treino rápido
            if rating >= 8 or rating <= 2:
                self.trainer.maybe_retrain(self.buffer, force=True)

    def reformulate_green(self, target_molecules, rounds=30):
        if not target_molecules: return []
        
        # Define o vetor alvo para a função de fitness focar em similaridade
        self.target_vector = FeatureEncoder.encode_blend(target_molecules)
        
        original_eco, _ = self.compliance.calculate_eco_score(target_molecules)
        print(f" [REFORMULADOR] Alvo definido. Eco inicial: {original_eco:.2f}. Otimizando para Eco Score...")

        # Roda o discovery com o vetor alvo setado
        results = self.discover(rounds=rounds, goal="Green Reformulation", initial_seed=target_molecules)
        
        # Limpa o alvo para não atrapalhar buscas futuras
        self.target_vector = None
        return results

    def _diversity_penalty(self, molecules):
        names = [m.get("name", "UNK") for m in molecules]
        ratio = len(set(names)) / max(len(names), 1)
        return ratio if ratio == 1 else ratio * 0.5

    def _invalid_result(self, molecules):
        return {
            "id": str(uuid.uuid4()),
            "fitness": 0.0,
            "molecules": molecules,
            "eco_score": 0.0,
            "market": {"compliance": {"legal": False}, "market_tier": "Invalid"},
            "chemistry": {}
        }