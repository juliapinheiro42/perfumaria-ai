import numpy as np
import torch
from torch_geometric.data import Data
from rdkit import Chem, DataStructs
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

class FeatureEncoder:
    # Configurações para o Surrogate (Vetor fixo de 271 posições)
    FP_BITS = 256
    INPUT_SIZE = 15 + FP_BITS 
    
    # Configuração para a GNN (5 características por átomo)
    NODE_FEATURES = 5 

    @staticmethod
    def smiles_to_graph(smiles):
        """Converte um SMILES num objeto Data do PyTorch Geometric."""
        if not isinstance(smiles, str): return None
        mol = Chem.MolFromSmiles(smiles)
        if not mol: return None

        # Características dos Nós (Número Atómico, Grau, Carga, Aromaticidade, Massa)
        nodes = []
        for atom in mol.GetAtoms():
            nodes.append([
                float(atom.GetAtomicNum()),
                float(atom.GetDegree()),
                float(atom.GetFormalCharge()),
                float(atom.GetIsAromatic()),
                float(atom.GetMass() / 100.0)
            ])
        x = torch.tensor(nodes, dtype=torch.float)

        # Arestas (Conexões entre átomos)
        edge_index = []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edge_index.append([i, j])
            edge_index.append([j, i])
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        return Data(x=x, edge_index=edge_index)

    @staticmethod
    def encode_graphs(molecules):
        """Retorna uma lista de grafos para processamento na GNN."""
        graphs = []
        for m in molecules:
            graph = FeatureEncoder.smiles_to_graph(m.get("smiles"))
            if graph:
                graphs.append(graph)
        return graphs

    @staticmethod
    def encode_blend(molecules, fp_bits: int = None) -> np.ndarray:
        """Gera o vetor fixo para o BayesianSurrogate, evitando erros de forma."""
        if fp_bits is None:
            fp_bits = FeatureEncoder.FP_BITS

        mw, pol, bp, mol_mr, rot_bonds, h_donors = [], [], [], [], [], []
        fingerprints = []

        for m in molecules:
            mw.append(m.get("molecular_weight", 150.0))
            pol.append(m.get("polarity", 2.0))
            bp.append(m.get("boiling_point", 200.0))
            
            rd = m.get("rdkit", {})
            mol_mr.append(rd.get("MolMR", 0.0))
            rot_bonds.append(rd.get("RotBonds", 0.0))
            h_donors.append(rd.get("HDonors", 0.0))

            smiles = m.get("smiles")
            mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
            
            if mol:
                gen = GetMorganGenerator(radius=2, fpSize=fp_bits)
                fp = gen.GetFingerprint(mol)
                arr = np.zeros(fp_bits, dtype=np.float32)
                DataStructs.ConvertToNumpyArray(fp, arr)
                fingerprints.append(arr)
            else:
                fingerprints.append(np.zeros(fp_bits, dtype=np.float32))

        numeric_features = np.array([
            np.mean(mw), np.std(mw), np.min(mw), np.max(mw),
            np.mean(pol), np.std(pol),
            np.mean(bp), np.std(bp), np.min(bp), np.max(bp),
            float(len(molecules)),
            np.mean(np.array(mw) / (np.array(bp) + 1.0)),
            np.mean(mol_mr), np.mean(rot_bonds), np.mean(h_donors)
        ], dtype=np.float32)

        return np.concatenate([numeric_features, np.mean(fingerprints, axis=0)])