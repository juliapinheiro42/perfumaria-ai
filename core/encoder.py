import pandas as pd
import numpy as np
from sqlalchemy.orm import Session
from infra.models import Ingredient, Molecule, Composition
import torch
from torch_geometric.data import Data


class FeatureEncoder:
    def __init__(self, session: Session = None):
        self.char_to_int = {}
        self.int_to_char = {}
        self.vocab_size = 0
        self.max_length = 0
        self.data = None
        self.vectors = None

        if session:
            self.load_from_db(session)

    def load_from_db(self, session: Session):
        """
        Carrega dados do PostgreSQL e prepara os vetores para a IA.
        Recria a estrutura 'flat' que existia no CSV para facilitar o uso nos modelos.
        """
        print(" Conectando ao Banco de Dados para carregar Encoder...")

        results = session.query(
            Ingredient.name,
            Ingredient.category,
            Ingredient.price,
            Molecule.smiles,
            Molecule.molecular_weight,
            Molecule.log_p
        ).join(Composition, Ingredient.id == Composition.ingredient_id)\
         .join(Molecule, Composition.molecule_id == Molecule.id)\
         .all()

        if not results:
            raise ValueError(
                " O banco de dados está vazio! Rode o 'migrate_db.py' primeiro.")

        self.data = pd.DataFrame(results, columns=[
            'name', 'category', 'price', 'smiles', 'molecular_weight', 'log_p'
        ])

        self.names = self.data['name'].tolist()
        self.name_to_idx = {name: i for i, name in enumerate(self.names)}
        self.idx_to_name = {i: name for i, name in enumerate(self.names)}

        features = self.data[['molecular_weight', 'log_p', 'price']].fillna(0)

        features = self.data[['molecular_weight', 'log_p', 'price']].fillna(0)

        denominator = features.max() - features.min()
        denominator = denominator.replace(0, 1.0)
        self.vectors = (features - features.min()) / denominator
        self.vectors = self.vectors.values.astype(np.float32)

        print(
            f" Encoder carregado com {len(self.data)} ingredientes do Banco de Dados.")

    @staticmethod
    def encode_blend(molecules):
        """
        Converte uma lista de moléculas em um vetor numérico único (média das features).
        Usado para comparar similaridade entre perfumes.
        """
        if not molecules:
            return np.zeros(5, dtype=np.float32)

        vectors = []
        for m in molecules:
            v = [
                float(m.get('molecular_weight', 0)),
                float(m.get('polarity', 0)),
                float(m.get('russell_valence', 0)),
                float(m.get('russell_arousal', 0)),
                float(m.get('weight_factor', 1.0))
            ]
            vectors.append(v)

        return np.mean(vectors, axis=0).astype(np.float32)

    @staticmethod
    def encode_graphs(molecules):
        """
        Converte moléculas em grafos para a Rede Neural (GNN).
        """
        graphs = []
        for m in molecules:

            node_features = [
                float(m.get('molecular_weight', 0)),
                float(m.get('polarity', 0)),
                float(m.get('russell_valence', 0)),
                float(m.get('russell_arousal', 0)),
                1.0
            ]

            x = torch.tensor([node_features], dtype=torch.float)
            edge_index = torch.tensor([[0], [0]], dtype=torch.long)

            data = Data(x=x, edge_index=edge_index)
            graphs.append(data)

        return graphs

    def get_closest(self, target_vector, n=5):
        """Encontra os N ingredientes mais parecidos quimicamente."""
        dists = np.linalg.norm(self.vectors - target_vector, axis=1)
        nearest_indices = dists.argsort()[:n]
        return self.data.iloc[nearest_indices]
