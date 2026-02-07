import pytest
import numpy as np
from unittest.mock import MagicMock
from core.encoder import FeatureEncoder
from infra.models import Ingredient, Molecule, Composition

def test_feature_encoder_load_from_db(mock_session):
    # Setup data
    ing = Ingredient(name="Rose", category="Heart", price=100.0)
    mol = Molecule(smiles="C=CC", molecular_weight=120.0, log_p=1.5)
    comp = Composition(ingredient=ing, molecule=mol)
    mock_session.add_all([ing, mol, comp])
    mock_session.commit()

    encoder = FeatureEncoder(session=mock_session)

    assert encoder.data is not None
    assert len(encoder.data) == 1
    assert encoder.names == ["Rose"]
    assert encoder.vectors is not None
    assert encoder.vectors.shape == (1, 3) # mw, log_p, price

def test_encode_blend(sample_molecules):
    vector = FeatureEncoder.encode_blend(sample_molecules)
    assert isinstance(vector, np.ndarray)
    assert vector.shape == (5,) # mw, polarity, valence, arousal, weight

    # Test empty
    vector_empty = FeatureEncoder.encode_blend([])
    assert np.all(vector_empty == 0)

def test_encode_graphs(sample_molecules):
    graphs = FeatureEncoder.encode_graphs(sample_molecules)
    assert len(graphs) == 1
    # Since we mocked torch_geometric.data.Data, we check if it was called
    # In conftest we set Data = MagicMock
    # So graphs[0] is a MagicMock instance
    assert graphs[0] is not None

def test_get_closest(mock_session):
    # Setup data
    ing1 = Ingredient(name="Rose", category="Heart", price=100.0)
    mol1 = Molecule(smiles="C=CC", molecular_weight=120.0, log_p=1.5)
    comp1 = Composition(ingredient=ing1, molecule=mol1)

    ing2 = Ingredient(name="Jasmine", category="Heart", price=120.0)
    mol2 = Molecule(smiles="C=CC2", molecular_weight=130.0, log_p=1.6)
    comp2 = Composition(ingredient=ing2, molecule=mol2)

    mock_session.add_all([ing1, mol1, comp1, ing2, mol2, comp2])
    mock_session.commit()

    encoder = FeatureEncoder(session=mock_session)

    target_vector = encoder.vectors[0] # Rose vector
    closest = encoder.get_closest(target_vector, n=1)

    assert len(closest) == 1
    assert closest.iloc[0]['name'] == "Rose"
