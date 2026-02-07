import pytest
import sys
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
from core.discovery import DiscoveryEngine
from infra.models import Ingredient, Molecule, Composition, Psychophysics, Sustainability

@pytest.fixture
def mock_dependencies():
    # We patch classes used in DiscoveryEngine.__init__ and methods
    with patch("core.discovery.ChemistryEngine") as MockChem, \
         patch("core.discovery.ReplayBuffer") as MockBuffer, \
         patch("core.discovery.ModelTrainer") as MockTrainer, \
         patch("core.discovery.BayesianSurrogate") as MockSurrogate, \
         patch("core.discovery.ComplianceEngine") as MockCompliance, \
         patch("core.discovery.EvolutionEngine") as MockEvo, \
         patch("core.discovery.business_evaluation") as mock_biz_eval, \
         patch("core.discovery.FeatureEncoder") as MockEncoder:

        # Setup ChemistryEngine return values
        chem_instance = MockChem.return_value
        chem_instance.evaluate_blend.return_value = {
            "longevity": 8.0, "projection": 7.0, "stability": 0.9,
            "neuro_score": 0.6, "complexity": 5.0, "temporal_curve": [5.0, 4.0, 3.0],
            "eco_stats": {}
        }

        # Setup ComplianceEngine return values
        comp_instance = MockCompliance.return_value
        comp_instance.calculate_eco_score.return_value = (0.8, {})
        comp_instance.check_safety.return_value = (True, [], {})

        # Setup business_evaluation
        mock_biz_eval.return_value = {
            "financials": {"market_tier": "Mass Market"},
            "market_strategy": {},
            "compliance": {"is_legal": True, "logs": []},
            "market_tier": "Mass Market"
        }

        # Setup FeatureEncoder
        MockEncoder.encode_blend.return_value = np.array([1.0, 2.0, 3.0])
        MockEncoder.encode_graphs.return_value = [MagicMock()]

        yield {
            "chem": chem_instance,
            "comp": comp_instance,
            "biz": mock_biz_eval,
            "encoder": MockEncoder
        }

def test_discovery_init(mock_session, mock_dependencies):
    model = MagicMock()
    engine = DiscoveryEngine(model, session=mock_session)
    assert engine.session == mock_session
    assert engine.df_insumos is not None # Even if empty initially

def test_discovery_load_db(mock_session, mock_dependencies):
    # Add data to session
    ing = Ingredient(name="Rose", category="Heart", price=100.0, olfactive_family="Floral", traditional_use="Love")
    psycho = Psychophysics(odor_potency="high", russell_valence=0.8, russell_arousal=0.5)
    ing.psychophysics = psycho

    mol = Molecule(smiles="C=CC", molecular_weight=120.0, log_p=1.5)
    comp = Composition(ingredient=ing, molecule=mol)
    mock_session.add_all([ing, mol, comp])
    mock_session.commit()

    model = MagicMock()
    engine = DiscoveryEngine(model, session=mock_session)

    assert not engine.df_insumos.empty
    assert "Rose" in engine.insumos_dict

def test_evaluate(mock_session, mock_dependencies, sample_molecules):
    model = MagicMock()
    model.predict.return_value = [[0.8]] # Mock prediction

    engine = DiscoveryEngine(model, session=mock_session)

    # Configure DataLoader mock by patching core.discovery.DataLoader
    with patch("core.discovery.DataLoader") as MockDataLoader:
        loader_instance = MagicMock()
        loader_instance.__iter__.return_value = iter([MagicMock()])
        MockDataLoader.return_value = loader_instance

        result = engine.evaluate(sample_molecules)

        assert "fitness" in result
        assert result["fitness"] > 0

        # Check if mocks were called
        mock_dependencies["chem"].evaluate_blend.assert_called()
        mock_dependencies["comp"].calculate_eco_score.assert_called()
        mock_dependencies["biz"].assert_called()

def test_discover_loop(mock_session, mock_dependencies):
    # Setup data
    ing = Ingredient(name="Rose", category="Heart", price=100.0, olfactive_family="Floral", traditional_use="Love")
    psycho = Psychophysics(odor_potency="high", russell_valence=0.8, russell_arousal=0.5)
    ing.psychophysics = psycho

    mol = Molecule(smiles="C=CC", molecular_weight=120.0, log_p=1.5)
    comp = Composition(ingredient=ing, molecule=mol)
    mock_session.add_all([ing, mol, comp])
    mock_session.commit()

    model = MagicMock()
    engine = DiscoveryEngine(model, session=mock_session)

    # Mock _generate_molecules to return something valid
    # Also Mock _enrich_with_rdkit because we rely on it

    with patch.object(engine, '_generate_molecules', return_value=[{"name": "Rose", "smiles": "C=CC", "weight_factor": 1.0}]):
         with patch.object(engine, 'evaluate') as mock_evaluate:
            mock_evaluate.return_value = {
                "id": "1", "fitness": 0.9, "molecules": [{"name": "Rose"}],
                "eco_score": 0.5, "ai_score": 0.5, "chemistry": {}, "market": {}
            }
            results = engine.discover(rounds=2)
            assert len(results) > 0

def test_create_flanker(mock_session, mock_dependencies):
    # Setup DB
    ing = Ingredient(name="Rose", category="Heart", price=100.0, olfactive_family="Floral", traditional_use="Love")
    psycho = Psychophysics(odor_potency="high", russell_valence=0.8, russell_arousal=0.5)
    ing.psychophysics = psycho

    mol = Molecule(smiles="C=CC", molecular_weight=120.0, log_p=1.5)
    comp = Composition(ingredient=ing, molecule=mol)
    mock_session.add_all([ing, mol, comp])

    ing2 = Ingredient(name="Oud", category="Base", price=500.0, olfactive_family="Woody", traditional_use="Incense")
    psycho2 = Psychophysics(odor_potency="high", russell_valence=0.5, russell_arousal=0.5)
    ing2.psychophysics = psycho2

    mol2 = Molecule(smiles="C1=CC=CC=C1", molecular_weight=200.0, log_p=3.5)
    comp2 = Composition(ingredient=ing2, molecule=mol2)
    mock_session.add_all([ing2, mol2, comp2])

    mock_session.commit()

    model = MagicMock()
    engine = DiscoveryEngine(model, session=mock_session)

    parent = [{"name": "Rose", "weight_factor": 1.0, "category": "Heart", "olfactive_family": "Floral"}]

    with patch.object(engine, 'evaluate') as mock_evaluate:
        mock_evaluate.return_value = {
             "id": "2", "fitness": 0.8, "molecules": [], "market_tier": "Flanker Intense"
        }
        flanker = engine.create_flanker(parent, flanker_type="Intense")
        assert flanker["market_tier"].startswith("Flanker")
