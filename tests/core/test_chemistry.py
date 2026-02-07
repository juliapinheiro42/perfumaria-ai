import pytest
import sys
from unittest.mock import MagicMock
from core.chemistry import ChemistryEngine

rdkit = sys.modules["rdkit"]
Chem = rdkit.Chem


@pytest.fixture
def chemistry_engine():
    return ChemistryEngine()


def test_evaluate_blend_empty(chemistry_engine):
    result = chemistry_engine.evaluate_blend([])
    assert result["longevity"] == 0.0
    assert result["projection"] == 0.0
    assert result["stability"] == 0.0


def test_evaluate_blend_basic(chemistry_engine, sample_molecules):
    mols = sample_molecules

    result = chemistry_engine.evaluate_blend(mols)
    assert result["longevity"] >= 1.0
    assert result["projection"] >= 1.0
    assert result["stability"] <= 1.0
    assert result["neuro_score"] > 0
    assert "temporal_curve" in result


def test_calculate_projection(chemistry_engine, sample_molecules):
    vapor_pressures = [10.0]
    logps = [2.5]

    proj = chemistry_engine._calculate_projection(
        sample_molecules, vapor_pressures, logps)
    assert isinstance(proj, float)
    assert 1.0 <= proj <= 10.0


def test_calculate_longevity(chemistry_engine, sample_molecules):
    vapor_pressures = [10.0]
    logps = [2.5]

    long_score = chemistry_engine._calculate_longevity(
        sample_molecules, vapor_pressures, logps)
    assert isinstance(long_score, float)
    assert 1.0 <= long_score <= 10.0


def test_calculate_neuro_impact(chemistry_engine, sample_molecules):
    score, vectors = chemistry_engine._calculate_neuro_impact(sample_molecules)
    assert isinstance(score, float)
    assert "valence" in vectors
    assert "arousal" in vectors


def test_estimate_vapor_pressure(chemistry_engine):
    vp = chemistry_engine._estimate_vapor_pressure(100.0)
    assert vp > 0


def test_estimate_bp(chemistry_engine):
    mol = {"molecular_weight": 200, "polarity": 2.5}
    bp = chemistry_engine._estimate_bp(mol)
    assert 80.0 <= bp <= 450.0


def test_detect_chemical_risks(chemistry_engine):

    m1 = {"name": "Test1", "smiles": "C=O", "olfactive_family": "Floral"}
    m2 = {"name": "Test2", "smiles": "CN", "olfactive_family": "Floral"}

    mock_mol = MagicMock()
    mock_mol.HasSubstructMatch.return_value = True

    Chem.MolFromSmiles.return_value = mock_mol

    risks, penalty = chemistry_engine._detect_chemical_risks([m1, m2])

    assert len(risks) > 0
    assert penalty > 0
