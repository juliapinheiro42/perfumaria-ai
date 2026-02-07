import pytest
from unittest.mock import patch, MagicMock
import core.market
from core.market import PerfumeBusinessEngine, business_evaluation

@pytest.fixture
def mock_market_data():
    data = {
        "Rose": {"price_per_kg": 150.0, "ifra_limit": 1.0, "traditional_use": "Relaxing", "russell_valence": 0.8, "russell_arousal": -0.2},
        "Jasmine": {"price_per_kg": 200.0, "ifra_limit": 0.5, "traditional_use": "Sensual", "russell_valence": 0.7, "russell_arousal": 0.4}
    }
    # Patch the global variable DADOS_INSUMOS in core.market
    with patch.dict(core.market.DADOS_INSUMOS, data, clear=True):
        yield

def test_business_evaluation(mock_market_data):
    molecules = [{"name": "Rose", "weight_factor": 1.0}]
    result = business_evaluation(molecules, tech_score=5.0, neuro_score=0.5)

    assert "financials" in result
    assert "market_strategy" in result
    assert "compliance" in result
    assert result["market_tier"] == "Mass Market"

def test_calculate_global_fit(mock_market_data):
    engine = PerfumeBusinessEngine()
    molecules = [{"name": "Rose", "weight_factor": 1.0}]
    result = engine.calculate_global_fit(molecules)

    assert "rankings" in result
    assert "best" in result
    # Rose valence is 0.8
    assert result["coords"]["valence"] == 0.8

def test_estimate_financials(mock_market_data):
    engine = PerfumeBusinessEngine()
    molecules = [{"name": "Rose", "weight_factor": 1.0}]
    result = engine.estimate_financials(molecules, tech_score=8.0, neuro_score=0.8)

    assert result["cost"] > 0
    assert result["price"] > result["cost"]
    assert result["market_tier"] == "Luxury" # Because tech_score > 7.0

def test_validate_ifra_ok(mock_market_data):
    engine = PerfumeBusinessEngine(dilution=0.1)
    molecules = [{"name": "Rose", "weight_factor": 1.0}]
    # Rose limit is 1.0 (100%). Exposure is 0.1 (10%). Should be OK.
    result = engine.validate_ifra(molecules)
    assert result["is_legal"] is True
    assert len(result["logs"]) == 0

def test_validate_ifra_fail(mock_market_data):
    engine = PerfumeBusinessEngine(dilution=0.6) # High dilution
    molecules = [{"name": "Jasmine", "weight_factor": 1.0}]
    # Jasmine limit is 0.5 (50%). Exposure is 0.6 (60%). Should fail.
    result = engine.validate_ifra(molecules)
    assert result["is_legal"] is False
    assert len(result["logs"]) > 0

def test_unknown_ingredient(mock_market_data):
    engine = PerfumeBusinessEngine()
    molecules = [{"name": "Unknown", "weight_factor": 1.0}]
    result = engine.calculate_global_fit(molecules)
    # Should handle missing data gracefully (defaults to 0.0)
    assert result["coords"]["valence"] == 0.0
