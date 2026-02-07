import sys
from unittest.mock import MagicMock

# Mock dependencies BEFORE imports
# rdkit
rdkit = MagicMock()
rdkit.Chem = MagicMock()
rdkit.Chem.Descriptors = MagicMock()
# Set explicit return values for Descriptors to allow float operations
rdkit.Chem.Descriptors.MolWt.return_value = 150.0
rdkit.Chem.Descriptors.MolLogP.return_value = 2.5
rdkit.Chem.Descriptors.TPSA.return_value = 50.0
rdkit.Chem.Descriptors.MolMR.return_value = 40.0
rdkit.Chem.Descriptors.NumRotatableBonds.return_value = 5
rdkit.Chem.Descriptors.NumHDonors.return_value = 1

sys.modules["rdkit"] = rdkit
sys.modules["rdkit.Chem"] = rdkit.Chem
sys.modules["rdkit.Chem.Descriptors"] = rdkit.Chem.Descriptors

# torch
torch = MagicMock()
torch.nn = MagicMock()

class MockModule:
    def __init__(self, *args, **kwargs):
        pass
    def __call__(self, *args, **kwargs):
        return MagicMock()
    def state_dict(self):
        return {}
    def load_state_dict(self, state_dict):
        pass
    def eval(self):
        pass
    def to(self, device):
        return self

torch.nn.Module = MockModule

def MockLayerFactory(*args, **kwargs):
    return MagicMock()

torch.nn.Linear = MockLayerFactory
torch.nn.functional = MagicMock()
torch.sigmoid = MagicMock()
torch.tensor = MagicMock()
torch.Tensor = MagicMock # Scipy needs this to be a class
torch.float = "float"
torch.long = "long"
torch.optim = MagicMock()
sys.modules["torch"] = torch
sys.modules["torch.nn"] = torch.nn
sys.modules["torch.nn.functional"] = torch.nn.functional
sys.modules["torch.optim"] = torch.optim

# torch_geometric
torch_geometric = MagicMock()
torch_geometric.nn = MagicMock()
torch_geometric.nn.GCNConv = MockLayerFactory
torch_geometric.nn.global_mean_pool = MagicMock()
torch_geometric.data = MagicMock()
torch_geometric.data.Data = MagicMock
torch_geometric.loader = MagicMock()
torch_geometric.loader.DataLoader = MagicMock
sys.modules["torch_geometric"] = torch_geometric
sys.modules["torch_geometric.nn"] = torch_geometric.nn
sys.modules["torch_geometric.data"] = torch_geometric.data
sys.modules["torch_geometric.loader"] = torch_geometric.loader

# UI/Viz libs
sys.modules["altair"] = MagicMock()
sys.modules["plotly"] = MagicMock()
sys.modules["plotly.express"] = MagicMock()
sys.modules["streamlit"] = MagicMock()

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
try:
    from infra.models import Base
except ImportError:
    # If infra.models fails (e.g. dependency issues), we mock Base
    Base = MagicMock()
    Base.metadata = MagicMock()
    Base.metadata.create_all = MagicMock()

@pytest.fixture
def mock_session():
    # Use SQLite in-memory for testing DB interactions
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()

@pytest.fixture
def sample_molecule_data():
    return {
        "name": "Test Molecule",
        "molecular_weight": 200.0,
        "polarity": 2.5,
        "category": "Heart",
        "smiles": "CCO",
        "price_per_kg": 150.0,
        "ifra_limit": 1.0,
        "olfactive_family": "Floral",
        "traditional_use": "Relaxing",
        "russell_valence": 0.8,
        "russell_arousal": -0.2,
        "odor_potency": "high",
        "biodegradability": True,
        "renewable_source": True,
        "carbon_footprint": 5.0,
        "weight_factor": 1.0,
        "complexity_tier": 1
    }

@pytest.fixture
def sample_molecules(sample_molecule_data):
    return [sample_molecule_data]
