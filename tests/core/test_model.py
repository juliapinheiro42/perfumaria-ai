import pytest
from unittest.mock import MagicMock, patch
import sys

# Ensure mocks are in place via conftest (which runs before this)
# But we can also access them via sys.modules["torch"]

from core.model import MoleculeGNN

def test_gnn_initialization():
    model = MoleculeGNN(num_node_features=5)
    assert model.conv1 is not None
    assert model.fc is not None

def test_gnn_forward():
    model = MoleculeGNN()

    # Create mock data
    data = MagicMock()
    data.x = MagicMock()
    data.edge_index = MagicMock()
    data.batch = MagicMock()

    # Setup return values for layers so we don't crash
    # The layers (conv1, etc) are already mocks because GCNConv is a mock class

    # We need to ensure F.relu returns something (it is a mock)
    # torch.sigmoid returns something

    output = model.forward(data)
    assert output is not None

def test_gnn_predict():
    model = MoleculeGNN()
    batch_data = MagicMock()

    # forward returns a mock
    # we need that mock to have .cpu().numpy()

    # By default MagicMock returns a MagicMock on call, so .cpu() returns a mock, .numpy() returns a mock.

    prediction = model.predict(batch_data)
    assert prediction is not None

@patch("torch.save")
@patch("os.makedirs")
def test_gnn_save(mock_makedirs, mock_save):
    model = MoleculeGNN()
    model.save("test_path.pth")

    mock_makedirs.assert_called()
    mock_save.assert_called()

@patch("torch.load")
@patch("os.path.exists")
def test_gnn_load(mock_exists, mock_load):
    model = MoleculeGNN()

    # Case: File exists
    mock_exists.return_value = True
    result = model.load("test_path.pth")

    mock_load.assert_called()
    assert result is True

    # Case: File does not exist
    mock_exists.return_value = False
    result = model.load("test_path.pth")
    assert result is False
