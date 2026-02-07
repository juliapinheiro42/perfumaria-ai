import pytest
from unittest.mock import MagicMock, patch
import sys


from core.model import MoleculeGNN


def test_gnn_initialization():
    model = MoleculeGNN(num_node_features=5)
    assert model.conv1 is not None
    assert model.fc is not None


def test_gnn_forward():
    model = MoleculeGNN()

    data = MagicMock()
    data.x = MagicMock()
    data.edge_index = MagicMock()
    data.batch = MagicMock()

    output = model.forward(data)
    assert output is not None


def test_gnn_predict():
    model = MoleculeGNN()
    batch_data = MagicMock()

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

    mock_exists.return_value = True
    result = model.load("test_path.pth")

    mock_load.assert_called()
    assert result is True

    mock_exists.return_value = False
    result = model.load("test_path.pth")
    assert result is False
