import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class MoleculeGNN(nn.Module):
    def __init__(self, num_node_features=5):
        super(MoleculeGNN, self).__init__()
        self.conv1 = GCNConv(num_node_features, 64)
        self.conv2 = GCNConv(64, 128)
        self.conv3 = GCNConv(128, 64)
        self.fc = nn.Linear(64, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))

        x = global_mean_pool(x, batch) 
        return torch.sigmoid(self.fc(x))

    def predict(self, batch_data):
        self.eval()
        with torch.no_grad():
            return self.forward(batch_data).cpu().numpy()

    def save(self, path="results/perfume_gnn.pth"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)

    def load(self, path="results/perfume_gnn.pth"):
        if os.path.exists(path):
            self.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
            self.eval()
            return True
        return False