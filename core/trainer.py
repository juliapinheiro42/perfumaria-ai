import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader

class ModelTrainer:
    def __init__(self, model, lr=0.005, weight_decay=5e-4):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.criterion = nn.MSELoss() # Mean Squared Error para prever o Fitness
        self.batch_size = 32
        self.min_buffer_size = 10

    def train_step(self, buffer):
        """
        Executa UM passo de otimização (Gradient Descent).
        Usado para aprendizado imediato logo após o feedback humano.
        """
        if buffer.size() < 1: 
            return 0.0

        self.model.train()
        
        # Pega um batch do buffer (a função sample já deve lidar com pesos se existirem)
        data_list = buffer.sample(self.batch_size)
        
        if not data_list:
            return 0.0

        # Cria o loader do PyTorch Geometric para empilhar os grafos
        loader = DataLoader(data_list, batch_size=len(data_list), shuffle=True)
        
        try:
            batch = next(iter(loader))
        except StopIteration:
            return 0.0

        # 1. Zerar gradientes
        self.optimizer.zero_grad()

        # 2. Forward Pass (Previsão)
        out = self.model(batch)  # Shape esperado: [Batch_Size, 1]
        
        # Garante que o target tenha o mesmo shape
        target = batch.y.view(-1, 1).float()

        # 3. Cálculo da Perda
        loss = self.criterion(out, target)

        # 4. Backward Pass (Aprendizado)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def maybe_retrain(self, buffer, threshold=50):
        """
        Retreina apenas se o buffer tiver dados novos suficientes.
        Retorna True se treinou.
        """
        # Se o buffer cresceu muito desde o último treino completo, treina de novo
        if buffer.size() >= self.min_buffer_size and buffer.size() % threshold == 0:
            avg_loss = self.retrain(buffer, epochs=3)
            return True
        return False

    def retrain(self, buffer, epochs=10):
        """
        Treinamento completo em todo o histórico (Replay Buffer).
        Usado periodicamente ou no Warmup.
        """
        if buffer.size() < self.min_buffer_size:
            return 0.0

        self.model.train()
        total_loss = 0
        
        # Usa todo o dataset disponível
        data_list = buffer.get_all()
        loader = DataLoader(data_list, batch_size=self.batch_size, shuffle=True)

        for epoch in range(epochs):
            epoch_loss = 0
            steps = 0
            
            for batch in loader:
                self.optimizer.zero_grad()
                
                out = self.model(batch)
                target = batch.y.view(-1, 1).float()
                
                loss = self.criterion(out, target)
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                steps += 1
            
            if steps > 0:
                total_loss += (epoch_loss / steps)

        return total_loss / max(epochs, 1)

    def save(self, path="model_checkpoint.pth"):
        torch.save(self.model.state_dict(), path)

    def load(self, path="model_checkpoint.pth"):
        try:
            self.model.load_state_dict(torch.load(path))
            self.model.eval()
            return True
        except FileNotFoundError:
            return False