import numpy as np
import torch
import random

class ReplayBuffer:
    def __init__(self, max_size=2000):
        self.max_size = max_size
        self.storage = []
        self.targets = []
        self.weights = []

    def add(self, graphs, fitness, weight=1.0):
        """
        Adiciona uma nova experiência (fórmula) ao buffer.
        graphs: Lista de objetos Data (PyTorch Geometric)
        fitness: Nota float (0.0 a 1.0 ou mais)
        weight: Peso do treino (ex: 5.0 para feedback humano, 1.0 para IA)
        """
        if len(self.storage) >= self.max_size:
            self.storage.pop(0)
            self.targets.pop(0)
            self.weights.pop(0)
        
        self.storage.append(graphs)
        self.targets.append(float(fitness))
        self.weights.append(float(weight))

    def size(self):
        return len(self.storage)

    def sample(self, batch_size):
        """
        Retorna uma lista plana de grafos prontos para o DataLoader.
        Seleciona aleatoriamente 'batch_size' fórmulas do histórico.
        """
        current_size = len(self.storage)
        if current_size == 0:
            return []

        indices = np.random.choice(current_size, min(current_size, batch_size), replace=False)
        
        flat_data_list = []
        
        for i in indices:
            graphs_list = self.storage[i]
            target_val = self.targets[i]
            weight_val = self.weights[i]
            
            for graph in graphs_list:
                g = graph.clone()
                
                g.y = torch.tensor([target_val], dtype=torch.float)
                g.weight = torch.tensor([weight_val], dtype=torch.float)
                
                flat_data_list.append(g)
                
        return flat_data_list

    def get_all(self):
        """
        Retorna TODO o histórico formatado para o DataLoader.
        Usado para retreino completo (Warmup ou Checkpoints).
        """
        flat_data_list = []
        
        for i, graphs_list in enumerate(self.storage):
            target_val = self.targets[i]
            weight_val = self.weights[i]
            
            for graph in graphs_list:
                g = graph.clone()
                g.y = torch.tensor([target_val], dtype=torch.float)
                g.weight = torch.tensor([weight_val], dtype=torch.float)
                flat_data_list.append(g)
                
        return flat_data_list
    
    def generate_negative_examples(formula, compliance_engine, num_variants=3):

        negatives = []
    
        restricted_chems = list(compliance_engine.IFRA_LIMITS.keys())

        for _ in range(num_variants):
            corrupted_formula = [m.copy() for m in formula]
        
            target_chem = random.choice(restricted_chems)
        
            for ingredient in corrupted_formula:
                if target_chem in ingredient['name']:
                    ingredient['weight_factor'] *= random.uniform(10.0, 50.0)
        
            is_safe, _, _ = compliance_engine.check_safety(corrupted_formula)
        
            if not is_safe:
                negatives.append((corrupted_formula, 0.05)) 
            
        return negatives