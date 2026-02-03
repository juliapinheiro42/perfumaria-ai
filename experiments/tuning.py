import os
import json
import optuna
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from core.discovery import DiscoveryEngine
from core.model import PerfumeTechModel
from core.encoder import FeatureEncoder

def load_data(n_samples=500):
    """
    Gera um dataset usando o DiscoveryEngine e o CSV.
    """
    dummy_model = PerfumeTechModel(input_size=FeatureEncoder.INPUT_SIZE)
    engine = DiscoveryEngine(model=dummy_model, csv_path="insumos.csv")
    
    print(f"[TUNING] Gerando {n_samples} amostras para otimização...")
    
    X_list = []
    y_list = []
    
    count = 0
    while count < n_samples:
        molecules = engine._generate_molecules(None)
        
        molecules = [engine._enrich_with_rdkit(m) for m in molecules]
        
        if not engine._validate_chemical_synergy(molecules):
            continue
            
        result = engine.evaluate(molecules)
        if result["fitness"] <= 0.01:
            continue
            
        features = FeatureEncoder.encode_blend(molecules)
        
        X_list.append(features)
        y_list.append(result["fitness"])
        count += 1
        
    return np.array(X_list), np.array(y_list)

CACHED_X, CACHED_Y = None, None

def objective(trial):
    global CACHED_X, CACHED_Y
    
    if CACHED_X is None:
        CACHED_X, CACHED_Y = load_data(n_samples=500)
        
    X, y = CACHED_X, CACHED_Y
    
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    hidden_size = trial.suggest_int("hidden_size", 32, 256)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    epochs = trial.suggest_int("epochs", 20, 100)
    
    model = PerfumeTechModel(input_size=X.shape[1])
    
    model.network = nn.Sequential(
        nn.Linear(X.shape[1], hidden_size),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_size, hidden_size // 2),
        nn.ReLU(),
        nn.Dropout(dropout / 2),
        nn.Linear(hidden_size // 2, 1),
        nn.Sigmoid()
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    X_tensor = torch.from_numpy(X).float()
    y_tensor = torch.from_numpy(y).float().view(-1, 1)
    
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        preds = model(X_tensor)
        loss = criterion(preds, y_tensor)
        loss.backward()
        optimizer.step()
        
    model.eval()
    with torch.no_grad():
        preds = model(X_tensor)
        final_loss = criterion(preds, y_tensor)
        
    return final_loss.item()

if __name__ == "__main__":
    print(" Iniciando Tuning de Hiperparâmetros...")
    
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)

    print("\n Melhores Hiperparâmetros:")
    print(study.best_params)
    
    os.makedirs("results", exist_ok=True)
    with open("results/best_params.json", "w") as f:
        json.dump(study.best_params, f)
    print(" Parâmetros salvos em results/best_params.json")