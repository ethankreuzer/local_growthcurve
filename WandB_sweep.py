#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import wandb
import copy
import os

import sys
sys.path.append('/home/ethan/GrowthCurve/scripts')  # Add the scripts directory to sys.path
from NeuralNet import NN_sum  

df_train = pd.read_pickle('/home/ethan/GrowthCurve/data/df_train.pkl')
df_test = pd.read_pickle('/home/ethan/GrowthCurve/data/df_test.pkl')
df_train = df_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

columns_to_remove = ['Compound', 'Activity', 'Smiles', 'indx_conc', 'indx_time', 
                     'scaffold', 'maccs_fp', 'ecfp_fp', 'rdkit_fp', 'OD']
X_train = df_train.drop(columns=columns_to_remove)
X_test = df_test.drop(columns=columns_to_remove)

y_train = df_train['OD']
y_test = df_test['OD']

time_cols = [col for col in X_train.columns if col.startswith('time')]
conc_cols = [col for col in X_train.columns if col.startswith('conc')]
other_cols = [col for col in X_train.columns if col not in time_cols + conc_cols]

t_in = len(time_cols)
c_in = len(conc_cols)
f_in = len(other_cols)


X_train_ordered = X_train[time_cols + conc_cols + other_cols]
X_test_ordered = X_test[time_cols + conc_cols + other_cols]

X_train_tensor = torch.tensor(X_train_ordered.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).squeeze(dim=-1)

X_test_tensor = torch.tensor(X_test_ordered.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).squeeze(dim=-1)

# --------------------------
# Define W&B Sweep Config
# --------------------------

sweep_config = {
    "method": "bayes",
    "metric": {"name": "test_loss", "goal": "minimize"},
    "parameters": {
        "patience": {"values": [3, 5, 10, 15]},
        "weight_decay": {"distribution": "log_uniform_values", "min": 1e-6, "max": 1e-2}, 
        "dropout": {"min": 0.0, "max": 0.5},
        "learning_rate": {"distribution": "log_uniform_values", "min": 1e-6, "max": 1e-2}, 
        "batch_size": {"values": [64, 128, 256, 512]},
        "n": {"values": [32, 64, 128, 256, 512]},
        "t_layers": {"values": [1, 2, 3, 4, 5]},
        "t_dim": {"values": [8, 16, 32, 64, 128, 256, 512]},  
        "c_layers": {"values": [1, 2, 3, 4, 5]},
        "c_dim": {"values": [8, 16, 32, 64, 128, 256, 512]}, 
        "f_layers": {"values": [2, 3, 4, 5, 6, 7, 8, 9, 10]},
        "f_dim": {"values": [8, 16, 32, 64, 128, 256, 512]}, 
        "last_mlp_layers": {"values": [2, 3, 4, 5, 6, 7, 8, 9, 10]},
        "last_mlp_dim": {"values": [16, 32, 64, 128, 256, 512]} 
    }
}

# --------------------------
# Training Function
# --------------------------

def train():
    """Train function for W&B sweeps."""
    wandb.init()
    config = wandb.config

    # Load data inside function to avoid UnboundLocalError
    df_train = pd.read_pickle('/home/ethan/GrowthCurve/data/df_train.pkl')
    df_test = pd.read_pickle('/home/ethan/GrowthCurve/data/df_test.pkl')
    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    columns_to_remove = ['Compound', 'Activity', 'Smiles', 'indx_conc', 'indx_time', 
                         'scaffold', 'maccs_fp', 'ecfp_fp', 'rdkit_fp', 'OD']
    X_train = df_train.drop(columns=columns_to_remove)
    X_test = df_test.drop(columns=columns_to_remove)

    y_train = df_train['OD']
    y_test = df_test['OD']

    time_cols = [col for col in X_train.columns if col.startswith('time')]
    conc_cols = [col for col in X_train.columns if col.startswith('conc')]
    other_cols = [col for col in X_train.columns if col not in time_cols + conc_cols]

    t_in = len(time_cols)
    c_in = len(conc_cols)
    f_in = len(other_cols)

    X_train_ordered = X_train[time_cols + conc_cols + other_cols]
    X_test_ordered = X_test[time_cols + conc_cols + other_cols]

    # Convert to torch tensors
    X_train_tensor = torch.tensor(X_train_ordered.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)

    X_test_tensor = torch.tensor(X_test_ordered.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

    # Initialize Model with Configurable Hyperparameters
    model = NN_sum(
        t_in, c_in, f_in, 
        config.t_layers, config.t_dim, 
        config.c_layers, config.c_dim, 
        config.f_layers, config.f_dim, 
        config.n,
        config.last_mlp_layers, config.last_mlp_dim,
        config.dropout  
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    X_train_tensor, y_train_tensor = X_train_tensor.to(device), y_train_tensor.to(device)
    X_test_tensor, y_test_tensor = X_test_tensor.to(device), y_test_tensor.to(device)

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor),
        batch_size=config.batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor),
        batch_size=config.batch_size, shuffle=False
    )

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    

    best_loss = float("inf")
    patience_counter = 0
    best_model_state = copy.deepcopy(model.state_dict())

    max_epochs = 200  
    patience = config.patience 

    for epoch in range(max_epochs):
        model.train()
        running_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            
            loss = criterion(outputs, batch_y)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * batch_X.size(0)
        

        epoch_train_loss = running_loss / len(train_loader.dataset)
        
        model.eval()
        test_running_loss = 0.0
        with torch.no_grad():
            for test_X, test_y in test_loader:
                test_X, test_y = test_X.to(device), test_y.to(device)
                test_outputs = model(test_X)

                test_loss = criterion(test_outputs, test_y)
                test_running_loss += test_loss.item() * test_X.size(0)

        epoch_test_loss = test_running_loss / len(test_loader.dataset)
        
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": epoch_train_loss,
            "test_loss": epoch_test_loss,
            "patience": patience,
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
            "weight_decay": config.weight_decay,
            "dropout": config.dropout,
            "t_layers": config.t_layers,
            "t_dim": config.t_dim,
            "c_layers": config.c_layers,
            "c_dim": config.c_dim,
            "f_layers": config.f_layers,
            "f_dim": config.f_dim,
            "last_mlp_layers": config.last_mlp_layers,
            "last_mlp_dim": config.last_mlp_dim
            })

        if epoch_test_loss < best_loss:
            best_loss = epoch_test_loss
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                print(f"Early stopping at epoch {epoch + 1} (patience {patience})")
                break

    
    model.load_state_dict(best_model_state)
    torch.save(model.state_dict(), f"/home/ethan/GrowthCurve/models/best_model_{wandb.run.id}.pt")

    artifact = wandb.Artifact("best_model", type="model", description="Best model from this run")
    artifact.add_file(f"/home/ethan/GrowthCurve/models/best_model_{wandb.run.id}.pt")
    
    artifact.metadata = dict(wandb.config)
    wandb.log_artifact(artifact)
    wandb.finish()


# --------------------------
# Run Sweep
# --------------------------

wandb.login(key="de72b97eb2e03a1787b54e0a865d70bd01be94bb")

sweep_id = wandb.sweep(sweep_config, project="GrowthCurve_tuning_sum_embeddings_model")

wandb.agent(sweep_id, function=train, count=100)
