"""
Q4_2.py

For the MLP architecture, this script investigates the effect of the choice of non-linearity
while keeping all other hyperparameters at their default settings. For each activation
function, training and validation losses and accuracies are recorded over epochs.
At the end, four plots are generated with epoch on the x-axis and the legend indicating
the non-linearity used.

Usage: 
    python Q4_2.py
"""

import os
import time
import json
import warnings
import argparse

import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms
from tqdm import tqdm

from config import get_config_parser
from mlp import MLP
from utils import seed_experiment, to_device, cross_entropy_loss, compute_accuracy, generate_plots

# -------------------------
# Training and Evaluation Functions (adapted from main.py)
# -------------------------
def train(epoch, model, dataloader, optimizer, device, print_every):
    model.train()
    epoch_accuracy = 0
    epoch_loss = 0
    start_time = time.time()
    
    for idx, batch in enumerate(dataloader):
        batch = to_device(batch, device)
        optimizer.zero_grad()
        imgs, labels = batch
        logits = model(imgs)
        loss = cross_entropy_loss(logits, labels)
        acc = compute_accuracy(logits, labels)

        loss.backward()
        optimizer.step()
        epoch_accuracy += acc.item() / len(dataloader)
        epoch_loss += loss.item() / len(dataloader)
        
        if idx % print_every == 0:
            tqdm.write(f"[TRAIN] Epoch: {epoch}, Iter: {idx}, Loss: {loss.item():.5f}")
            
    tqdm.write(f"== [TRAIN] Epoch: {epoch}, Accuracy: {epoch_accuracy:.3f} ==>")
    return epoch_loss, epoch_accuracy, time.time() - start_time

def evaluate(epoch, model, dataloader, device, print_every, mode="val"):
    model.eval()
    epoch_accuracy = 0
    epoch_loss = 0
    start_time = time.time()
    
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            batch = to_device(batch, device)
            imgs, labels = batch
            logits = model(imgs)
            loss = cross_entropy_loss(logits, labels)
            acc = compute_accuracy(logits, labels)
            epoch_accuracy += acc.item() / len(dataloader)
            epoch_loss += loss.item() / len(dataloader)
            
            if idx % print_every == 0:
                tqdm.write(f"[{mode.upper()}] Epoch: {epoch}, Iter: {idx}, Loss: {loss.item():.5f}")
                
        tqdm.write(f"=== [{mode.upper()}] Epoch: {epoch}, Accuracy: {epoch_accuracy:.3f} ===>")
    return epoch_loss, epoch_accuracy, time.time() - start_time

# -------------------------
# Main Experiment Routine
# -------------------------
def main():
    # Parse command-line arguments (using defaults if not provided)
    parser = get_config_parser()
    args = parser.parse_args([])

    # Check for the device
    if (args.device == "cuda") and not torch.cuda.is_available():
        warnings.warn(
            "CUDA is not available, make sure your environment is running on GPU "
            "(e.g. in the Notebook Settings in Google Colab). Forcing device='cpu'."
        )
        args.device = "cpu"

    if args.device == "cpu":
        warnings.warn(
            "You are about to run on CPU, and might run out of memory shortly. "
            "You can try setting batch_size=1 to reduce memory usage."
        )
    
    # Force the use of the MLP architecture for this experiment.
    args.model = "mlp"

    # Load the base model configuration from mlp.json
    base_config_path = '/home/ethan/IFT6135/IFT6135-2025/HW1_2025/assignment1_release/model_configs/mlp.json'
    with open(base_config_path, 'r') as f:
        base_model_config = json.load(f)
    
    # Define transforms and load CIFAR10 dataset (similar to main.py)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.49139968, 0.48215841, 0.44653091],
                             [0.24703223, 0.24348513, 0.26158784])
    ])
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop((32, 32), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize([0.49139968, 0.48215841, 0.44653091],
                             [0.24703223, 0.24348513, 0.26158784])
    ])
    
    train_dataset = CIFAR10(root='/home/ethan/IFT6135/IFT6135-2025/HW1_2025/assignment1_release/data', train=True, transform=train_transform, download=True)
    val_dataset = CIFAR10(root='/home/ethan/IFT6135/IFT6135-2025/HW1_2025/assignment1_release/data', train=True, transform=test_transform, download=True)
    train_set, _ = torch.utils.data.random_split(train_dataset, [45000, 5000])
    _, val_set = torch.utils.data.random_split(val_dataset, [45000, 5000])
    test_set = CIFAR10(root='/home/ethan/IFT6135/IFT6135-2025/HW1_2025/assignment1_release/data', train=False, transform=test_transform, download=True)
    
    train_dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True, pin_memory=True, num_workers=4)
    valid_dataloader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=4)
    # (Test loader is not used in this experiment)
    
    # Seed for reproducibility
    seed_experiment(args.seed)
    
    # List of non-linearities to test
    nonlinearities = ['sigmoid','tanh','relu']
    log_dirs = []
    legend_names = []
    
    # Loop over each activation function experiment
    for nl in nonlinearities:
        tqdm.write(f"Running experiment with activation: {nl}")
        log_dir = f'/home/ethan/IFT6135/IFT6135-2025/HW1_2025/assignment1_release/logs/mlp_{nl}'
        os.makedirs(log_dir, exist_ok=True)
        log_dirs.append(log_dir)
        legend_names.append(nl)
        
        # Update the model configuration with the current activation function
        model_config = base_model_config.copy()
        #model_config['activation'] = nl
        
        # Build the MLP model (the MLP constructor takes input_size, hidden_sizes, and num_classes)
        model = MLP(model_config['input_size'], model_config['hidden_sizes'], model_config['num_classes'],activation=str(nl))
        model.to(args.device)
        
        # Set up the optimizer (using defaults from args)
        if args.optimizer == "adamw":
            optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer == "adam":
            optimizer = optim.Adam(model.parameters(), lr=args.lr)
        elif args.optimizer == "sgd":
            optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer == "momentum":
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        else:
            raise ValueError("Unsupported optimizer")
        
        # Record training statistics for this experiment
        train_losses, valid_losses = [], []
        train_accs, valid_accs = [], []
        
        for epoch in range(args.epochs):
            tqdm.write(f"====== Activation: {nl} | Epoch {epoch} ======>")
            loss, acc, _ = train(epoch, model, train_dataloader, optimizer, args.device, args.print_every)
            train_losses.append(loss)
            train_accs.append(acc)
            loss, acc, _ = evaluate(epoch, model, valid_dataloader, args.device, args.print_every, mode="val")
            valid_losses.append(loss)
            valid_accs.append(acc)
        
        # Save the results in a JSON file in the log directory
        results = {
            "train_losses": train_losses,
            "valid_losses": valid_losses,
            "train_accs": train_accs,
            "valid_accs": valid_accs
        }
        with open(os.path.join(log_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=4)
    
    # After all experiments, generate plots for the four metrics.
    plots_dir = '/home/ethan/IFT6135/IFT6135-2025/HW1_2025/assignment1_release/plots/Q4_2'
    os.makedirs(plots_dir, exist_ok=True)
    generate_plots(log_dirs, legend_names, plots_dir)
    
    print("Experiments completed and plots generated in the 'plots' directory.")

if __name__ == "__main__":
    main()
