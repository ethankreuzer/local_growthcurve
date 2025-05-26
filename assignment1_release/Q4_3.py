"""
Q4_3.py

For the ResNet18 architecture, this script investigates the effect of learning rate on training
and validation performance when using the Adam optimizer. We perform experiments with learning 
rates of 0.1, 0.01, 0.001, 0.0001, and 0.00001, and record training loss, validation loss, 
training accuracy, and validation accuracy over epochs. At the end, four plots are generated with 
epochs on the x-axis and a legend indicating the learning rate used.

Usage:
    python Q4_3.py
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
from resnet18 import ResNet18
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
    
    # Force the use of the ResNet18 architecture and Adam optimizer for this experiment.
    args.model = "resnet18"
    args.optimizer = "adam"
    
    # For ResNet18, we assume that the model can be constructed without a JSON configuration.
    # If your ResNet18 implementation requires parameters from a config file, you can load them here.
    # In this example, we'll use an empty dictionary.
    model_config = {}  # Adjust if needed

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
    
    data_root = '/home/ethan/IFT6135/IFT6135-2025/HW1_2025/assignment1_release/data'
    train_dataset = CIFAR10(root=data_root, train=True, transform=train_transform, download=True)
    val_dataset = CIFAR10(root=data_root, train=True, transform=test_transform, download=True)
    train_set, _ = torch.utils.data.random_split(train_dataset, [45000, 5000])
    _, val_set = torch.utils.data.random_split(val_dataset, [45000, 5000])
    test_set = CIFAR10(root=data_root, train=False, transform=test_transform, download=True)
    
    train_dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True, pin_memory=True, num_workers=4)
    valid_dataloader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=4)
    
    # Seed for reproducibility
    seed_experiment(args.seed)
    
    # List of learning rates to test
    learning_rates = [0.1, 0.01, 0.001, 0.0001, 0.00001]
    log_dirs = []
    legend_names = []
    
    # Loop over each learning rate experiment
    for lr in learning_rates:
        tqdm.write(f"Running experiment with learning rate: {lr}")
        log_dir = f'/home/ethan/IFT6135/IFT6135-2025/HW1_2025/assignment1_release/logs/resnet18_lr_{lr}'
        os.makedirs(log_dir, exist_ok=True)
        log_dirs.append(log_dir)
        legend_names.append(f"lr={lr}")
        
        # Build the ResNet18 model
        model = ResNet18(**model_config)  # Adjust if ResNet18 requires specific parameters
        model.to(args.device)
        
        # Set up the optimizer with the current learning rate (using Adam)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        # Record training statistics for this experiment
        train_losses, valid_losses = [], []
        train_accs, valid_accs = [], []
        
        for epoch in range(args.epochs):
            tqdm.write(f"====== LR: {lr} | Epoch {epoch} ======>")
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
    plots_dir = '/home/ethan/IFT6135/IFT6135-2025/HW1_2025/assignment1_release/plots/Q4_3'
    os.makedirs(plots_dir, exist_ok=True)
    generate_plots(log_dirs, legend_names, plots_dir)
    
    print("Experiments completed and plots generated in the 'plots' directory.")

if __name__ == "__main__":
    main()
