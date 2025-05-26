"""
Q4_5_tuning.py

Perform hyperparameter tuning for the ResNet18 model using CIFAR10 by tuning the weight decay.
The learning rate is fixed at 1e-6, and early stopping is implemented (with a patience of 5 epochs)
for each weight decay candidate. For each configuration, the best epoch (the stopping epoch)
is recorded along with the best validation accuracy. Additionally, a plot is saved for each
weight decay candidate showing training loss, validation loss, training accuracy, and validation accuracy
vs. epoch.

Usage:
    python Q4_5_tuning.py
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
import matplotlib.pyplot as plt

from config import get_config_parser
from resnet18 import ResNet18
from utils import seed_experiment, to_device, cross_entropy_loss, compute_accuracy

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

def plot_training_curves(epochs, train_losses, valid_losses, train_accs, valid_accs, save_path, config_str):
    """
    Create and save a plot with four subplots:
      - Training Loss vs. Epoch
      - Validation Loss vs. Epoch
      - Training Accuracy vs. Epoch
      - Validation Accuracy vs. Epoch
    """
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs[0, 0].plot(epochs, train_losses, marker='o')
    axs[0, 0].set_title(f"Training Loss\n{config_str}")
    axs[0, 0].set_xlabel("Epoch")
    axs[0, 0].set_ylabel("Loss")

    axs[0, 1].plot(epochs, valid_losses, marker='o', color='orange')
    axs[0, 1].set_title("Validation Loss")
    axs[0, 1].set_xlabel("Epoch")
    axs[0, 1].set_ylabel("Loss")

    axs[1, 0].plot(epochs, train_accs, marker='o', color='green')
    axs[1, 0].set_title("Training Accuracy")
    axs[1, 0].set_xlabel("Epoch")
    axs[1, 0].set_ylabel("Accuracy")

    axs[1, 1].plot(epochs, valid_accs, marker='o', color='red')
    axs[1, 1].set_title("Validation Accuracy")
    axs[1, 1].set_xlabel("Epoch")
    axs[1, 1].set_ylabel("Accuracy")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved training curves to {save_path}")

def main():
    # Parse arguments (using defaults)
    parser = get_config_parser()
    args = parser.parse_args([])

    # Check device
    if (args.device == "cuda") and not torch.cuda.is_available():
        warnings.warn("CUDA is not available, forcing device='cpu'.")
        args.device = "cpu"
    if args.device == "cpu":
        warnings.warn("Running on CPU. Consider reducing batch_size if memory is an issue.")

    # Use ResNet18 model for tuning.
    args.model = "resnet18"

    # Set fixed learning rate and define candidate weight decays.
    fixed_lr = 1e-4
    candidate_weight_decays = [0, 1e-3,5e-4, 1e-4, 1e-5]
    
    # Early stopping parameterss
    patience = 8
    max_epochs = 200  # maximum epochs if early stopping doesn't trigger

    # Define transforms and load CIFAR10 dataset.
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
    
    data_root = './data'
    train_dataset = CIFAR10(root=data_root, train=True, transform=train_transform, download=True)
    val_dataset = CIFAR10(root=data_root, train=True, transform=test_transform, download=True)
    train_set, _ = torch.utils.data.random_split(train_dataset, [45000, 5000])
    _, val_set = torch.utils.data.random_split(val_dataset, [45000, 5000])
    
    train_dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                  drop_last=True, pin_memory=True, num_workers=4)
    valid_dataloader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                                  drop_last=False, num_workers=4)
    
    seed_experiment(args.seed)
    
    # Directory to save plots for each weight decay
    plots_dir = '/home/ethan/IFT6135/IFT6135-2025/HW1_2025/assignment1_release/plots'
    os.makedirs(plots_dir, exist_ok=True)
    
    # Dictionary to record best epochs per weight decay
    best_epochs = {}
    
    # Loop over each candidate weight decay
    for wd in candidate_weight_decays:
        config_str = f"lr={fixed_lr}, wd={wd}"
        tqdm.write(f"Running experiment with {config_str}")
        
        # Instantiate a fresh ResNet18 model.
        model = ResNet18(num_classes=10)
        model.to(args.device)
        
        # Set up the optimizer with the fixed learning rate and current weight decay.
        optimizer = optim.Adam(model.parameters(), lr=fixed_lr, weight_decay=wd)
        
        best_val_acc = 0.0
        best_epoch = 0
        no_improve = 0
        
        # Lists to record per-epoch metrics for plotting.
        train_losses, valid_losses = [], []
        train_accs, valid_accs = [], []
        epoch_numbers = []
        
        for epoch in range(max_epochs):
            tqdm.write(f"------ {config_str} | Epoch {epoch} ------")
            train_loss, train_acc, _ = train(epoch, model, train_dataloader, optimizer, args.device, args.print_every)
            _, val_acc, _ = evaluate(epoch, model, valid_dataloader, args.device, args.print_every, mode="val")
            
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            valid_losses.append(0)  # Optional: you could record validation loss if desired.
            valid_accs.append(val_acc)
            epoch_numbers.append(epoch)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                no_improve = 0
            else:
                no_improve += 1
                
            # Early stopping if no improvement for 'patience' epochs.
            if no_improve >= patience:
                tqdm.write(f"Early stopping triggered at epoch {epoch} for {config_str}.")
                break
        
        best_epochs[wd] = best_epoch
        tqdm.write(f"Best validation accuracy for {config_str} at epoch {best_epoch}: {best_val_acc:.3f}")
        
        # Save a plot of training curves for this weight decay.
        plot_save_path = os.path.join(plots_dir, f"resnet18_wd_{wd}.png")
        # Here we plot training accuracy and validation accuracy vs epoch.
        # You can modify to include training loss if you record it.
        plt.figure(figsize=(10, 6))
        plt.plot(epoch_numbers, train_accs, marker='o', label='Train Accuracy')
        plt.plot(epoch_numbers, valid_accs, marker='o', label='Validation Accuracy')
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(f"Training curves for {config_str}\nBest epoch: {best_epoch}")
        plt.legend()
        plt.savefig(plot_save_path)
        plt.close()
        tqdm.write(f"Saved training curves plot to {plot_save_path}")
    
    print("Hyperparameter tuning complete with early stopping.")
    print("Best epochs for each weight decay:")
    for wd, epoch in best_epochs.items():
        print(f"Weight Decay: {wd} -> Best Epoch: {epoch}")

if __name__ == "__main__":
    main()
