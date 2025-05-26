"""
Q4_6_tuning.py

Perform hyperparameter tuning for an MLPMixer model on CIFAR10 by tuning the number of Mixer blocks.
We fix the learning rate at 1e-4 and use early stopping (with a patience of 8 epochs).
The embedding dimension is set to 512 and the patch size to 4.
For each configuration, the best epoch (the stopping epoch) is recorded along with
the best validation accuracy. Additionally, a plot is saved for each candidate showing
training accuracy and validation accuracy vs. epoch.

Usage:
    python Q4_6_tuning.py
"""

import os
import time
import warnings
import argparse

import torch
from torch import optim
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

from config import get_config_parser  # if you use a common config parser
from mlpmixer import MLPMixer        # your MLPMixer model file
from utils import seed_experiment, to_device, cross_entropy_loss, compute_accuracy

def train(epoch, model, dataloader, optimizer, device, print_every):
    model.train()
    epoch_loss = 0.0
    epoch_acc = 0.0
    for idx, batch in enumerate(dataloader):
        batch = to_device(batch, device)
        imgs, labels = batch
        optimizer.zero_grad()
        logits = model(imgs)
        loss = cross_entropy_loss(logits, labels)
        acc = compute_accuracy(logits, labels)
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item() / len(dataloader)
        epoch_acc += acc.item() / len(dataloader)
        
        if idx % print_every == 0:
            tqdm.write(f"[TRAIN] Epoch: {epoch}, Iter: {idx}, Loss: {loss.item():.5f}")
    
    tqdm.write(f"== [TRAIN] Epoch: {epoch}, Accuracy: {epoch_acc:.3f} ==>")
    return epoch_loss, epoch_acc

def evaluate(epoch, model, dataloader, device, print_every, mode="val"):
    model.eval()
    epoch_loss = 0.0
    epoch_acc = 0.0
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            batch = to_device(batch, device)
            imgs, labels = batch
            logits = model(imgs)
            loss = cross_entropy_loss(logits, labels)
            acc = compute_accuracy(logits, labels)
            
            epoch_loss += loss.item() / len(dataloader)
            epoch_acc += acc.item() / len(dataloader)
            
            if idx % print_every == 0:
                tqdm.write(f"[{mode.upper()}] Epoch: {epoch}, Iter: {idx}, Loss: {loss.item():.5f}")
                
        tqdm.write(f"=== [{mode.upper()}] Epoch: {epoch}, Accuracy: {epoch_acc:.3f} ===>")
    return epoch_loss, epoch_acc

def main():
    parser = get_config_parser()
    args = parser.parse_args([])

    # Set device
    if (args.device == "cuda") and not torch.cuda.is_available():
        warnings.warn("CUDA is not available, forcing device='cpu'.")
        args.device = "cpu"
    if args.device == "cpu":
        warnings.warn("Running on CPU. Consider reducing batch_size if memory is an issue.")

    # Hyperparameters
    fixed_lr = 1e-4             # fixed learning rate
    fixed_weight_decay = 0.0      # fixed weight decay
    patch_size = 4              # fixed patch size as specified
    embed_dim = 512             # set embed_dim to 512 as required
    candidate_num_blocks = [2, 4, 6, 8]  # hyperparameter to tune
    patience = 8
    max_epochs = 120            # maximum epochs if early stopping doesn't trigger

    # Data transforms
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
    full_train_dataset = CIFAR10(root=data_root, train=True, transform=train_transform, download=True)
    full_val_dataset = CIFAR10(root=data_root, train=True, transform=test_transform, download=True)
    
    # Create training (45k) and validation (5k) splits
    train_set, _ = random_split(full_train_dataset, [45000, 5000])
    _, val_set = random_split(full_val_dataset, [45000, 5000])
    
    train_dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                  drop_last=True, pin_memory=True, num_workers=4)
    valid_dataloader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                                  drop_last=False, num_workers=4)
    
    seed_experiment(args.seed)
    
    # Directory to save plots for each candidate
    plots_dir = '/home/ethan/IFT6135/IFT6135-2025/HW1_2025/assignment1_release/plots'
    os.makedirs(plots_dir, exist_ok=True)
    
    best_config = None
    best_overall_val_acc = 0.0
    
    # Dictionary to record best epochs for each candidate number of blocks
    best_epochs = {}
    
    for nb in candidate_num_blocks:
        config_str = f"MLPMixer_lr={fixed_lr}, num_blocks={nb}, patch={patch_size}, embed_dim={embed_dim}"
        tqdm.write(f"Running experiment with {config_str}")
        
        # Instantiate MLPMixer with current candidate number of blocks
        model = MLPMixer(
            num_classes=10,
            img_size=32,
            patch_size=patch_size,
            embed_dim=embed_dim,
            num_blocks=nb,
            drop_rate=0.0,
            activation='gelu'
        )
        model.to(args.device)
        
        optimizer = optim.Adam(model.parameters(), lr=fixed_lr, weight_decay=fixed_weight_decay)
        
        best_val_acc = 0.0
        best_epoch = 0
        no_improve = 0
        
        # Lists to record per-epoch metrics for plotting
        train_accs = []
        valid_accs = []
        epoch_numbers = []
        
        for epoch in range(max_epochs):
            tqdm.write(f"------ {config_str} | Epoch {epoch} ------")
            train_loss, train_acc = train(epoch, model, train_dataloader, optimizer, args.device, args.print_every)
            val_loss, val_acc = evaluate(epoch, model, valid_dataloader, args.device, args.print_every, mode="val")
            
            train_accs.append(train_acc)
            valid_accs.append(val_acc)
            epoch_numbers.append(epoch)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                no_improve = 0
            else:
                no_improve += 1
            
            if no_improve >= patience:
                tqdm.write(f"Early stopping triggered at epoch {epoch} for {config_str}.")
                break
        
        best_epochs[nb] = best_epoch
        tqdm.write(f"Best validation accuracy for {config_str} at epoch {best_epoch}: {best_val_acc:.3f}")
        
        if best_val_acc > best_overall_val_acc:
            best_overall_val_acc = best_val_acc
            best_config = config_str
        
        # Save a plot of training curves for this candidate
        plot_save_path = os.path.join(plots_dir, f"mlpmixer_numblocks_{nb}.png")
        plt.figure(figsize=(10, 6))
        plt.plot(epoch_numbers, train_accs, marker='o', label='Train Accuracy')
        plt.plot(epoch_numbers, valid_accs, marker='o', label='Validation Accuracy')
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(f"MLPMixer Training Curves\n{config_str}\nBest Epoch: {best_epoch}")
        plt.legend()
        plt.savefig(plot_save_path)
        plt.close()
        tqdm.write(f"Saved training curves plot to {plot_save_path}")
    
    print("Hyperparameter tuning for MLPMixer complete with early stopping.")
    print("Best epochs for each candidate num_blocks:")
    for nb, epoch in best_epochs.items():
        print(f"num_blocks: {nb} -> Best Epoch: {epoch}")
    print(f"Best overall config: {best_config} with validation accuracy: {best_overall_val_acc:.3f}")

if __name__ == "__main__":
    main()
