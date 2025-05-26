"""
Q4_6.py

Train an MLPMixer model on CIFAR10 using the best overall configuration:
  - Learning rate: 1e-4
  - num_blocks: 6
  - Patch size: 4
  - embed_dim: 512
  - Activation: GELU, mlp_ratio: (0.5, 4.0), Dropout: 0.0
Train for 62 epochs and then save the model’s state dictionary.

Usage:
    python Q4_6.py
"""

import os
import warnings
import torch
from torch import optim
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10
from torchvision import transforms
import matplotlib.pyplot as plt

# Import your MLPMixer model and utility functions
from mlpmixer import MLPMixer
from utils import seed_experiment, cross_entropy_loss, compute_accuracy, to_device

def train_epoch(epoch, model, dataloader, optimizer, device, print_every=100):
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    for i, (imgs, labels) in enumerate(dataloader):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = cross_entropy_loss(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        running_acc += compute_accuracy(outputs, labels).item()
        
        if i % print_every == 0:
            print(f"[TRAIN] Epoch {epoch}, Iter {i}, Loss: {loss.item():.5f}")
    
    avg_loss = running_loss / len(dataloader)
    avg_acc = running_acc / len(dataloader)
    print(f"Epoch {epoch} -- Train Loss: {avg_loss:.5f}  Train Acc: {avg_acc:.3f}")
    return avg_loss, avg_acc

def evaluate_epoch(epoch, model, dataloader, device, print_every=100):
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    with torch.no_grad():
        for i, (imgs, labels) in enumerate(dataloader):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = cross_entropy_loss(outputs, labels)
            
            running_loss += loss.item()
            running_acc += compute_accuracy(outputs, labels).item()
            
            if i % print_every == 0:
                print(f"[VAL] Epoch {epoch}, Iter {i}, Loss: {loss.item():.5f}")
    
    avg_loss = running_loss / len(dataloader)
    avg_acc = running_acc / len(dataloader)
    print(f"Epoch {epoch} -- Val Loss: {avg_loss:.5f}  Val Acc: {avg_acc:.3f}")
    return avg_loss, avg_acc

def main():
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        warnings.warn("CUDA not available, running on CPU.")
    
    # For reproducibility
    seed_experiment(42)
    
    # Best overall configuration hyperparameters
    learning_rate = 1e-4
    num_blocks = 6
    patch_size = 4
    embed_dim = 512
    drop_rate = 0.0
    activation = "gelu"
    mlp_ratio = (0.5, 4.0)
    num_classes = 10
    img_size = 32
    batch_size = 128
    num_epochs = 62  # Train for exactly 62 epochs
    
    # Define data transforms
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop((img_size, img_size), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
    ])
    
    data_root = "./data"
    # Load CIFAR10 and create train/validation splits (45k/5k)
    full_train = CIFAR10(root=data_root, train=True, transform=train_transform, download=True)
    full_val = CIFAR10(root=data_root, train=True, transform=test_transform, download=True)
    train_set, _ = random_split(full_train, [45000, 5000])
    _, val_set = random_split(full_val, [45000, 5000])
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Instantiate the MLPMixer model with best overall config
    model = MLPMixer(
        num_classes=num_classes,
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        num_blocks=num_blocks,
        drop_rate=drop_rate,
        activation=activation,
        mlp_ratio=mlp_ratio
    )
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop for exactly 62 epochs
    for epoch in range(num_epochs):
        print(f"---------- Epoch {epoch} ----------")
        train_epoch(epoch, model, train_loader, optimizer, device)
        evaluate_epoch(epoch, model, val_loader, device)
    
    print("Training complete.")
    
    # Save the model state dictionary
    model_save_path = "/home/ethan/IFT6135/IFT6135-2025/HW1_2025/assignment1_release/mlpmixer_trained.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

if __name__ == "__main__":
    main()
