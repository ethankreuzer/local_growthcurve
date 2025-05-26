"""
Q4_5.py

Train a ResNet18 model on CIFAR10 for 56 epochs with a learning rate of 0.0001 and zero weight decay.
During training, both training and validation loss/accuracy are evaluated.
After training, the model's visualize method is called to save the kernel visualization,
and the model weights are saved for future use.

Usage:
    python Q4_5.py
"""

import os
import time
import warnings

import torch
from torch import optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from torchvision.datasets import CIFAR10
from torchvision import transforms

from resnet18 import ResNet18
from utils import seed_experiment, cross_entropy_loss, compute_accuracy

def train(epoch, model, dataloader, optimizer, device, print_every=100):
    model.train()
    epoch_loss = 0.0
    epoch_acc = 0.0
    for i, (imgs, labels) in enumerate(dataloader):
        imgs = imgs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = cross_entropy_loss(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += compute_accuracy(outputs, labels).item()
        
        if i % print_every == 0:
            print(f"[TRAIN] Epoch: {epoch}, Batch: {i}, Loss: {loss.item():.5f}")
    
    avg_loss = epoch_loss / len(dataloader)
    avg_acc = epoch_acc / len(dataloader)
    print(f"== [TRAIN] Epoch: {epoch}, Avg Loss: {avg_loss:.5f}, Avg Acc: {avg_acc:.3f} ==\n")
    return avg_loss, avg_acc

def evaluate(epoch, model, dataloader, device, print_every=100, mode="val"):
    model.eval()
    epoch_loss = 0.0
    epoch_acc = 0.0
    with torch.no_grad():
        for i, (imgs, labels) in enumerate(dataloader):
            imgs = imgs.to(device)
            labels = labels.to(device)
            outputs = model(imgs)
            loss = cross_entropy_loss(outputs, labels)
            acc = compute_accuracy(outputs, labels)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
            if i % print_every == 0:
                print(f"[{mode.upper()}] Epoch: {epoch}, Batch: {i}, Loss: {loss.item():.5f}")
    avg_loss = epoch_loss / len(dataloader)
    avg_acc = epoch_acc / len(dataloader)
    print(f"== [{mode.upper()}] Epoch: {epoch}, Avg Loss: {avg_loss:.5f}, Avg Acc: {avg_acc:.3f} ==\n")
    return avg_loss, avg_acc

def main():
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        warnings.warn("CUDA is not available. Running on CPU.")
    
    # Set seed for reproducibility
    seed_experiment(42)
    
    # Hyperparameters
    learning_rate = 0.0001
    weight_decay = 0.0
    num_epochs = 56
    batch_size = 128  # Adjust batch size as needed
    
    data_root = './data'
    
    # Define transforms for training and validation data
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop((32, 32), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize([0.49139968, 0.48215841, 0.44653091],
                             [0.24703223, 0.24348513, 0.26158784])
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.49139968, 0.48215841, 0.44653091],
                             [0.24703223, 0.24348513, 0.26158784])
    ])
    
    # Load CIFAR10 dataset and create training and validation splits.
    full_train_dataset = CIFAR10(root=data_root, train=True, transform=train_transform, download=True)
    full_val_dataset = CIFAR10(root=data_root, train=True, transform=test_transform, download=True)
    
    # Split: Use 45,000 samples for training and 5,000 for validation.
    train_set, _ = random_split(full_train_dataset, [45000, 5000])
    _, val_set = random_split(full_val_dataset, [45000, 5000])
    
    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                  drop_last=True, pin_memory=True, num_workers=4)
    valid_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                                  drop_last=False, num_workers=4)
    
    # Instantiate ResNet18 model and move it to device
    model = ResNet18(num_classes=10)
    model.to(device)
    
    # Set up the optimizer with fixed learning rate and zero weight decay
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Training loop for 56 epochs, including evaluation on the validation set.
    for epoch in range(num_epochs):
        print(f"------ Epoch {epoch} ------")
        train_loss, train_acc = train(epoch, model, train_dataloader, optimizer, device, print_every=100)
        val_loss, val_acc = evaluate(epoch, model, valid_dataloader, device, print_every=100, mode="val")
        print(f"Epoch {epoch}: Train Loss: {train_loss:.5f}, Train Acc: {train_acc:.3f} | "
              f"Val Loss: {val_loss:.5f}, Val Acc: {val_acc:.3f}\n")
    
    print("Training complete.")
    
    # Save the model weightssqueue
    model_save_path = "/home/ethan/IFT6135/IFT6135-2025/HW1_2025/assignment1_release/resnet18_trained"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
    
    # Call the model's visualize method to save the kernel visualization.
    model.visualize(logdir="/home/ethan/IFT6135/IFT6135-2025/HW1_2025/assignment1_release/plots/Q4_5")
    print("Kernel visualization complete.")
    
    # Example of loading the model later for further plotting:
    '''
    model = ResNet18(num_classes=10)
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    model.to(device)
    print("Model loaded for further visualization or analysis.")
    '''

if __name__ == "__main__":
    main()
