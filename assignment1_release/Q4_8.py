"""
Q4_8.py

Compare the gradient flow (norms of gradients at different layers) during backpropagation
for three architectures: MLP, ResNet18, and MLPMixer.

For each model, the script:
  1. Loads the corresponding model configuration from a JSON file.
  2. Instantiates the model using the configuration.
  3. Loads a batch of data from CIFAR10.
  4. Trains each model for 15 epochs using CrossEntropyLoss and SGD.
  5. Computes the L2 norm of gradients for each parameter per epoch.
  6. Plots the gradient norms across epochs for each model.

Usage:
    python Q4_8.py
"""

import json
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os

from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader, random_split

# Import models
from mlp import MLP
from resnet18 import ResNet18
from mlpmixer import MLPMixer

def get_gradient_norms(model):
    """
    Compute the L2 norm of the gradients for each parameter in the model.
    Returns a dictionary mapping parameter names to gradient norms.
    """
    grad_norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norms[name] = param.grad.norm().item()
        else:
            grad_norms[name] = 0.0
    return grad_norms

def plot_gradient_norms(gradient_history, model_names, save_path):
    """
    Plots the gradient norms for each model over epochs.
    
    Parameters:
      gradient_history: Dict where keys are model names and values are lists of gradient norms per epoch.
      model_names: List of model names.
      save_path: Path to save the plot.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for model_name in model_names:
        avg_grad_norms = [sum(epoch_norms.values()) / len(epoch_norms) for epoch_norms in gradient_history[model_name]]
        ax.plot(range(1, len(avg_grad_norms) + 1), avg_grad_norms, label=model_name)
    
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Average Gradient Norm")
    ax.set_title("Gradient Norms Over Training Epochs")
    ax.legend()
    
    plt.savefig(save_path)
    plt.show()
    
def train_model(model, model_name, train_loader, num_epochs=15, learning_rate=0.01, device="cpu"):
    """
    Trains a model and tracks gradient norms per epoch.
    
    Parameters:
      model: The model to train.
      model_name: Name of the model for logging.
      train_loader: DataLoader for training data.
      num_epochs: Number of training epochs.
      learning_rate: Learning rate for optimization.
      device: Device to train on ("cpu" or "cuda").
    
    Returns:
      List of gradient norms per epoch.
    """
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    gradient_history = []  # Store gradient norms per epoch
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            if isinstance(model, MLP):  # Flatten input for MLP
                images = images.view(images.size(0), -1)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            
            grad_norms = get_gradient_norms(model)
            gradient_history.append(grad_norms)  # Store gradient norms per epoch
            
            optimizer.step()
            running_loss += loss.item()
        
        avg_loss = running_loss / len(train_loader)
        print(f"{model_name} | Epoch [{epoch+1}/{num_epochs}] - Loss: {avg_loss:.4f}")

    return gradient_history

def main():
    # --------------------------
    # Load model configurations
    # --------------------------
    config_path = "/home/ethan/IFT6135/IFT6135-2025/HW1_2025/assignment1_release/model_configs/"
    
    with open(config_path + "mlp.json", "r") as f:
        mlp_config = json.load(f)
    with open(config_path + "resnet18.json", "r") as f:
        resnet18_config = json.load(f)
    with open(config_path + "mlpmixer.json", "r") as f:
        mlpmixer_config = json.load(f)

    # --------------------------
    # Instantiate models
    # --------------------------
    mlp_model = MLP(
        input_size=mlp_config["input_size"],
        hidden_sizes=mlp_config["hidden_sizes"],
        num_classes=mlp_config["num_classes"],
        activation="relu"
    )
    resnet_model = ResNet18(num_classes=resnet18_config["num_classes"])
    
    mlpmixer_model = MLPMixer(
        num_classes=mlpmixer_config["num_classes"],
        img_size=mlpmixer_config["img_size"],
        patch_size=mlpmixer_config["patch_size"],
        embed_dim=mlpmixer_config["embed_dim"],
        num_blocks=mlpmixer_config["num_blocks"],
        drop_rate=mlpmixer_config.get("drop_rate", 0.0),
        activation=mlpmixer_config.get("activation", "gelu")
    )

    # --------------------------
    # Prepare CIFAR10 Data
    # --------------------------
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop((32, 32), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize([0.49139968, 0.48215841, 0.44653091],
                             [0.24703223, 0.24348513, 0.26158784])
    ])

    data_root = "./data"
    full_train_dataset = CIFAR10(root=data_root, train=True, transform=transform_train, download=True)
    train_set, _ = random_split(full_train_dataset, [45000, 5000])

    batch_size = 16
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              drop_last=True, pin_memory=True, num_workers=4)

    # --------------------------
    # Train Each Model & Track Gradients
    # --------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_epochs = 15
    learning_rate = 0.01
    
    gradient_history = {}

    print("\nTraining MLP...")
    gradient_history["MLP"] = train_model(mlp_model, "MLP", train_loader, num_epochs, learning_rate, device)
    
    print("\nTraining ResNet18...")
    gradient_history["ResNet18"] = train_model(resnet_model, "ResNet18", train_loader, num_epochs, learning_rate, device)
    
    print("\nTraining MLPMixer...")
    gradient_history["MLPMixer"] = train_model(mlpmixer_model, "MLPMixer", train_loader, num_epochs, learning_rate, device)

    # --------------------------
    # Plot Gradient Norms Over Epochs
    # --------------------------
    save_path = "/home/ethan/IFT6135/IFT6135-2025/HW1_2025/assignment1_release/plots/Q4_8/gradient_norms.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plot_gradient_norms(gradient_history, ["MLP", "ResNet18", "MLPMixer"], save_path)

if __name__ == "__main__":
    main()
