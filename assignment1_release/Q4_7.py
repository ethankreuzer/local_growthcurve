"""
Q4_7.py

Investigate how the width of the token-mixing and channel-mixing MLP layers affects
the MLPMixer model’s generalization ability. In this experiment, we vary the hidden width
of both MLP layers (i.e., the capacity of the token-mixing and channel-mixing MLPs) by
modifying the mlp_ratio. We run experiments with widths of 256, 512, and 1024. For each
experiment, training and validation losses and accuracies are recorded over epochs. The
script also prints out the number of model parameters and the overall training time.
At the end, four plots are generated (training loss, validation loss, training accuracy,
validation accuracy) with epochs on the x-axis and the legend indicating the MLP width.

Usage:
    python Q4_7.py
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
from mlpmixer import MLPMixer
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
            "CUDA is not available, forcing device='cpu'."
        )
        args.device = "cpu"
    if args.device == "cpu":
        warnings.warn(
            "Running on CPU. Consider reducing batch_size if memory is an issue."
        )
    
    # Force the use of the MLPMixer architecture for this experiment.
    args.model = "mlpmixer"

    # Load the base model configuration from mlpmixer.json
    base_config_path = '/home/ethan/IFT6135/IFT6135-2025/HW1_2025/assignment1_release/model_configs/mlpmixer.json'
    with open(base_config_path, 'r') as f:
        base_model_config = json.load(f)
    
    # Candidate widths for both token-mixing and channel-mixing MLP layers
    # (These widths will be achieved by setting mlp_ratio = (width/embed_dim, width/embed_dim)).
    candidate_widths = [128, 256, 512, 1024]
    
    # Define transforms and load CIFAR10 dataset
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
    
    train_dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True, pin_memory=True, num_workers=4)
    valid_dataloader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=4)
    
    # Seed for reproducibility
    seed_experiment(args.seed)
    
    # Prepare to record logs and legends for plotting
    log_dirs = []
    legend_names = []
    
    # Loop over each candidate width
    for width in candidate_widths:
        tqdm.write(f"Running experiment with MLP width: {width}")
        log_dir = f'/home/ethan/IFT6135/IFT6135-2025/HW1_2025/assignment1_release/logs/mlpmixer_width_{width}'
        os.makedirs(log_dir, exist_ok=True)
        log_dirs.append(log_dir)
        legend_names.append(f"width={width}")
        
        # Update model configuration: set mlp_ratio such that:
        # mlp_ratio[0] = width / embed_dim and mlp_ratio[1] = width / embed_dim.
        embed_dim = base_model_config['embed_dim']  # from the base config
        #custom_mlp_ratio = (width / embed_dim, width / embed_dim)
        
        # Build the MLPMixer model.
        # (This instantiation assumes that MLPMixer accepts an extra parameter `mlp_ratio`.)
        model = MLPMixer(
            num_classes=base_model_config['num_classes'],
            img_size=base_model_config['img_size'],
            patch_size=base_model_config['patch_size'],
            embed_dim=width,
            num_blocks=base_model_config['num_blocks'],
            drop_rate=base_model_config.get('drop_rate', 0.0),
            activation=base_model_config.get('activation', 'gelu'),
            
        )
        model.to(args.device)
        
        # Print total number of model parameters
        total_params = sum(p.numel() for p in model.parameters())
        tqdm.write(f"MLPMixer with width={width} (mlp_ratio={width}) has {total_params} parameters.")
        
        # Set up the optimizer (using Adam with the default learning rate from args)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        
        # Record training statistics for this experiment
        train_losses, valid_losses = [], []
        train_accs, valid_accs = [], []
        total_train_time = 0
        
        for epoch in range(args.epochs):
            tqdm.write(f"====== Width: {width} | Epoch {epoch} ======>")
            loss, acc, train_time = train(epoch, model, train_dataloader, optimizer, args.device, args.print_every)
            total_train_time += train_time
            train_losses.append(loss)
            train_accs.append(acc)
            loss, acc, _ = evaluate(epoch, model, valid_dataloader, args.device, args.print_every, mode="val")
            valid_losses.append(loss)
            valid_accs.append(acc)
        
        tqdm.write(f"Total training time for width={width}: {total_train_time:.2f} seconds.")
        
        # Save the results for this experiment in a JSON file
        results = {
            "train_losses": train_losses,
            "valid_losses": valid_losses,
            "train_accs": train_accs,
            "valid_accs": valid_accs,
            "total_params": total_params,
            "total_train_time": total_train_time
        }
        with open(os.path.join(log_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=4)
    
    # After all experiments, generate plots for the four metrics.
    plots_dir = '/home/ethan/IFT6135/IFT6135-2025/HW1_2025/assignment1_release/plots/Q4_7'
    os.makedirs(plots_dir, exist_ok=True)
    generate_plots(log_dirs, legend_names, plots_dir)
    
    print("Experiments completed and plots generated in the 'plots' directory.")

if __name__ == "__main__":
    main()
