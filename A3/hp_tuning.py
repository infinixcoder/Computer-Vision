import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Import from our modularized ecosystem
from utils import CropData, FocalLoss, evaluate_model
from models import get_model

def get_dataloaders(args, batch_size):
    """Helper to re-initialize dataloaders since batch_size can change during tuning."""
    train_dataset = CropData(args.img_dir, args.train_csv)
    val_dataset = CropData(args.img_dir, args.val_csv)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    return train_dataset, train_loader, val_loader

def run_experiment(args, device, params, experiment_name):
    """Runs a single training loop and returns the history for plotting."""
    print(f"\n{'-'*50}")
    print(f"Running Experiment: {experiment_name}")
    print(f"Params: {params}")
    print(f"{'-'*50}")

    # 1. Dataloaders
    train_dataset, train_loader, val_loader = get_dataloaders(args, params['batch_size'])

    # 2. Model Init
    model = get_model("resnet18", num_classes=10).to(device)

    # 3. Optimizer, Loss, and Scheduler
    criterion = FocalLoss(alpha=params['alpha'], gamma=params['gamma'])
    optimizer = optim.Adam(model.parameters(), lr=params['lr'])
    
    scheduler = None
    if params['scheduler'] == 'step':
        # Halves the learning rate every 3 epochs
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    elif params['scheduler'] == 'cosine':
        # Gradually decreases LR following a cosine curve
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = {'train_loss': [], 'val_auc': []}
    best_val_auc = 0.0
    best_wts = copy.deepcopy(model.state_dict())

    # 4. Training Loop
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            
        # Step the scheduler if it exists
        if scheduler:
            scheduler.step()
            
        epoch_loss = running_loss / len(train_dataset)
        val_acc, val_f1, val_auc = evaluate_model(model, val_loader, device)
        
        # Log history for plotting
        history['train_loss'].append(epoch_loss)
        history['val_auc'].append(val_auc)
        
        print(f"Epoch {epoch+1}/{args.epochs} | Loss: {epoch_loss:.4f} | Val AUC: {val_auc:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_wts = copy.deepcopy(model.state_dict())

    return history, best_val_auc, best_wts

def plot_tuning_curves(all_histories, baseline_params, variations, save_path):
    """Generates the side-by-side plots for Train Loss and Val AUC."""
    print("\nGenerating tuning curves plot...")
    
    # We are testing 5 parameters: lr, batch_size, scheduler, alpha, gamma
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(25, 10))
    fig.suptitle('Hyperparameter Tuning: One-At-A-Time (OAT) Analysis', fontsize=20, fontweight='bold')
    
    param_keys = ['lr', 'batch_size', 'scheduler', 'alpha', 'gamma']
    epochs_range = range(1, len(next(iter(all_histories.values()))['train_loss']) + 1)
    
    for col_idx, param in enumerate(param_keys):
        # Top Row: Train Loss
        ax_loss = axes[0, col_idx]
        ax_loss.set_title(f'Varying {param.upper()}', fontsize=14)
        ax_loss.set_xlabel('Epochs')
        ax_loss.set_ylabel('Train Loss')
        ax_loss.grid(True, linestyle='--', alpha=0.7)
        
        # Bottom Row: Val AUC
        ax_auc = axes[1, col_idx]
        ax_auc.set_xlabel('Epochs')
        ax_auc.set_ylabel('Validation AUC')
        ax_auc.grid(True, linestyle='--', alpha=0.7)
        
        # Plot Baseline
        base_label = f"Base ({baseline_params[param]})"
        ax_loss.plot(epochs_range, all_histories['Baseline']['train_loss'], label=base_label, linewidth=3, color='black', linestyle='--')
        ax_auc.plot(epochs_range, all_histories['Baseline']['val_auc'], label=base_label, linewidth=3, color='black', linestyle='--')
        
        # Plot Variations
        for val in variations[param]:
            exp_name = f"{param}={val}"
            ax_loss.plot(epochs_range, all_histories[exp_name]['train_loss'], label=str(val), linewidth=2)
            ax_auc.plot(epochs_range, all_histories[exp_name]['val_auc'], label=str(val), linewidth=2)
            
        ax_loss.legend()
        ax_auc.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved successfully to {save_path}")

def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # --- 1. Define the Baseline Configuration ---
    baseline_params = {
        'lr': 1e-5,
        'batch_size': 32,
        'scheduler': 'none',
        'alpha': 0.5,
        'gamma': 1.0
    }
    
    # --- 2. Define the Variations ---
    # We will swap these in one at a time while keeping the rest at baseline
    variations = {
        'lr': [1e-6, 1e-7],
        'batch_size': [16, 64],
        'scheduler': ['step', 'cosine'],
        'alpha': [0.25, 1.0],
        'gamma': [0.25, 0.5]
    }
    
    all_histories = {}
    best_overall_auc = 0.0
    best_overall_wts = None
    best_overall_name = ""

    # --- 3. Run Baseline ---
    hist, auc, wts = run_experiment(args, device, baseline_params, "Baseline")
    all_histories['Baseline'] = hist
    best_overall_auc, best_overall_wts, best_overall_name = auc, wts, "Baseline"
    
    # --- 4. Run One-At-A-Time Variations ---
    for param, values in variations.items():
        for val in values:
            # Copy baseline, change exactly ONE parameter
            current_params = baseline_params.copy()
            current_params[param] = val
            exp_name = f"{param}={val}"
            
            hist, auc, wts = run_experiment(args, device, current_params, exp_name)
            all_histories[exp_name] = hist
            
            if auc > best_overall_auc:
                best_overall_auc = auc
                best_overall_wts = copy.deepcopy(wts)
                best_overall_name = exp_name

    # --- 5. Generate Plots ---
    plot_save_path = os.path.join(os.path.dirname(args.save_weights), "tuning_curves.png")
    plot_tuning_curves(all_histories, baseline_params, variations, plot_save_path)
    
    # --- 6. Save the Absolute Winner ---
    print(f"\n>>> Best Configuration Found: {best_overall_name} with AUC: {best_overall_auc:.4f} <<<")
    os.makedirs(os.path.dirname(args.save_weights), exist_ok=True)
    numpy_weights = {k: v.cpu().numpy() for k, v in best_overall_wts.items()}
    np.savez(args.save_weights, **numpy_weights)
    print(f"Saved winning weights to {args.save_weights}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", type=str, required=True)
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--val_csv", type=str, required=True)
    parser.add_argument("--save_weights", type=str, required=True, help="Path to save best weights (e.g., ./weights/best_tuned.npz)")
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    main(args)