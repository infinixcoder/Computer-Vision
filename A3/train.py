import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
from torch.utils.data import DataLoader
from utils import CropData, FocalLoss, evaluate_model
from models import get_model

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data
    train_dataset = CropData(args.img_dir, args.train_csv)
    val_dataset = CropData(args.img_dir, args.val_csv)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Model
    model = get_model(args.model, num_classes=10).to(device)

    # Optimizer & Loss
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    if args.loss == "ce":
        criterion = nn.CrossEntropyLoss()
    elif args.loss == "focal":
        criterion = FocalLoss(alpha=args.alpha, gamma=args.gamma)

    # Training Loop
    best_val_auc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    print(f"Starting training for {args.model} using {args.loss} loss...")
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
            
        epoch_loss = running_loss / len(train_dataset)
        val_acc, val_f1, val_auc = evaluate_model(model, val_loader, device)
        
        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {epoch_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f} | Val AUC: {val_auc:.4f}")
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_wts = copy.deepcopy(model.state_dict())
            print("--> Validation AUC improved! Saving weights.")

    # Save weights
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    numpy_weights = {k: v.cpu().numpy() for k, v in best_model_wts.items()}
    np.savez(args.save_path, **numpy_weights)
    print(f"Training Complete. Best weights saved to {args.save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", type=str, required=True)
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--val_csv", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--model", type=str, choices=["resnet18", "resnet18_se", "deit3", "deit3_dyt"], required=True)
    parser.add_argument("--loss", type=str, choices=["ce", "focal"], default="ce")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=2.0)
    args = parser.parse_args()
    main(args)