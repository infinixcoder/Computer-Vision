import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import argparse
import matplotlib.pyplot as plt

# Import Grad-CAM tools
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# Import from our modularized ecosystem
from utils import CropData
from models import get_model

# --- 1. Helper Function: Load .npz weights ---
def load_npz_to_model(model, filepath):
    """Loads a .npz file and applies it to a PyTorch model's state_dict."""
    if not os.path.exists(filepath):
        print(f"Warning: File {filepath} not found. Using untrained weights.")
        return
        
    npz_weights = np.load(filepath)
    state_dict = model.state_dict()
    
    for key in state_dict.keys():
        if key in npz_weights:
            state_dict[key] = torch.from_numpy(npz_weights[key])
        else:
            print(f"Warning: Missing key '{key}' in the .npz file.")
            
    model.load_state_dict(state_dict)

# --- 2. Helper Function: Get one sample per class ---
def get_class_samples(dataset, num_classes=10):
    """Finds and returns exactly one image per class from the dataset."""
    class_samples = {}
    for i in range(len(dataset)):
        img, label = dataset[i]
        lbl_idx = label.item()
        
        if lbl_idx not in class_samples:
            class_samples[lbl_idx] = img
            
        if len(class_samples) == num_classes:
            break
            
    # Sort dictionary by key so classes 0-9 are in order
    return {k: class_samples[k] for k in sorted(class_samples)}

# --- 3. Helper Function: Denormalize Image for Plotting ---
def denormalize(img_tensor):
    """Converts a normalized PyTorch tensor back to a standard RGB image."""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    img = img_tensor.cpu().numpy()
    img = np.moveaxis(img, 0, -1) # [C, H, W] to [H, W, C]
    img = std * img + mean
    img = np.clip(img, 0, 1)
    return img

# --- 4. Attention Extraction Hook for DeiT ---
deit_attention_map = []

def deit_attention_hook(module, input, output):
    """Hooks into the attention dropout layer to grab the attention probabilities."""
    deit_attention_map.clear()
    deit_attention_map.append(output.detach().cpu())

# --- MAIN EXECUTION ---
def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Test Data
    test_dataset = CropData(img_dir_path=args.img_dir, csv_file_path=args.test_csv)
    print("Gathering sample images...")
    samples = get_class_samples(test_dataset)

    # 2. Initialize Models and Load Weights
    # --- Model 1: Base ResNet-18 ---
    resnet_base = get_model("resnet18", num_classes=10)
    load_npz_to_model(resnet_base, args.weights_resnet)
    resnet_base.eval().to(device)
    
    # --- Model 2: SE-ResNet-18 ---
    resnet_se = get_model("resnet18_se", num_classes=10)
    load_npz_to_model(resnet_se, args.weights_se)
    resnet_se.eval().to(device)
    
    # --- Model 3: DeiT-3 Small ---
    deit = get_model("deit3", num_classes=10)
    load_npz_to_model(deit, args.weights_deit)
    deit.eval().to(device)

    # ---> NEW FIX: Disable Fused Attention on the last block <---
    if hasattr(deit.blocks[-1].attn, 'fused_attn'):
        deit.blocks[-1].attn.fused_attn = False

    # Register the hook on the last attention block of DeiT
    deit.blocks[-1].attn.attn_drop.register_forward_hook(deit_attention_hook)

    # 3. Setup Grad-CAM for ResNets
    target_layers_base = [resnet_base.layer4[-1]]
    cam_base = GradCAM(model=resnet_base, target_layers=target_layers_base)
    
    target_layers_se = [resnet_se.se4] 
    cam_se = GradCAM(model=resnet_se, target_layers=target_layers_se)

    # 4. Generate Visualizations
    num_classes = len(samples)
    fig, axes = plt.subplots(num_classes, 4, figsize=(16, 4 * num_classes))
    
    # Column titles
    cols = ["Original Image", "Task 1.1: ResNet Grad-CAM", "Task 1.2: SE-ResNet Grad-CAM", "Task 2.1: DeiT Attention"]
    for ax, col in zip(axes[0], cols):
        ax.set_title(col, fontsize=14, pad=10)

    for i, (class_idx, img_tensor) in enumerate(samples.items()):
        input_tensor = img_tensor.unsqueeze(0).to(device)
        rgb_img = denormalize(img_tensor)
        
        # --- A. Original Image ---
        axes[i, 0].imshow(rgb_img)
        axes[i, 0].set_ylabel(f"Class {class_idx}", fontsize=12)
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])

        # --- B. ResNet-18 Grad-CAM ---
        grayscale_cam = cam_base(input_tensor=input_tensor, targets=[ClassifierOutputTarget(class_idx)])[0]
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        axes[i, 1].imshow(visualization)
        axes[i, 1].axis('off')

        # --- C. SE-ResNet-18 Grad-CAM ---
        grayscale_cam_se = cam_se(input_tensor=input_tensor, targets=[ClassifierOutputTarget(class_idx)])[0]
        visualization_se = show_cam_on_image(rgb_img, grayscale_cam_se, use_rgb=True)
        axes[i, 2].imshow(visualization_se)
        axes[i, 2].axis('off')

        # --- D. DeiT-3 Attention Map ---
        _ = deit(input_tensor) # Trigger the forward pass to fill our hook list
        
        attn_matrix = deit_attention_map[0]
        attn_matrix = attn_matrix.mean(dim=1).squeeze(0) # Average across heads
        cls_attention = attn_matrix[0, 1:] # Extract [CLS] attention to patches
        cls_attention = cls_attention.reshape(14, 14).numpy()
        
        # Normalize and overlay
        cls_attention = (cls_attention - cls_attention.min()) / (cls_attention.max() - cls_attention.min())
        cls_attention_resized = cv2.resize(cls_attention, (224, 224))
        
        heatmap = cv2.applyColorMap(np.uint8(255 * cls_attention_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        heatmap = np.float32(heatmap) / 255
        
        overlay = 0.5 * heatmap + 0.5 * rgb_img
        overlay = np.clip(overlay, 0, 1)
        
        axes[i, 3].imshow(overlay)
        axes[i, 3].axis('off')

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.save_plot), exist_ok=True)
    plt.savefig(args.save_plot, dpi=300)
    print(f"Saved visualization grid to '{args.save_plot}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", type=str, required=True, help="Path to image directory")
    parser.add_argument("--test_csv", type=str, required=True, help="Path to test.csv")
    parser.add_argument("--weights_resnet", type=str, required=True, help="Path to best ResNet-18 weights (.npz)")
    parser.add_argument("--weights_se", type=str, required=True, help="Path to best SE-ResNet-18 weights (.npz)")
    parser.add_argument("--weights_deit", type=str, required=True, help="Path to best DeiT-3 weights (.npz)")
    parser.add_argument("--save_plot", type=str, default="./visualizations.png", help="Path to save the output plot")
    
    args = parser.parse_args()
    main(args)