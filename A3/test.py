import torch
import numpy as np
import argparse
from torch.utils.data import DataLoader
from utils import CropData, evaluate_model
from models import get_model

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Test Data
    test_dataset = CropData(args.img_dir, args.test_csv)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Initialize Model
    model = get_model(args.model, num_classes=10).to(device)

    # Load .npz Weights
    print(f"Loading weights from {args.weights_path}...")
    npz_weights = np.load(args.weights_path)
    state_dict = model.state_dict()
    for key in state_dict.keys():
        if key in npz_weights:
            state_dict[key] = torch.from_numpy(npz_weights[key])
    model.load_state_dict(state_dict)

    # Evaluate
    print("Running evaluation on test set...")
    test_acc, test_f1, test_auc = evaluate_model(model, test_loader, device)
    
    print("\n" + "="*40)
    print(f"RESULTS FOR {args.model.upper()}")
    print("="*40)
    print(f"Test Accuracy:  {test_acc:.4f}")
    print(f"Test Macro F1:  {test_f1:.4f}")
    print(f"Test Macro AUC: {test_auc:.4f}")
    print("="*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", type=str, required=True)
    parser.add_argument("--test_csv", type=str, required=True)
    parser.add_argument("--weights_path", type=str, required=True)
    parser.add_argument("--model", type=str, choices=["resnet18", "resnet18_se", "deit3", "deit3_dyt"], required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    main(args)