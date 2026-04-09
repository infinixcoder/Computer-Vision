# COL780 Assignment 3: Image Classification and Attention Mechanisms

This repository contains the modularized Python scripts for training and evaluating various Convolutional Neural Networks (CNNs) and Vision Transformers (ViTs) on a 10-class land-use image classification dataset.

## 📂 Project Structure

* `utils.py`: Contains the custom PyTorch `Dataset` class (`CropData`), the `FocalLoss` implementation, and the `evaluate_model` metric calculation function.
* `models.py`: Contains the model definitions, including the Squeeze-and-Excitation (`SEBlock`) wrapper for ResNet and the Dynamic Tanh (`DynamicTanh`) modification for DeiT.
* `train.py`: The unified training script. Uses command-line arguments to dynamically switch between model architectures, loss functions, and hyperparameters.
* `test.py`: The unified evaluation script. Loads saved `.npz` weights into a specified architecture and calculates Final Accuracy, Macro F1, and Macro ROC-AUC on the test set.

## ⚙️ Prerequisites

Ensure you have the required libraries installed before running the scripts:

```bash
pip install torch torchvision timm pandas numpy opencv-python scikit-learn
```

## 🗄️ Dataset Setup

Ensure your dataset is organized with a main image directory and corresponding CSV files containing Filename and Label columns.

Example structure:
```
./A3_Dataset/
├── images/ (contains all .jpg/.png files)
├── train.csv
├── val.csv
└── test.csv
```

## 🚀 Training Instructions

The ```train.py``` script supports the following key arguments:

```--model```: Architecture to train (```resnet18, resnet18_se, deit3, deit3_dyt```).

```--loss```: Loss function to use (```ce``` for Cross-Entropy, ```focal``` for Focal Loss).

```--alpha / --gamma```: Focal loss hyperparameters (only active if ```--loss focal``` is set).

```--epochs / --batch_size / --lr```: Standard training hyperparameters.

### Task 1.1: Baseline ResNet-18

With standard Cross-Entropy Loss:
``` bash
python train.py \
  --img_dir ./A3_Dataset \
  --train_csv ./A3_Dataset/train.csv \
  --val_csv ./A3_Dataset/val.csv \
  --save_path ./weights/resnet18_ce.npz \
  --model resnet18 \
  --loss ce
```

With Focal Loss (Ablation Study):
``` bash
python train.py \
  --img_dir ./A3_Dataset \
  --train_csv ./A3_Dataset/train.csv \
  --val_csv ./A3_Dataset/val.csv \
  --save_path ./weights/resnet18_focal.npz \
  --model resnet18 \
  --loss focal \
  --alpha 0.5 \
  --gamma 2.0
```

### Task 1.2: SE-ResNet-18
```bash 
python train.py \
  --img_dir ./A3_Dataset \
  --train_csv ./A3_Dataset/train.csv \
  --val_csv ./A3_Dataset/val.csv \
  --save_path ./weights/resnet18_se.npz \
  --model resnet18_se \
  --loss ce
```

### Task 2.1: Baseline DeiT-3 Small
``` bash
python train.py \
  --img_dir ./A3_Dataset \
  --train_csv ./A3_Dataset/train.csv \
  --val_csv ./A3_Dataset/val.csv \
  --save_path ./weights/deit3.npz \
  --model deit3 \
  --loss ce
```

### Task 2.2: DeiT-3 with Dynamic Tanh (DyT)
``` bash
python train.py \
  --img_dir ./A3_Dataset \
  --train_csv ./A3_Dataset/train.csv \
  --val_csv ./A3_Dataset/val.csv \
  --save_path ./weights/deit3_dyt.npz \
  --model deit3_dyt \
  --loss ce \
  --epochs 10
```

## 🧪 Testing Instructions
To evaluate a trained model, pass the corresponding .npz weights file and specify the architecture so the script knows how to load the tensors correctly.

### Evaluating the Baseline ResNet-18:
``` bash
python test.py \
  --img_dir ./A3_Dataset \
  --test_csv ./A3_Dataset/test.csv \
  --weights_path ./weights/resnet18_ce.npz \
  --model resnet18
```
### Evaluating the DeiT-3 (DyT):
``` bash
python test.py \
  --img_dir ./A3_Dataset \
  --test_csv ./A3_Dataset/test.csv \
  --weights_path ./weights/deit3_dyt.npz \
  --model deit3_dyt
```

## Grad-Cam Visualizations
```bash
python visualize.py \
  --img_dir ./A3_Dataset \
  --test_csv ./A3_Dataset/test.csv \
  --weights_resnet ./weights/best_resnet18_focal_weights.npz \
  --weights_se ./weights/best_resnet18_se_focal_weights.npz \
  --weights_deit ./weights/best_deit3_focal_weights.npz \
  --save_plot ./plots/bonus_task_visualizations_focal.png
```


## Hyperparameter Tuning Script
``` bash
python hp_tuning.py \
  --img_dir ./A3_Dataset \
  --train_csv ./A3_Dataset/train.csv \
  --val_csv ./A3_Dataset/val.csv \
  --save_weights ./weights best_resnet18_focal_tuned_weights.npz \
  --epochs 10
```

## 📝 Additional Notes

Early Stopping: The training script automatically tracks the Validation Macro ROC-AUC and saves the weights of the best-performing epoch to the specified --save_path.

Device Management: The scripts automatically detect and utilize an available CUDA GPU. If no GPU is found, they default to CPU execution.



