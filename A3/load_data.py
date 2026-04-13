import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

#Dataset loading class 
class CropData(Dataset):
    def __init__(self, img_dir_path, csv_file_path):
        self.img_dir_path = img_dir_path
        self.df = pd.read_csv(csv_file_path)
        self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_name = row["Filename"]
        label = row["Label"]

        img_path = os.path.join(self.img_dir_path, image_name)

        # Read image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(label).long()

        return image, label


#Example usage
if __name__ == "__main__":

    img_dir = "./A3_Dataset"
    csv_path = "./A3_Dataset/train.csv"

    dataset = CropData(
        img_dir_path=img_dir,
        csv_file_path=csv_path
    )

    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4
    )
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    for images, labels in dataloader:
        print(images.shape)  # [B, 3, 224, 224]
        print(labels.shape)  # [B]
        image = images[0].detach().numpy()
        image = np.moveaxis(image, 0, -1)
        image = (image*std) + mean
        cv2.imwrite("IMG.png", image*255)
        print(labels[0])

        break