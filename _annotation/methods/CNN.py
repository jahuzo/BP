# Imports
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(r'/mnt/c/Users/jahuz/Links/BP/_annotation')

# paths
from paths import *


# Debugging and checks
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

input("Press Enter to continue...")
print("OK...moving on")

root_dir = result_dir

# Custom Dataset
class CustomDataset(Dataset):
    def __init__(self, folders, root_dir, transform=None):
        self.folders = folders
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.masks = []
        self.load_data()

    def load_data(self):
        for folder in self.folders:
            image_path = os.path.join(self.root_dir, folder, 'image.jpg')
            json_path = os.path.join(self.root_dir, folder, 'polygons.json')

            # Load the image
            image = Image.open(image_path)
            image = image.convert("RGB")  # Ensure image is RGB
            self.images.append(image)

            # Load the polygons
            with open(json_path, 'r') as f:
                polygons = json.load(f)

            # Create a mask for the polygons
            mask = np.zeros((256, 256), dtype=np.uint8)  # Assuming original size is 256x256
            for polygon in polygons:
                if polygon.get('label') == 'a':
                    points = np.array([[point['x'], point['y']] for point in polygon['points']], dtype=np.int32)
                    cv2.fillPoly(mask, [points], 1)

            self.masks.append(mask)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]

        if self.transform:
            image = self.transform(image)
            mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)  # Add channel dimension

        return image, mask

# Define transforms
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])

# Load training and testing data
train_folders = ['001', '002', '003', '004', '005', '006', '007']
test_folders = ['008', '009']

train_dataset = CustomDataset(train_folders, root_dir, transform=transform)
test_dataset = CustomDataset(test_folders, root_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Define the model
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.encoder1 = self.conv_block(3, 32)
        self.encoder2 = self.conv_block(32, 64)
        self.encoder3 = self.conv_block(64, 128)
        self.decoder1 = self.upconv_block(128, 64)
        self.decoder2 = self.upconv_block(64, 32)
        self.final_conv = nn.Conv2d(32, 1, kernel_size=1) # removed  activation='sigmoid'

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)

        dec1 = self.decoder1(enc3)
        dec1 = torch.cat((dec1, enc2), dim=1)  # Skip connection
        dec1 = self.decoder1(dec1)

        dec2 = self.decoder2(dec1)
        dec2 = torch.cat((dec2, enc1), dim=1)  # Skip connection
        output = self.final_conv(dec2)

        return torch.sigmoid(output)

# Instantiate the model
model = UNet().to(device)

# Define loss and optimizer
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Train the model
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

def calculate_metrics(outputs, masks):
    # Binarize outputs
    predicted = (outputs > 0.5).float()
    
    # Flatten the predicted and true masks for metric calculation
    predicted_flat = predicted.view(-1).cpu()
    masks_flat = masks.view(-1).cpu()

    # Calculate True Positives, False Positives, True Negatives, False Negatives
    TP = ((predicted_flat == 1) & (masks_flat == 1)).sum().item()
    TN = ((predicted_flat == 0) & (masks_flat == 0)).sum().item()
    FP = ((predicted_flat == 1) & (masks_flat == 0)).sum().item()
    FN = ((predicted_flat == 0) & (masks_flat == 1)).sum().item()

    # Calculate metrics
    IoU = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0
    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
    Accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
    Precision = TP / (TP + FP) if (TP + FP) > 0 else 0

    return IoU, FPR, TPR, Accuracy, Precision

# Evaluate the model
model.eval()
with torch.no_grad():
    total_loss = 0
    all_IoU = []
    all_FPR = []
    all_TPR = []
    all_Accuracy = []
    all_Precision = []

    for images, masks in test_loader:
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)
        loss = criterion(outputs, masks)
        total_loss += loss.item()

        # Calculate metrics for this batch
        IoU, FPR, TPR, Accuracy, Precision = calculate_metrics(outputs, masks)
        all_IoU.append(IoU)
        all_FPR.append(FPR)
        all_TPR.append(TPR)
        all_Accuracy.append(Accuracy)
        all_Precision.append(Precision)

average_loss = total_loss / len(test_loader)
average_IoU = sum(all_IoU) / len(all_IoU)
average_FPR = sum(all_FPR) / len(all_FPR)
average_TPR = sum(all_TPR) / len(all_TPR)
average_Accuracy = sum(all_Accuracy) / len(all_Accuracy)
average_Precision = sum(all_Precision) / len(all_Precision)

print(f"Average Test Loss: {average_loss:.4f}")
print(f"Average IoU: {average_IoU:.4f}")
print(f"Average FPR: {average_FPR:.4f}")
print(f"Average TPR (Recall): {average_TPR:.4f}")
print(f"Average Accuracy: {average_Accuracy:.4f}")
print(f"Average Precision: {average_Precision:.4f}")

# Save the model
model_dir = os.path.join(result_dir, 'model_UNet.pth')
os.makedirs(os.path.dirname(model_dir), exist_ok=True)
torch.save(model.state_dict(), model_dir)



