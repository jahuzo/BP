import sys

# Paths import (update the path accordingly)
sys.path.append(r'/mnt/c/Users/jahuz/Links/BP/_annotation')

from paths import *
from train_gen import *

import random
from sklearn.model_selection import train_test_split  # Added for stratified splitting

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageDraw  # Updated for augmentation
import json
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Added a learning rate scheduler # NEW!
from torch.optim.lr_scheduler import StepLR


# Define the PyTorch model equivalent to the provided Keras model
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = torch.relu(out)
        return out

class EnhancedCNNModel(nn.Module):
    def __init__(self):
        super(EnhancedCNNModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.resblock1 = ResidualBlock(32, 32)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.6)  # Increased dropout rate for more regularization

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.resblock2 = ResidualBlock(64, 64)

        example_input = torch.zeros(1, 3, 64, 64)
        self.flattened_size = self._get_flattened_size(example_input)

        self.fc1 = nn.Linear(self.flattened_size, 128)
        self.fc2 = nn.Linear(128, 2)

    def _get_flattened_size(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.resblock1(x)
        x = self.pool(x)
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.resblock2(x)
        x = self.pool(x)
        return x.view(1, -1).size(1)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.resblock1(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.resblock2(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = x.reshape(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return torch.softmax(x, dim=1)

# Instantiate the enhanced model
model = EnhancedCNNModel()

# Define the loss function (Focal Loss) and optimizer
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss()

    def forward(self, inputs, targets):
        logpt = -self.ce(inputs, targets)
        pt = torch.exp(logpt)
        focal_loss = -((1 - pt) ** self.gamma) * logpt
        return self.alpha * focal_loss

criterion = FocalLoss(alpha=1, gamma=2)
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=0.00001)

class EarlyStopping:
    def __init__(self, patience=5, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print("Early stopping triggered.")

early_stopping = EarlyStopping(patience=5, verbose=True)

# Training loop
num_epochs = 15
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

def accuracy(preds, labels):
    _, pred_classes = torch.max(preds, 1)
    return (pred_classes == labels).sum().item() / labels.size(0)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    running_accuracy = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs.permute(0, 3, 1, 2))  # Permute dimensions for PyTorch [batch, channels, height, width]
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_accuracy += accuracy(outputs, labels) * inputs.size(0)
    train_loss = running_loss / len(train_loader.dataset)
    train_accuracy = running_accuracy / len(train_loader.dataset)

    model.eval()
    val_loss, val_accuracy = 0.0, 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs.permute(0, 3, 1, 2))
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            val_accuracy += accuracy(outputs, labels) * inputs.size(0)
    val_loss /= len(val_loader.dataset)
    val_accuracy /= len(val_loader.dataset)

    # Step the scheduler # NEW!
    scheduler.step(val_loss)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accuracies.append(train_accuracy)
    val_accuracies.append(val_accuracy)

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
    
    early_stopping(val_loss)
    if early_stopping.early_stop:
        break

# Plot training & validation accuracy and loss
plt.figure(figsize=(12, 4))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

#summary(model, (3, 64, 64))

def infer_and_update_polygons(model, data_dir, confidence_threshold=0.2):  # Lowered confidence threshold to 0.2
    model.eval()
    model.to(device)

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    window_size = 64
    stride = 32

    for folder in os.listdir(data_dir):
        if folder not in ['008', '009']:
            continue

        folder_path = os.path.join(data_dir, folder)
        if os.path.isdir(folder_path):
            image_path = os.path.join(folder_path, 'image.jpg')
            json_path = os.path.join(folder_path, 'polygons.json')

            if not os.path.isfile(image_path):
                print(f"Image not found: {image_path}")
                continue

            input_image = Image.open(image_path).convert("RGB")
            image_width, image_height = input_image.size

            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    existing_data = json.load(f)
            else:
                existing_data = []

            detected_boxes = []

            with torch.no_grad():
                for y in range(0, image_height - window_size + 1, stride):
                    for x in range(0, image_width - window_size + 1, stride):
                        patch = input_image.crop((x, y, x + window_size, y + window_size))
                        input_tensor = transform(patch).unsqueeze(0).to(device)

                        predictions = model(input_tensor)
                        predicted_probs = torch.softmax(predictions, dim=1)
                        predicted_class = torch.argmax(predicted_probs, dim=1).item()
                        confidence = predicted_probs[0, predicted_class].item()

                        if predicted_class == 1 and confidence > confidence_threshold:
                            detected_box = {
                                "label": "detected",
                                "polygon": [
                                    {"x": x, "y": y},
                                    {"x": x + window_size, "y": y},
                                    {"x": x + window_size, "y": y + window_size},
                                    {"x": x, "y": y + window_size}
                                ]
                            }
                            detected_boxes.append(detected_box)
                            draw = ImageDraw.Draw(input_image)
                            draw.rectangle([x, y, x + window_size, y + window_size], outline="red", width=2)

            if len(detected_boxes) in [1, 150]:
                unique_boxes = {json.dumps(box, sort_keys=True): box for box in existing_data + detected_boxes}
                updated_data = list(unique_boxes.values())
                ndetected = len(detected_boxes)
                with open(json_path, 'w') as f:
                    json.dump(updated_data, f, indent=4)
                print(f"{ndetected} detected polygons saved in {json_path}")
            else:
                print(f"No polygons detected in {image_path}")
              
infer_and_update_polygons(model, result_dir)

