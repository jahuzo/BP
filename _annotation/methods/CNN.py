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
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


class CNNModel(nn.Module):
    def __init__(self, input_size=(64, 64)):
        super(CNNModel, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        
        # Flattening size calculation
        example_input = torch.zeros(1, 3, input_size[0], input_size[1])
        self.flattened_size = self._get_flattened_size(example_input)

        # Fully connected layers
        self.fc1 = nn.Linear(self.flattened_size, 128)
        self.fc2 = nn.Linear(128, 2)  # 2 output classes

    def _get_flattened_size(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        return x.view(1, -1).size(1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x  # Return raw logits

model = CNNModel(input_size=(64, 64)).to(device)

# Define Cross Entropy Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
num_epochs = 10
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
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs.permute(0, 3, 1, 2))
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
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs.permute(0, 3, 1, 2))
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            val_accuracy += accuracy(outputs, labels) * inputs.size(0)
    val_loss /= len(val_loader.dataset)
    val_accuracy /= len(val_loader.dataset)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accuracies.append(train_accuracy)
    val_accuracies.append(val_accuracy)

    print(f"Epoch {epoch+1}/{num_epochs}, "
          f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
          f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
    

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

def infer_and_update_polygons(model, data_dir, confidence_threshold=0.6):  # Simplified inference
    model.eval()
    model.to(device)

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    window_size = 64
    stride = 32  # Simplified stride for faster inference

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

                        if predicted_class == 1:
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

            if detected_boxes:
                updated_data = existing_data + detected_boxes
                with open(json_path, 'w') as f:
                    json.dump(updated_data, f, indent=4)
                print(f"{len(detected_boxes)} detected polygons saved in {json_path}")
            else:
                print(f"No polygons detected in {image_path}")
              
infer_and_update_polygons(model, result_dir)

