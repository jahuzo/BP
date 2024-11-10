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
        if residual.shape == out.shape:
            out += residual  # Ensure residual match the output shape
        out = torch.relu(out)
        return out

class EnhancedCNNModel(nn.Module):
    def __init__(self):
        super(EnhancedCNNModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.resblock1 = ResidualBlock(32, 32)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)  # Adjusted dropout rate for regularization

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.resblock2 = ResidualBlock(64, 64)

        self.resblock3 = ResidualBlock(64, 64)  # Adding an additional residual block for deeper learning

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
        x = self.resblock3(x)
        x = self.pool(x)
        return x.view(1, -1).size(1)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.resblock1(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.pool(x)
        x = self.dropout(x)
        x = x.reshape(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Instantiate the enhanced model
model = EnhancedCNNModel()
model = model.to(device)  # Ensure model is moved to the appropriate device

# Define the loss function (CrossEntropyLoss) and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = StepLR(optimizer, step_size=5, gamma=0.5)  # Adjusted learning rate scheduler

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
        outputs = model(inputs.permute(0, 3, 1, 2))  # Ensure correct input shape for Conv2D
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
            outputs = model(inputs.permute(0, 3, 1, 2))  # Ensure correct input shape for Conv2D
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
    
    scheduler.step()

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

#import torchvision

def infer_and_update_polygons(model, data_dir, confidence_threshold=0.80):  # Increased confidence threshold
    model.eval()
    model.to(device)

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Added normalization to improve accuracy
    ])

    window_size = 64
    stride = 48  # Reduced stride for more overlap, improving detection accuracy

    min_box_size = 30  # Increased minimum size of detected box to reduce small false positives

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
                        confidence = predicted_probs[0, predicted_class].item() * 1.1

                        if predicted_class == 1 and confidence > confidence_threshold:
                            box_width = window_size
                            box_height = window_size
                            if box_width >= min_box_size and box_height >= min_box_size:
                                detected_boxes.append({
                                    "x1": x, "y1": y, 
                                    "x2": x + window_size, "y2": y + window_size,
                                    "score": confidence
                                })

            # Apply NMS
            if detected_boxes:
                boxes_tensor = torch.tensor([[box['x1'], box['y1'], box['x2'], box['y2'], box['score']] for box in detected_boxes])
                nms_indices = torchvision.ops.nms(boxes_tensor[:, :4], boxes_tensor[:, 4], iou_threshold=0.55)  # Reduced IoU threshold to reduce overlapping boxes
                unique_boxes = [detected_boxes[i] for i in nms_indices]

                # Convert back to JSON format and save
                updated_data = existing_data + [{
                    "label": "detected",
                    "polygon": [
                        {"x": box['x1'], "y": box['y1']},
                        {"x": box['x2'], "y": box['y1']},
                        {"x": box['x2'], "y": box['y2']},
                        {"x": box['x1'], "y": box['y2']}
                    ]
                } for box in unique_boxes]

                with open(json_path, 'w') as f:
                    json.dump(updated_data, f, indent=4)
                print(f"{len(unique_boxes)} detected polygons saved in {json_path}")
            else:
                print(f"No polygons detected in {image_path}")
            
delete_detected_labels(result_dir)  #clear json file
infer_and_update_polygons(model, result_dir)
