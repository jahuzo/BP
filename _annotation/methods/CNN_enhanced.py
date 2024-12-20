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
from PIL import Image, ImageDraw  # Updated for augmentation
import json
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


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
        # Convolutional and BatchNorm layers remain the same
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.resblock1 = ResidualBlock(32, 32)
        self.dropout = nn.Dropout(0.5)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.resblock2 = ResidualBlock(64, 64)
        self.resblock3 = ResidualBlock(64, 64)

        # Use Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # Output size of 1x1
        self.fc1 = nn.Linear(64, 128)  # Input size is now 64 after global pooling
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.resblock1(x)
        x = self.dropout(x)

        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.dropout(x)

        x = self.global_pool(x)  # Global Average Pooling
        x = x.view(x.size(0), -1)  # Flatten to (batch_size, 64)

        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Instantiate the enhanced model
model = EnhancedCNNModel().to(device)  # Ensure model weights are moved to the same device as the input

# Define Cross Entropy Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
num_epochs = 25
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
        # Add a channel dimension
        inputs = inputs.unsqueeze(1).to(device)
        outputs = model(inputs)
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
            # Add channel dimension if needed
            if inputs.dim() == 3:
                inputs = inputs.unsqueeze(1)
            outputs = model(inputs)
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

def infer_and_update_polygons(model, data_dir, confidence_threshold=0.6):
    model.eval()
    model = model.to(device)
    w_size = 128
    transform = transforms.Compose([
        transforms.Resize((w_size, w_size)),
        transforms.ToTensor()
    ])

    window_size = w_size
    stride = w_size

    for folder in os.listdir(data_dir):
        if folder not in true_test_folders:
            continue

        folder_path = os.path.join(data_dir, folder)
        if os.path.isdir(folder_path):
            image_path = os.path.join(folder_path, 'image.jpg')
            json_path = os.path.join(folder_path, 'polygons.json')

            if not os.path.isfile(image_path):
                print(f"Image not found: {image_path}")
                continue

            # Convert image to grayscale
            input_image = Image.open(image_path).convert("L")
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

                        # Confirm the shape of input_tensor
                        # print(f"Input tensor shape: {input_tensor.shape}")

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

            # Apply Non-Maximum Suppression (NMS)
            def nms(boxes, overlap_thresh=0.2):  # Increased NMS threshold for more aggressive merging
                if len(boxes) == 0:
                    return []
                
                boxes_np = np.array([[box['polygon'][0]['x'], box['polygon'][0]['y'], 
                                      box['polygon'][2]['x'], box['polygon'][2]['y']] for box in boxes])
                
                x1 = boxes_np[:, 0]
                y1 = boxes_np[:, 1]
                x2 = boxes_np[:, 2]
                y2 = boxes_np[:, 3]
                
                areas = (x2 - x1 + 1) * (y2 - y1 + 1)
                order = np.argsort([box['label'] for box in boxes])  # Sorting based on confidence
                
                keep = []
                while order.size > 0:
                    i = order[-1]
                    keep.append(i)
                    
                    xx1 = np.maximum(x1[i], x1[order[:-1]])
                    yy1 = np.maximum(y1[i], y1[order[:-1]])
                    xx2 = np.minimum(x2[i], x2[order[:-1]])
                    yy2 = np.minimum(y2[i], y2[order[:-1]])
                    
                    w = np.maximum(0.0, xx2 - xx1 + 1)
                    h = np.maximum(0.0, yy2 - yy1 + 1)
                    inter = w * h
                    
                    iou = inter / (areas[i] + areas[order[:-1]] - inter)
                    
                    order = order[np.where(iou <= overlap_thresh)[0]]
                
                return [boxes[idx] for idx in keep]

            unique_boxes = nms(detected_boxes)

            if unique_boxes:
                updated_data = existing_data + unique_boxes
                with open(json_path, 'w') as f:
                    json.dump(updated_data, f, indent=4)
                print(f"{len(unique_boxes)} detected polygons saved in {json_path}")
            else:
                print(f"No polygons detected in {image_path}")
            
delete_detected_labels(result_dir)  #clear json file
infer_and_update_polygons(model, result_dir)
