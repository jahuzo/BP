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

# Load a pre-trained Faster R-CNN model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

# Adjust the classifier to detect only one class ('a' + background)
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
    model.roi_heads.box_predictor.cls_score.in_features, 2  # background + 'a'
)

# Freeze backbone layers initially to fine-tune the head
for param in model.backbone.parameters():
    param.requires_grad = False

model.to(device)

# Training configuration
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

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

    for inputs, targets in train_loader:
        # Move inputs and targets to the appropriate device (GPU or CPU)
        inputs = [input_image.to(device) for input_image in inputs]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        
        # The model expects a list of images and a list of targets during training
        loss_dict = model(inputs, targets)
        
        # The loss_dict contains multiple losses, you sum them to get the final loss
        loss = sum(loss for loss in loss_dict.values())
        
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * len(inputs)
    
    train_loss = running_loss / len(train_loader.dataset)

    # Validation (in eval mode, no gradient calculation)
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = [input_image.to(device) for input_image in inputs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(inputs, targets)
            loss = sum(loss for loss in loss_dict.values())

            val_loss += loss.item() * len(inputs)

    val_loss /= len(val_loader.dataset)
    
    # Logging
    print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    

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

def infer_and_update_polygons(model, data_dir, confidence_threshold=0.6):  # Further lowered confidence threshold
    model.eval()
    model.to(device)

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    window_size = 64
    stride = 96  # Keeping stride to ensure sufficient coverage

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

                        print(f"Predicted Class: {predicted_class}, Confidence: {confidence}, Location: ({x}, {y})")  # Log detection details

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

            # Filter unique boxes with some overlap tolerance (slightly relaxed)
            unique_boxes = []
            for box in detected_boxes:
                if all(not (abs(box['polygon'][0]['x'] - ub['polygon'][0]['x']) < window_size * 0.75 and
                            abs(box['polygon'][0]['y'] - ub['polygon'][0]['y']) < window_size * 0.75) for ub in unique_boxes):
                    unique_boxes.append(box)

            if unique_boxes:
                updated_data = existing_data + unique_boxes
                with open(json_path, 'w') as f:
                    json.dump(updated_data, f, indent=4)
                print(f"{len(unique_boxes)} detected polygons saved in {json_path}")
            else:
                print(f"No polygons detected in {image_path}")
            
delete_detected_labels(result_dir)  #clear json file
infer_and_update_polygons(model, result_dir)