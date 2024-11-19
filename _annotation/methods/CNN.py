import sys
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold

# Paths import (update the path accordingly)
sys.path.append(r'/mnt/c/Users/jahuz/Links/BP/_annotation')

from paths import *
from train_gen import folds, test_loader, y_train

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

plot_enable = False

# Implementing the CNN model with adjustments
class CNNModel(nn.Module):
    def __init__(self, input_size=(64, 64)):
        super(CNNModel, self).__init__()
        self.input_size = input_size
        
        self.features = nn.Sequential(
            # First Convolutional Block
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.MaxPool2d(2),

            # Second Convolutional Block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.MaxPool2d(2),

            # Third Convolutional Block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.MaxPool2d(2),
        )
        self.flattened_size = self._get_flattened_size()
        self.classifier = nn.Sequential(
            nn.Linear(self.flattened_size, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  # Assuming binary classification
        )

    def _get_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros(1, 1, self.input_size[0], self.input_size[1])
            x = self.features(x)
            size = x.view(1, -1).size(1)
        return size

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# Implementing Focal Loss to handle class imbalance
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # Weighting factor for classes
        self.gamma = gamma  # Focusing parameter
        self.reduction = reduction

    def forward(self, inputs, targets):
        CE_loss = nn.functional.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-CE_loss)
        F_loss = ((1 - pt) ** self.gamma) * CE_loss
        if self.reduction == 'mean':
            return F_loss.mean()
        else:
            return F_loss.sum()

# Training Loop with Early Stopping and K-Fold Cross-Validation
num_epochs = 50  # Maximum number of epochs
patience = 5  # Early stopping patience

# Calculate class weights
class_counts = np.bincount(y_train)
total_samples = sum(class_counts)
class_weights = [total_samples / count for count in class_counts]
class_weights = torch.FloatTensor(class_weights).to(device)
print(f"Class counts: {class_counts}")
print(f"Class weights: {class_weights}")

for fold, (train_loader, val_loader) in enumerate(folds, 1):
    print(f'Fold {fold}')
    model = CNNModel(input_size=(64, 64)).to(device)
    # Use Focal Loss with class weights
    criterion = FocalLoss(alpha=class_weights, gamma=2)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
    
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    best_val_loss = float('inf')
    trigger_times = 0

    def accuracy(preds, labels):
        _, pred_classes = torch.max(preds, 1)
        return (pred_classes == labels).sum().item() / labels.size(0)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_accuracy = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            if inputs.dim() == 3:
                inputs = inputs.unsqueeze(1)  # Add channel dimension if missing
            optimizer.zero_grad()
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

        print(f"Fold {fold}, Epoch {epoch+1}/{num_epochs}, "
              f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

        scheduler.step(val_loss)

        # Early Stopping Logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trigger_times = 0
            # Save the best model for this fold
            torch.save(model.state_dict(), f'best_model_fold{fold}.pth')
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print("Early stopping!")
                break

    if plot_enable == True:
        # Plot Training & Validation Accuracy and Loss for Each Fold
        plt.figure(figsize=(12, 4))

        # Plot Accuracy
        plt.subplot(1, 2, 1)
        plt.plot(train_accuracies, label='Train Accuracy')
        plt.plot(val_accuracies, label='Validation Accuracy')
        plt.title(f'Model Accuracy - Fold {fold}')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        # Plot Loss
        plt.subplot(1, 2, 2)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title(f'Model Loss - Fold {fold}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.tight_layout()
        plt.show()

# After cross-validation, evaluate on the test set using the best model
# Here, we'll use the model from the last fold as an example
model.load_state_dict(torch.load(f'best_model_fold{fold}.pth'))
model.eval()
test_loss, test_accuracy = 0.0, 0.0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        if inputs.dim() == 3:
            inputs = inputs.unsqueeze(1)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item() * inputs.size(0)
        test_accuracy += accuracy(outputs, labels) * inputs.size(0)
test_loss /= len(test_loader.dataset)
test_accuracy /= len(test_loader.dataset)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")



def infer_and_update_polygons(model, data_dir, confidence_threshold=0.975):
    model.eval()
    model.to(device)

    window_sizes = [48, 64, 80]  # Varying window sizes
    strides = [int(ws * 0.25) for ws in window_sizes]  # 75% overlap for each window size

    # Transformation function: resize all patches to 64x64
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        # Include normalization if used during training
    ])

    for folder in os.listdir(data_dir):
        if folder not in ['009']:
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
            print(f"Processing image: {image_path}, Size: {input_image.size}")

            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    existing_data = json.load(f)
            else:
                existing_data = []

            detected_boxes = []

            total_windows = 0
            positive_predictions = 0
            class_counts = {0: 0, 1: 0}
            confidence_scores = []

            with torch.no_grad():
                for window_size, stride in zip(window_sizes, strides):
                    print(f"Processing window size: {window_size}, stride: {stride}")
                    for y in range(0, image_height - window_size + 1, stride):
                        for x in range(0, image_width - window_size + 1, stride):
                            total_windows += 1
                            patch = input_image.crop((x, y, x + window_size, y + window_size))
                            # Resize patch to 64x64
                            input_tensor = transform(patch).unsqueeze(0).to(device)

                            outputs = model(input_tensor)
                            predicted_probs = torch.softmax(outputs, dim=1)
                            predicted_class = torch.argmax(predicted_probs, dim=1).item()
                            confidence = predicted_probs[0, predicted_class].item()

                            class_counts[predicted_class] += 1
                            confidence_scores.append(confidence)

                            if predicted_class == 1 and confidence > confidence_threshold:
                                positive_predictions += 1
                                print(f"Detected Positive Class at ({x}, {y}), Window Size: {window_size}, Confidence: {confidence:.4f}")
                                detected_box = {
                                    "label": "detected",
                                    "polygon": [
                                        {"x": x, "y": y},
                                        {"x": x + window_size, "y": y},
                                        {"x": x + window_size, "y": y + window_size},
                                        {"x": x, "y": y + window_size}
                                    ],
                                    "confidence": confidence,
                                    "window_size": window_size  # Include window size for reference
                                }
                                detected_boxes.append(detected_box)
            def filter_detections(detections, min_size=20, aspect_ratio_range=(0.8, 1.2)):
                filtered = []
                for det in detections:
                    width = det['polygon'][2]['x'] - det['polygon'][0]['x']
                    height = det['polygon'][2]['y'] - det['polygon'][0]['y']
                    if width >= min_size and height >= min_size:
                        aspect_ratio = width / height
                        if aspect_ratio_range[0] <= aspect_ratio <= aspect_ratio_range[1]:
                            filtered.append(det)
                return filtered
            detected_boxes = filter_detections(detected_boxes)
            # Apply Non-Maximum Suppression (NMS)
            def nms(boxes, iou_threshold=0.2):
                if len(boxes) == 0:
                    return []

                boxes_np = np.array([[box['polygon'][0]['x'], box['polygon'][0]['y'],
                                    box['polygon'][2]['x'], box['polygon'][2]['y'], box['confidence']] for box in boxes])

                x1 = boxes_np[:, 0]
                y1 = boxes_np[:, 1]
                x2 = boxes_np[:, 2]
                y2 = boxes_np[:, 3]
                scores = boxes_np[:, 4]

                areas = (x2 - x1) * (y2 - y1)
                order = scores.argsort()[::-1]  # Sort boxes by confidence scores in descending order

                keep = []
                while order.size > 0:
                    i = order[0]
                    keep.append(i)

                    xx1 = np.maximum(x1[i], x1[order[1:]])
                    yy1 = np.maximum(y1[i], y1[order[1:]])
                    xx2 = np.minimum(x2[i], x2[order[1:]])
                    yy2 = np.minimum(y2[i], y2[order[1:]])

                    w = np.maximum(0, xx2 - xx1)
                    h = np.maximum(0, yy2 - yy1)

                    inter = w * h
                    iou = inter / (areas[i] + areas[order[1:]] - inter)

                    inds = np.where(iou <= iou_threshold)[0]
                    order = order[inds + 1]

                return [boxes[idx] for idx in keep]

            unique_boxes = nms(detected_boxes)

            if unique_boxes:
                updated_data = existing_data + unique_boxes
                with open(json_path, 'w') as f:
                    json.dump(updated_data, f, indent=4)
                print(f"{len(unique_boxes)} detected polygons saved in {json_path}")
            else:
                print(f"No polygons detected in {image_path}")

            # Print summary statistics
            print(f"Total windows processed: {total_windows}")
            print(f"Class counts: {class_counts}")
            print(f"Total positive predictions (before NMS): {positive_predictions}")
            if confidence_scores:
                avg_confidence = sum(confidence_scores) / len(confidence_scores)
                max_confidence = max(confidence_scores)
                min_confidence = min(confidence_scores)
                print(f"Average confidence score: {avg_confidence:.4f}")
                print(f"Max confidence score: {max_confidence:.4f}")
                print(f"Min confidence score: {min_confidence:.4f}")
            else:
                print("No confidence scores collected.")

# Call the inference function with the updated confidence threshold
delete_detected_labels(result_dir)
infer_and_update_polygons(model, result_dir)

from sklearn.metrics import classification_report

def evaluate_model(model, data_loader):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)[:, 1]  # Probability for class '1' (a)
            preds = (probs > 0.5).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(labels.numpy())

    print(classification_report(y_true, y_pred, digits=4))
    
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
