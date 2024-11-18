import sys

# Paths import (update the path accordingly)
sys.path.append(r'/mnt/c/Users/jahuz/Links/BP/_annotation')

from paths import *
from train_gen import folds, test_loader

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Simplified CNN Model with Regularization
class CNNModel(nn.Module):
    def __init__(self, input_size=(128, 128)):
        super(CNNModel, self).__init__()
        
        # Reduced number of layers and filters
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        
        # Flattening size calculation
        example_input = torch.zeros(1, 1, input_size[0], input_size[1])
        self.flattened_size = self._get_flattened_size(example_input)
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.flattened_size, 64)
        self.fc2 = nn.Linear(64, 2)  # 2 output classes
        
        # Add Dropout Layer
        self.dropout = nn.Dropout(p=0.5)

    def _get_flattened_size(self, x):
        with torch.no_grad():
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            return x.view(1, -1).size(1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(x)
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x  # Return raw logits

# Training Loop with Early Stopping and K-Fold Cross-Validation
num_epochs = 50  # Maximum number of epochs
patience = 5  # Early stopping patience

for fold, (train_loader, val_loader) in enumerate(folds, 1):
    print(f'Fold {fold}')
    model = CNNModel(input_size=(128, 128)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)  # With weight decay
    
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


def infer_and_update_polygons(model, data_dir, confidence_threshold=0.5): 
    model.eval()
    model.to(device)
    w_size = 128
    transform = transforms.Compose([
        transforms.Resize((w_size, w_size)),
        transforms.ToTensor(),
        # Note: ToTensor() scales pixel values to [0,1]
    ])

    window_size = w_size
    stride = int(window_size / 2)  # Ensure stride is an integer

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

            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    existing_data = json.load(f)
            else:
                existing_data = []

            detected_boxes = []
            draw_image = input_image.copy()  # Create a copy for drawing
            draw = ImageDraw.Draw(draw_image)

            with torch.no_grad():
                total_windows = 0
                positive_predictions = 0
                for y in range(0, image_height - window_size + 1, stride):
                    for x in range(0, image_width - window_size + 1, stride):
                        total_windows += 1
                        patch = input_image.crop((x, y, x + window_size, y + window_size))
                        input_tensor = transform(patch).unsqueeze(0).to(device)

                        outputs = model(input_tensor)
                        predicted_probs = torch.softmax(outputs, dim=1)
                        predicted_class = torch.argmax(predicted_probs, dim=1).item()
                        confidence = predicted_probs[0, predicted_class].item()

                        print(f"Window at ({x}, {y}): Predicted Class = {predicted_class}, Confidence = {confidence:.4f}")

                    if predicted_class == 1 and confidence > confidence_threshold:
                            positive_predictions += 1
                            print(f"Predicted Class: {predicted_class}, Confidence: {confidence:.4f}, Location: ({x}, {y})")
                            detected_box = {
                                "label": "detected",
                                "polygon": [
                                    {"x": x, "y": y},
                                    {"x": x + window_size, "y": y},
                                    {"x": x + window_size, "y": y + window_size},
                                    {"x": x, "y": y + window_size}
                                ],
                                "confidence": confidence  # Include confidence in the detected box
                            }
                            detected_boxes.append(detected_box)
                            # Draw preliminary detections (optional)
                            draw.rectangle([x, y, x + window_size, y + window_size], outline="red", width=1)

            # Apply Non-Maximum Suppression (NMS)
            def nms(boxes, overlap_thresh=0.5):
                if len(boxes) == 0:
                    return []
                
                boxes_np = np.array([[box['polygon'][0]['x'], box['polygon'][0]['y'], 
                                      box['polygon'][2]['x'], box['polygon'][2]['y'], box['confidence']] for box in boxes])
                
                x1 = boxes_np[:, 0]
                y1 = boxes_np[:, 1]
                x2 = boxes_np[:, 2]
                y2 = boxes_np[:, 3]
                scores = boxes_np[:, 4]
                
                areas = (x2 - x1 + 1) * (y2 - y1 + 1)
                order = scores.argsort()[::-1]  # Sort boxes by confidence scores in descending order
                
                keep = []
                while order.size > 0:
                    i = order[0]
                    keep.append(i)
                    
                    xx1 = np.maximum(x1[i], x1[order[1:]])
                    yy1 = np.maximum(y1[i], y1[order[1:]])
                    xx2 = np.minimum(x2[i], x2[order[1:]])
                    yy2 = np.minimum(y2[i], y2[order[1:]])
                    
                    w = np.maximum(0.0, xx2 - xx1 + 1)
                    h = np.maximum(0.0, yy2 - yy1 + 1)
                    inter = w * h
                    
                    union = areas[i] + areas[order[1:]] - inter
                    iou = inter / union
                    
                    inds = np.where(iou <= overlap_thresh)[0]
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
