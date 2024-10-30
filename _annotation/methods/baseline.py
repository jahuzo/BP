import sys

# Paths import (update the path accordingly)
sys.path.append(r'/mnt/c/Users/jahuz/Links/BP/_annotation')

from paths import *

import random
from sklearn.model_selection import train_test_split  # Added for stratified splitting

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

import numpy as np
import cv2
from PIL import Image, ImageEnhance  # Updated for augmentation
import json
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Added a learning rate scheduler # NEW!
from torch.optim.lr_scheduler import StepLR

data_dir = result_dir

def calculate_bounding_box(polygon):
    points = polygon['points']
    if len(points) < 3:
        raise ValueError("Polygon does not have enough points to form a bounding box.")
    
    x_coords = [point['x'] for point in points]
    y_coords = [point['y'] for point in points]
    
    # Calculate bounding box
    xmin = min(x_coords)
    xmax = max(x_coords)
    ymin = min(y_coords)
    ymax = max(y_coords)
    
    return [xmin, ymin, xmax, ymax]

def preprocess_polygons_to_bboxes(polygons):
    bboxes = []
    for polygon in polygons:
        if 'points' in polygon:
            try:
                bbox = calculate_bounding_box(polygon)
                bboxes.append(bbox + [1])  # Append label '1' for positive samples
            except ValueError as e:
                print(f"Error processing polygon: {e}")
    return bboxes

def convert_bboxes_to_integers(bboxes):
    converted_bboxes = []
    for bbox in bboxes:
        # Ensure each coordinate in bbox is converted to an integer
        int_bbox = [int(coord) for coord in bbox[:4]] + [bbox[4]]
        converted_bboxes.append(int_bbox)
    return converted_bboxes

def augment_image(image):  # NEW!
    """
    Apply random augmentations to the image to improve model generalization.
    """
    # Randomly change brightness
    enhancer = ImageEnhance.Brightness(image)
    factor = random.uniform(0.6, 1.4)  # Brightness factor between 0.6 and 1.4 for more variation
    image = enhancer.enhance(factor)

    # Randomly change contrast # NEW!
    enhancer = ImageEnhance.Contrast(image)
    factor = random.uniform(0.8, 1.2)  # Contrast factor between 0.8 and 1.2
    image = enhancer.enhance(factor)

    # Randomly flip the image horizontally
    if random.random() > 0.5:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
    
    # Randomly rotate the image slightly
    angle = random.uniform(-15, 15)  # Rotate between -15 and 15 degrees for more variation
    image = image.rotate(angle)

    return image

def load_data(data_dir, train_folders, test_folders):
    train_images = []
    train_labels = []
    test_images = []
    test_labels = []

    for folder in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder)
        if os.path.isdir(folder_path):
            image_path = os.path.join(folder_path, 'image.jpg')
            json_path = os.path.join(folder_path, 'polygons.json')
            
            if os.path.isfile(image_path):
                input_image = Image.open(image_path)

                if folder in train_folders:
                    train_images.append(input_image)
                elif folder in test_folders:
                    test_images.append(input_image)

                if os.path.isfile(json_path):
                    with open(json_path, 'r') as f:
                        existing_data = json.load(f)
                    
                    # Only use polygons labeled as "a"
                    polygons = [polygon for polygon in existing_data if polygon['label'] == 'a']
                    if polygons:
                        bboxes = preprocess_polygons_to_bboxes(polygons)
                        if bboxes:
                            if folder in train_folders:
                                train_labels.append(bboxes)  # Collect all bounding boxes for the image
                            elif folder in test_folders:
                                test_labels.append(bboxes)
                    else:
                        print(f"No 'a' polygons found in {json_path}")
                else:
                    print(f"Polygons file not found: {json_path}")

    return train_images, train_labels, test_images, test_labels

# Specify folders for training and testing
data_dir = result_dir
train_folders = ['001', '002', '003', '004', '005', '006', '007']  # Training folders
test_folders = ['008', '009']  # Testing folders

# Load data
train_images, train_labels, test_images, test_labels = load_data(data_dir, train_folders, test_folders)

def generate_training_samples(images, labels, num_samples=5, canvas_size=(256, 256)):
    samples = []
    annotations = []

    for i in range(num_samples):
        # Select a random image and corresponding bounding boxes
        img_index = random.randint(0, len(images) - 1)
        input_image = images[img_index]

        # Apply augmentation to the image # NEW!
        input_image = augment_image(input_image)

        # Resize image while maintaining aspect ratio, then pad to fit canvas size
        input_image.thumbnail(canvas_size, Image.LANCZOS)  # Resize maintaining aspect ratio
        background = Image.new('RGB', canvas_size, (255, 255, 255))  # Create a white canvas
        left = (canvas_size[0] - input_image.size[0]) // 2
        top = (canvas_size[1] - input_image.size[1]) // 2
        background.paste(input_image, (left, top))

        # Update bounding boxes to match the resized image position on the canvas
        scale_x = input_image.size[0] / images[img_index].size[0]
        scale_y = input_image.size[1] / images[img_index].size[1]
        bboxes = labels[img_index]
        bboxes = convert_bboxes_to_integers(bboxes)
        new_bboxes = []
        for bbox in bboxes:
            xmin, ymin, xmax, ymax, label = bbox
            xmin = int(left + xmin * scale_x)
            xmax = int(left + xmax * scale_x)
            ymin = int(top + ymin * scale_y)
            ymax = int(top + ymax * scale_y)
            new_bboxes.append([xmin, ymin, xmax, ymax, label])

        # Convert image to numpy array
        image = np.array(background)

        # Draw the bounding boxes
        for bbox in new_bboxes:
            xmin, ymin, xmax, ymax, label = bbox
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), thickness=3)

        samples.append(image)
        annotations.append(new_bboxes)

    return samples, annotations

# Downsample negative samples
def downsample_negatives(X, y):
    negative_indices = [i for i, label in enumerate(y) if label == 0]
    positive_indices = [i for i, label in enumerate(y) if label == 1]
    if len(positive_indices) < len(negative_indices):
        negative_indices = random.sample(negative_indices, len(positive_indices))
    balanced_indices = positive_indices + negative_indices
    random.shuffle(balanced_indices)
    X_balanced = [X[i] for i in balanced_indices]
    y_balanced = [y[i] for i in balanced_indices]
    return np.array(X_balanced), np.array(y_balanced)

# Generate training and test samples
train_samples, train_annotations = generate_training_samples(train_images, train_labels, num_samples=len(train_images), canvas_size=(512, 512))
test_samples, test_annotations = generate_training_samples(test_images, test_labels, num_samples=len(test_images), canvas_size=(512, 512))

X = []
y = []
image_size = (64, 64)

for image, bboxes in zip(train_samples, train_annotations):
    for bbox in bboxes:
        x_min, y_min, x_max, y_max, label = bbox
        if label == 3:  # Skip blank images
            continue
        # Crop and resize the image to the target size
        if x_max > x_min and y_max > y_min:  # Ensure valid bounding box
            cropped_image = image[y_min:y_max, x_min:x_max]
            if cropped_image.size > 0:  # Check if cropped image is not empty
                resized_image = cv2.resize(cropped_image, (64, 64))
                X.append(resized_image)
                y.append(label)

        # Generate random negative samples near the original bounding box
        for _ in range(2):  # Increased negative samples per bounding box to 2 for more negatives
            shift_x = random.randint(-(64 // 2), 64 // 2)  # Corrected shift range
            shift_y = random.randint(-(64 // 2), 64 // 2)  # Corrected shift range
            new_x_min = max(0, x_min + shift_x)
            new_y_min = max(0, y_min + shift_y)
            new_x_max = min(image.shape[1], new_x_min + (x_max - x_min))
            new_y_max = min(image.shape[0], new_y_min + (y_max - y_min))

            # Calculate overlap with all existing bounding boxes
            overlaps = []
            for other_bbox in bboxes:
                other_x_min, other_y_min, other_x_max, other_y_max, _ = other_bbox
                intersection_x_min = max(other_x_min, new_x_min)
                intersection_y_min = max(other_y_min, new_y_min)
                intersection_x_max = min(other_x_max, new_x_max)
                intersection_y_max = min(other_y_max, new_y_max)

                intersection_area = max(0, intersection_x_max - intersection_x_min) * max(0, intersection_y_max - intersection_y_min)
                other_area = (other_x_max - other_x_min) * (other_y_max - other_y_min)

                overlap = intersection_area / other_area if other_area > 0 else 0
                overlaps.append(overlap)

            # Only add negative sample if it doesn't overlap more than 50% with any existing bounding box
            if all(overlap <= 0.5 for overlap in overlaps):
                negative_cropped_image = image[new_y_min:new_y_max, new_x_min:new_x_max]
                if negative_cropped_image.size > 0:  # Check if cropped image is not empty
                    resized_negative_image = cv2.resize(negative_cropped_image, (64, 64))
                    X.append(resized_negative_image)
                    y.append(0)

# Append random negative samples from the image that do not overlap with any bounding boxes
num_random_samples = 100  # Increased number of purely random negative samples to 100
for _ in range(num_random_samples):
    new_x_min = random.randint(0, image.shape[1] - image_size[0])
    new_y_min = random.randint(0, image.shape[0] - image_size[1])
    new_x_max = new_x_min + image_size[0]
    new_y_max = new_y_min + image_size[1]

    # Calculate overlap with all existing bounding boxes
    overlaps = []
    for bbox in bboxes:
        other_x_min, other_y_min, other_x_max, other_y_max, _ = bbox
        intersection_x_min = max(other_x_min, new_x_min)
        intersection_y_min = max(other_y_min, new_y_min)
        intersection_x_max = min(other_x_max, new_x_max)
        intersection_y_max = min(other_y_max, new_y_max)

        intersection_area = max(0, intersection_x_max - intersection_x_min) * max(0, intersection_y_max - intersection_y_min)
        other_area = (other_x_max - other_x_min) * (other_y_max - other_y_min)

        overlap = intersection_area / other_area if other_area > 0 else 0
        overlaps.append(overlap)

    # Only add negative sample if it doesn't overlap with any existing bounding box
    if all(overlap == 0 for overlap in overlaps):
        random_cropped_image = image[new_y_min:new_y_max, new_x_min:new_x_max]
        if random_cropped_image.size > 0:  # Check if cropped image is not empty
            resized_random_image = cv2.resize(random_cropped_image, (64, 64))
            X.append(resized_random_image)
            y.append(0)

# Oversample positive examples to ensure better balance
positive_indices = [i for i, label in enumerate(y) if label == 1]
num_positives = len(positive_indices)
num_negatives = len([i for i in y if i == 0])

if num_positives < num_negatives:
    oversample_factor = num_negatives // num_positives
    X_positives = [X[i] for i in positive_indices]
    y_positives = [y[i] for i in positive_indices]
    X.extend(X_positives * oversample_factor)
    y.extend(y_positives * oversample_factor)

X = np.array(X)
y = np.array(y)

# Normalize images (assuming X is image data)
X = X / 255.0

# Convert X and y to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)  # For image data, use float32
y_tensor = torch.tensor(y, dtype=torch.long)     # For classification labels, use long

# Perform stratified split using sklearn's train_test_split
train_indices, test_indices = train_test_split(np.arange(len(y_tensor)), test_size=0.2, random_state=42, stratify=y)
train_dataset = torch.utils.data.Subset(TensorDataset(X_tensor, y_tensor), train_indices)
test_dataset = torch.utils.data.Subset(TensorDataset(X_tensor, y_tensor), test_indices)

# Optional: create DataLoaders if you want to iterate over data in batches
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Print information about the datasets
print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of testing samples: {len(test_dataset)}")
print(f"Number of training labels: {len(train_dataset)}")
print(f"Number of testing labels: {len(test_dataset)}")

# Getting unique labels in training and testing datasets
y_train_labels = y_tensor[train_indices]
y_test_labels = y_tensor[test_indices]
unique_train_labels = torch.unique(y_train_labels)
unique_test_labels = torch.unique(y_test_labels)

print(f"Unique labels in training data: {unique_train_labels}")
print(f"Unique labels in testing data: {unique_test_labels}")

# Plot histograms
plt.figure(figsize=(12, 5))

# Training set histogram
plt.subplot(1, 2, 1)
plt.hist(y_train_labels.numpy(), bins=np.arange(-0.5, len(torch.unique(y_train_labels))), rwidth=0.8, color='b', alpha=0.7)
plt.xlabel('Class Label')
plt.ylabel('Frequency')
plt.title('Histogram of Training Classes')

# Testing set histogram
plt.subplot(1, 2, 2)
plt.hist(y_test_labels.numpy(), bins=np.arange(-0.5, len(torch.unique(y_test_labels))), rwidth=0.8, color='g', alpha=0.7)
plt.xlabel('Class Label')
plt.ylabel('Frequency')
plt.title('Histogram of Testing Classes')

plt.tight_layout()
plt.show()

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
        for inputs, labels in test_loader:
            outputs = model(inputs.permute(0, 3, 1, 2))
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            val_accuracy += accuracy(outputs, labels) * inputs.size(0)
    val_loss /= len(test_loader.dataset)
    val_accuracy /= len(test_loader.dataset)

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

summary(model, (3, 64, 64))

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
                

infer_and_update_polygons(model, result_dir)
