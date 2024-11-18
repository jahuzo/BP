import sys 

# Paths import (update the path accordingly)
sys.path.append(r'/mnt/c/Users/jahuz/Links/BP/_annotation')

from paths import *

import numpy as np
import cv2
from PIL import Image, ImageEnhance  # Updated for augmentation
import json
import matplotlib.pyplot as plt
import random
import os
import torch
from torch.utils.data import TensorDataset, DataLoader, Subset
from sklearn.model_selection import StratifiedKFold

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

def preprocess_polygons_to_bboxes(polygons, label_value):
    bboxes = []
    for polygon in polygons:
        if 'points' in polygon:
            try:
                bbox = calculate_bounding_box(polygon)
                bboxes.append(bbox + [label_value])  # Append label for the sample
            except ValueError as e:
                print(f"Error processing polygon: {e}")
    return bboxes

def augment_image(image, brightness_factor=(0.98, 1.02), contrast_factor=(0.98, 1.02)):
    """
    Apply minimal augmentations to the image to preserve key features.
    """
    # Randomly change brightness
    enhancer = ImageEnhance.Brightness(image)
    factor = random.uniform(*brightness_factor)
    image = enhancer.enhance(factor)

    # Randomly change contrast
    enhancer = ImageEnhance.Contrast(image)
    factor = random.uniform(*contrast_factor)
    image = enhancer.enhance(factor)

    return image

def load_data(data_dir, folders):
    images = []
    labels = []

    for folder in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder)
        if os.path.isdir(folder_path) and folder in folders:
            image_path = os.path.join(folder_path, 'image.jpg')
            json_path = os.path.join(folder_path, 'polygons.json')
            
            if os.path.isfile(image_path):
                input_image = Image.open(image_path).convert('L')  # Convert image to grayscale
                images.append(input_image)

                if os.path.isfile(json_path):
                    with open(json_path, 'r') as f:
                        existing_data = json.load(f)
                    
                    # Use polygons labeled as "a" as positive samples
                    polygons_a = [polygon for polygon in existing_data if polygon['label'] == 'a']
                    bboxes_a = preprocess_polygons_to_bboxes(polygons_a, 1)  # Label '1' for positive samples
                    
                    # Use polygons labeled as "FP" as negative samples
                    polygons_fp = [polygon for polygon in existing_data if polygon['label'] == 'FP']
                    bboxes_fp = preprocess_polygons_to_bboxes(polygons_fp, 0)  # Label '0' for negative samples
                    
                    bboxes = bboxes_a + bboxes_fp
                    
                    if bboxes:
                        labels.append(bboxes)  # Collect all bounding boxes for the image
                    else:
                        print(f"No valid polygons found in {json_path}")
                else:
                    print(f"Polygons file not found: {json_path}")

    return images, labels

# Specify folders for training
data_dir = result_dir

# Folders
train_folders = ['001', '003', '004', '005', '007']  # Training folders
test_folders = ['002', '008']  # Validation folders
true_test_folders = ['009']      # Testing folders

# Load data
train_images, train_labels = load_data(data_dir, train_folders)
test_images, test_labels = load_data(data_dir, test_folders)

def generate_positive_negative_samples(images, labels, canvas_size=(256, 256), augment_ground_truth=True):
    X = []
    y = []

    for img_index in range(len(images)):
        input_image = images[img_index]
        bboxes = labels[img_index]

        # Convert image to numpy array for OpenCV processing
        image_np = np.array(input_image)

        # Generate positive samples from bounding boxes labeled as 'a'
        for bbox in bboxes:
            x_min, y_min, x_max, y_max, label = bbox
            if label == 1:
                if x_max > x_min and y_max > y_min:  # Ensure valid bounding box
                    if x_max <= image_np.shape[1] and y_max <= image_np.shape[0]:
                        cropped_image = image_np[int(y_min):int(y_max), int(x_min):int(x_max)]
                        if cropped_image.size > 0 and np.mean(cropped_image) > 10:
                            resized_image = cv2.resize(cropped_image, (128, 128), interpolation=cv2.INTER_CUBIC)
                            X.append(np.array(resized_image))
                            y.append(1)  # Positive sample

                            # Augment positive sample
                            if augment_ground_truth:
                                for _ in range(3):  # Create 3 slightly different augmentations
                                    aug_image = augment_image(Image.fromarray(resized_image))
                                    X.append(np.array(aug_image))
                                    y.append(1)

        # Generate negative samples from regions labeled as "FP"
        for bbox in bboxes:
            x_min, y_min, x_max, y_max, label = bbox
            if label == 0:
                if x_max > x_min and y_max > y_min:  # Ensure valid bounding box
                    if x_max <= image_np.shape[1] and y_max <= image_np.shape[0]:
                        cropped_image = image_np[int(y_min):int(y_max), int(x_min):int(x_max)]
                        if cropped_image.size > 0 and np.mean(cropped_image) > 10:
                            resized_image = cv2.resize(cropped_image, (128, 128), interpolation=cv2.INTER_CUBIC)
                            augmented_image = augment_image(Image.fromarray(resized_image))
                            X.append(np.array(augmented_image))
                            y.append(0)  # Negative sample

        # Generate additional negative samples from random locations
        num_negatives = 100  # Adjust as needed
        for _ in range(num_negatives):
            rand_x_min = random.randint(0, image_np.shape[1] - 128)
            rand_y_min = random.randint(0, image_np.shape[0] - 128)
            rand_x_max = rand_x_min + 128
            rand_y_max = rand_y_min + 128

            if rand_x_max <= image_np.shape[1] and rand_y_max <= image_np.shape[0]:
                cropped_image = image_np[rand_y_min:rand_y_max, rand_x_min:rand_x_max]
                if cropped_image.size > 0 and np.mean(cropped_image) > 10:
                    resized_image = cv2.resize(cropped_image, (128, 128), interpolation=cv2.INTER_CUBIC)
                    augmented_image = augment_image(Image.fromarray(resized_image))
                    X.append(np.array(augmented_image))
                    y.append(0)  # Negative sample

    return np.array(X), np.array(y)

# Generate training samples
X_train, y_train = generate_positive_negative_samples(train_images, train_labels, canvas_size=(512, 512))

# Normalize images
X_train = X_train / 255.0

# Convert X and y to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)  # For image data, use float32
y_train_tensor = torch.tensor(y_train, dtype=torch.long)     # For classification labels, use long

# Ensure that X_train_tensor and y_train_tensor are not empty
if len(X_train_tensor) == 0 or len(y_train_tensor) == 0:
    print("No training data generated. Please check the data loading and sample generation steps.")
    sys.exit(1)

# Set up Stratified K-Fold Cross-Validation
k = 5  # Number of folds
skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

folds = []

for fold, (train_indices, val_indices) in enumerate(skf.split(X_train_tensor, y_train_tensor), 1):
    print(f"Fold {fold}")

    train_dataset = Subset(TensorDataset(X_train_tensor, y_train_tensor), train_indices)
    val_dataset = Subset(TensorDataset(X_train_tensor, y_train_tensor), val_indices)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    folds.append((train_loader, val_loader))

# Generate test samples
X_test, y_test = generate_positive_negative_samples(test_images, test_labels, canvas_size=(512, 512))

# Normalize test images
X_test = X_test / 255.0

# Convert to tensors
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Create test dataset and loader
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Now folds and test_loader can be imported into the main script

# Optionally, print dataset information
print(f"Number of folds: {k}")
for i, (train_loader, val_loader) in enumerate(folds, 1):
    print(f"Fold {i}:")
    print(f"  Number of training samples: {len(train_loader.dataset)}")
    print(f"  Number of validation samples: {len(val_loader.dataset)}")

print(f"Number of testing samples: {len(test_dataset)}")
