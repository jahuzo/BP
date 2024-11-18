import sys
import os
import json
import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageFilter
import matplotlib.pyplot as plt
import random
import torch
from torch.utils.data import TensorDataset, DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from torchvision import transforms

# Paths import (update the path accordingly)
sys.path.append(r'/mnt/c/Users/jahuz/Links/BP/_annotation')

from paths import *

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

                if os.path.isfile(json_path):
                    with open(json_path, 'r') as f:
                        existing_data = json.load(f)
                    
                    # Use polygons labeled as "a" as positive samples
                    polygons_a = [polygon for polygon in existing_data if polygon['label'] == 'a']
                    bboxes_a = preprocess_polygons_to_bboxes(polygons_a, 1)  # Label '1' for positive samples
                    
                    # Use polygons labeled as "FP" as negative samples (optional)
                    # Since you prefer to include them, we can retain them
                    polygons_fp = [polygon for polygon in existing_data if polygon['label'] == 'FP']
                    bboxes_fp = preprocess_polygons_to_bboxes(polygons_fp, 0)  # Label '0' for negative samples
                    
                    bboxes = bboxes_a + bboxes_fp
                    
                    if bboxes:
                        images.append((input_image, bboxes))  # Store image and bounding boxes
                    else:
                        print(f"No valid polygons found in {json_path}")
                else:
                    print(f"Polygons file not found: {json_path}")
            else:
                print(f"Image file not found: {image_path}")

    return images

# Data augmentation transforms
data_augmentation_transforms = transforms.Compose([
    transforms.RandomRotation(degrees=15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomApply([transforms.RandomPerspective(distortion_scale=0.2)], p=0.5),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.3),
    transforms.ToTensor(),
])

# Specify folders for training
data_dir = result_dir

# Folders
train_folders = ['001', '003', '004', '005', '007']  # Training folders
test_folders = ['002', '008']  # Validation folders

# Load data
train_data = load_data(data_dir, train_folders)
test_data = load_data(data_dir, test_folders)

def calculate_iou(boxA, boxB):
    # Compute the intersection over union of two boxes
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute the area of intersection rectangle
    interWidth = max(0, xB - xA)
    interHeight = max(0, yB - yA)
    interArea = interWidth * interHeight

    # Compute the area of both the prediction and ground-truth rectangles
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # Compute the intersection over union
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def generate_samples(data, augment=True, num_random_negatives=200, iou_threshold=0.3):
    X = []
    y = []

    for (input_image, bboxes) in data:
        image_np = np.array(input_image)

        # Separate positive and negative bboxes
        positive_bboxes = [bbox for bbox in bboxes if bbox[4] == 1]
        negative_bboxes = [bbox for bbox in bboxes if bbox[4] == 0]

        # Generate positive samples
        for bbox in positive_bboxes:
            x_min, y_min, x_max, y_max, label = bbox
            if x_max > x_min and y_max > y_min:
                if x_max <= image_np.shape[1] and y_max <= image_np.shape[0]:
                    cropped_image = input_image.crop((x_min, y_min, x_max, y_max))
                    resized_image = cropped_image.resize((128, 128), Image.BICUBIC)
                    X.append(resized_image)
                    y.append(1)

                    # Data augmentation for positive samples
                    if augment:
                        for _ in range(5):
                            augmented_image = data_augmentation_transforms(resized_image)
                            X.append(transforms.ToPILImage()(augmented_image))
                            y.append(1)

        # Optionally include "FP" samples as hard negatives
        for bbox in negative_bboxes:
            x_min, y_min, x_max, y_max, label = bbox
            if x_max > x_min and y_max > y_min:
                if x_max <= image_np.shape[1] and y_max <= image_np.shape[0]:
                    cropped_image = input_image.crop((x_min, y_min, x_max, y_max))
                    resized_image = cropped_image.resize((128, 128), Image.BICUBIC)
                    X.append(resized_image)
                    y.append(0)

                    # Data augmentation for negative samples
                    if augment:
                        for _ in range(3):
                            augmented_image = data_augmentation_transforms(resized_image)
                            X.append(transforms.ToPILImage()(augmented_image))
                            y.append(0)

        # Generate additional random negative samples with allowable overlap
        for _ in range(num_random_negatives):
            rand_x_min = random.randint(0, image_np.shape[1] - 128)
            rand_y_min = random.randint(0, image_np.shape[0] - 128)
            rand_x_max = rand_x_min + 128
            rand_y_max = rand_y_min + 128

            if rand_x_max <= image_np.shape[1] and rand_y_max <= image_np.shape[0]:
                # Calculate IoU with all positive bboxes
                max_iou = 0
                for bbox in positive_bboxes:
                    x_min, y_min, x_max, y_max, label = bbox
                    boxA = [rand_x_min, rand_y_min, rand_x_max, rand_y_max]
                    boxB = [x_min, y_min, x_max, y_max]
                    iou = calculate_iou(boxA, boxB)
                    if iou > max_iou:
                        max_iou = iou

                # Include negative sample if max IoU is below threshold
                if max_iou < iou_threshold:
                    cropped_image = input_image.crop((rand_x_min, rand_y_min, rand_x_max, rand_y_max))
                    if cropped_image.size == (128, 128):
                        X.append(cropped_image)
                        y.append(0)

                        # Data augmentation for negative samples
                        if augment:
                            for _ in range(2):
                                augmented_image = data_augmentation_transforms(cropped_image)
                                X.append(transforms.ToPILImage()(augmented_image))
                                y.append(0)

    return X, y

# Generate training samples
X_train, y_train = generate_samples(train_data, augment=True)

# Plot 10 positive and 10 negative samples
def plot_samples(X, y, num_samples=10):
    # Separate positive and negative samples
    positive_samples = [img for img, label in zip(X, y) if label == 1]
    negative_samples = [img for img, label in zip(X, y) if label == 0]

    # Shuffle samples to get a random selection
    random.shuffle(positive_samples)
    random.shuffle(negative_samples)

    # Select up to num_samples from each
    positive_samples = positive_samples[:num_samples]
    negative_samples = negative_samples[:num_samples]

    # Plot positive samples
    plt.figure(figsize=(15, 3))
    for i, img in enumerate(positive_samples):
        plt.subplot(2, num_samples, i + 1)
        plt.imshow(img, cmap='gray')
        plt.title("Positive")
        plt.axis('off')

    # Plot negative samples
    for i, img in enumerate(negative_samples):
        plt.subplot(2, num_samples, num_samples + i + 1)
        plt.imshow(img, cmap='gray')
        plt.title("Negative")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

# Call the plot_samples function
plot_samples(X_train, y_train, num_samples=10)

# Convert images to tensors and normalize
X_train_tensor = []
for img in X_train:
    tensor_img = transforms.ToTensor()(img)
    X_train_tensor.append(tensor_img)
X_train_tensor = torch.stack(X_train_tensor)

y_train_tensor = torch.tensor(y_train, dtype=torch.long)

# Set up Stratified K-Fold Cross-Validation
k = 5  # Number of folds
skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

folds = []

for fold, (train_indices, val_indices) in enumerate(skf.split(X_train_tensor, y_train_tensor), 1):
    print(f"Fold {fold}")

    train_dataset = Subset(TensorDataset(X_train_tensor, y_train_tensor), train_indices)
    val_dataset = Subset(TensorDataset(X_train_tensor, y_train_tensor), val_indices)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    folds.append((train_loader, val_loader))

# Generate test samples
X_test, y_test = generate_samples(test_data, augment=False, num_random_negatives=50, iou_threshold=0.3)

# Convert test images to tensors and normalize
X_test_tensor = []
for img in X_test:
    tensor_img = transforms.ToTensor()(img)
    X_test_tensor.append(tensor_img)
X_test_tensor = torch.stack(X_test_tensor)

y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Create test dataset and loader
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Now folds and test_loader can be imported into the main script

# Optionally, print dataset information
print(f"Number of folds: {k}")
for i, (train_loader, val_loader) in enumerate(folds, 1):
    print(f"Fold {i}:")
    print(f"  Number of training samples: {len(train_loader.dataset)}")
    print(f"  Number of validation samples: {len(val_loader.dataset)}")

print(f"Number of testing samples: {len(test_dataset)}")

# Export y_train for use in the main script
y_train = y_train_tensor.numpy()
