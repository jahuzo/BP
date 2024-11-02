import sys

# Paths import (update the path accordingly)
sys.path.append(r'/mnt/c/Users/jahuz/Links/BP/_annotation')

from paths import *

import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageDraw  # Updated for augmentation
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

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

def augment_image(image, rotation_range=(0, 0), brightness_factor=(0.98, 1.02), contrast_factor=(0.98, 1.02)):  # Minimal augmentation
    """
    Apply minimal augmentations to the image to preserve key features.
    """
    # Randomly change brightness
    enhancer = ImageEnhance.Brightness(image)
    factor = random.uniform(*brightness_factor)  # Brightness factor for slight variation
    image = enhancer.enhance(factor)

    # Randomly change contrast
    enhancer = ImageEnhance.Contrast(image)
    factor = random.uniform(*contrast_factor)  # Contrast factor for slight variation
    image = enhancer.enhance(factor)

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

def generate_positive_negative_samples(images, labels, num_samples=5, canvas_size=(256, 256)):
    X = []
    y = []

    for i in range(num_samples):
        # Select a random image and corresponding bounding boxes
        img_index = random.randint(0, len(images) - 1)
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
                        if cropped_image.size > 0:
                            resized_image = cv2.resize(cropped_image, (64, 64), interpolation=cv2.INTER_NEAREST)
                            # Apply minimal augmentation to avoid deforming positive samples
                            augmented_image = augment_image(Image.fromarray(resized_image))
                            X.append(np.array(augmented_image))
                            y.append(1)  # Positive sample

        # Generate negative samples from regions not containing 'a'
        for _ in range(2 * len(bboxes)):  # Generate more negatives for balance
            x_min = random.randint(0, image_np.shape[1] - 64)
            y_min = random.randint(0, image_np.shape[0] - 64)
            x_max = x_min + 64
            y_max = y_min + 64

            # Ensure the generated negative sample does not overlap with any positive bounding box
            overlaps = [
                not (x_max <= bbox[0] or x_min >= bbox[2] or y_max <= bbox[1] or y_min >= bbox[3])
                for bbox in bboxes
            ]

            if not any(overlaps):
                if x_max <= image_np.shape[1] and y_max <= image_np.shape[0]:
                    negative_cropped_image = image_np[int(y_min):int(y_max), int(x_min):int(x_max)]
                    if negative_cropped_image.size > 0:
                        resized_negative_image = cv2.resize(negative_cropped_image, (64, 64), interpolation=cv2.INTER_NEAREST)
                        # Apply minimal augmentation to negative samples to enhance variety
                        augmented_negative_image = augment_image(Image.fromarray(resized_negative_image), rotation_range=(0, 0), brightness_factor=(0.95, 1.05), contrast_factor=(0.95, 1.05))
                        X.append(np.array(augmented_negative_image))
                        y.append(0)  # Negative sample

    return np.array(X), np.array(y)

# Generate training and test samples
X_train, y_train = generate_positive_negative_samples(train_images, train_labels, num_samples=len(train_images), canvas_size=(512, 512))
X_test, y_test = generate_positive_negative_samples(test_images, test_labels, num_samples=len(test_images), canvas_size=(512, 512))


# Select 5 positive and 5 negative samples
positive_samples = [X_train[i] for i in range(len(y_train)) if y_train[i] == 1][:5]
negative_samples = [X_train[i] for i in range(len(y_train)) if y_train[i] == 0][:5]

# Plot 5 positive and 5 negative samples
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
for i in range(5):
    # Plot positive samples
    axes[0, i].imshow(positive_samples[i], cmap='gray')
    axes[0, i].set_title("Positive")
    axes[0, i].axis("off")
    
    # Plot negative samples
    axes[1, i].imshow(negative_samples[i], cmap='gray')
    axes[1, i].set_title("Negative")
    axes[1, i].axis("off")

plt.suptitle("Sample Training Images - Positive and Negative")
plt.show()

# Normalize images (assuming X is image data)
X_train = X_train / 255.0
X_test = X_test / 255.0

# Convert X and y to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)  # For image data, use float32
y_train_tensor = torch.tensor(y_train, dtype=torch.long)     # For classification labels, use long
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Perform stratified split using sklearn's train_test_split
train_indices, val_indices = train_test_split(np.arange(len(y_train_tensor)), test_size=0.2, random_state=42, stratify=y_train)
train_dataset = torch.utils.data.Subset(TensorDataset(X_train_tensor, y_train_tensor), train_indices)
val_dataset = torch.utils.data.Subset(TensorDataset(X_train_tensor, y_train_tensor), val_indices)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

# Optional: create DataLoaders if you want to iterate over data in batches
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Print information about the datasets
print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of validation samples: {len(val_dataset)}")
print(f"Number of testing samples: {len(test_dataset)}")

# Getting unique labels in training and testing datasets
y_train_labels = y_train_tensor[train_indices]
y_val_labels = y_train_tensor[val_indices]
y_test_labels = y_test_tensor

unique_train_labels = torch.unique(y_train_labels)
unique_val_labels = torch.unique(y_val_labels)
unique_test_labels = torch.unique(y_test_labels)

print(f"Unique labels in training data: {unique_train_labels}")
print(f"Unique labels in validation data: {unique_val_labels}")
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