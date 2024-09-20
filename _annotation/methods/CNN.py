# imports
import tensorflow as tf
from tensorflow.python.keras import layers, models, metrics
import json
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import os

#module imports
from paths import *

os.chdir(cur_dir)

# Load and preprocess the dataset
with open('path_to_your_json_file.json', 'r') as f:
    data = json.load(f)

# Load the image
image = Image.open('image.jpg')
image = np.array(image)

# Load the polygon labels
with open('polygon.json', 'r') as f:
    polygons = json.load(f)

# Create a mask from the polygon labels
mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
for polygon in polygons:
    points = np.array(polygon['points'], dtype=np.int32)
    cv2.fillPoly(mask, [points], 1)

# Use the mask as labels
labels = mask

# Assuming you want to use the image as a single sample
images = np.expand_dims(image, axis=0)
labels = np.expand_dims(labels, axis=0)

# Normalize the images
images = images / 255.0

# Reshape the data to fit the model
images = images.reshape((images.shape[0], 28, 28, 1))

# Split the data into training and testing sets
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)

# Build the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='sgd',
              loss='mse',
              metrics= [metrics.IoU(num_classes=2, target_class_ids=[0]), metrics.Precision(), metrics.Accuracy()])

# Train the model
history = model.fit(train_images, train_labels, epochs=5, 
                    validation_data=(test_images, test_labels))



