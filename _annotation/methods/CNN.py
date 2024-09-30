# imports
import tensorflow as tf

from keras import layers, models, metrics
import json
import numpy as np
#from sklearn.model_selection import train_test_split
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(r'C:\Users\jahuz\Links\BP\_annotation')

#module imports
from paths import *

# debugging and checks
tf.debugging.set_log_device_placement(True)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU'))) # 
tf.test.is_gpu_available()

input("Press Enter to continue...")
print("OK...moving on")

root_dir = static_path

# Custom FPR metric
class FalsePositiveRate(tf.keras.metrics.Metric):
    def __init__(self, name="false_positive_rate", **kwargs):
        super(FalsePositiveRate, self).__init__(name=name, **kwargs)
        self.false_positives = self.add_weight(name="fp", initializer="zeros")
        self.true_negatives = self.add_weight(name="tn", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Threshold the predictions to get binary values
        y_pred = tf.cast(tf.greater_equal(y_pred, 0.5), tf.float32)
        # Calculate False Positives and True Negatives
        fp = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true, 0), tf.equal(y_pred, 1)), tf.float32))
        tn = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true, 0), tf.equal(y_pred, 0)), tf.float32))
        self.false_positives.assign_add(fp)
        self.true_negatives.assign_add(tn)

    def result(self):
        return self.false_positives / (self.false_positives + self.true_negatives + tf.keras.backend.epsilon())

    def reset_states(self):
        self.false_positives.assign(0.0)
        self.true_negatives.assign(0.0)



# Define training and testing folder names
train_folders = ['001', '002', '003', '004', '005', '006', '007']
test_folders = ['008', '009']

# Function to load data from a list of folders
def load_data(train_folders, root_dir, target_size=(256, 256)):
    train_images = []
    train_masks = []

    for folder in train_folders:
        image_path = os.path.join(root_dir, folder, 'image.jpg')
        json_path = os.path.join(root_dir, folder, 'polygons.json')

        # Load the image
        image = Image.open(image_path)
        image = image.resize(target_size)  # Resize image
        image = np.array(image)

        # Load the polygons
        with open(json_path, 'r') as f:
            polygons = json.load(f)  # This should return a list

        # Create a mask for the polygons
        mask = np.zeros((target_size[0], target_size[1]), dtype=np.uint8)  # Create mask with target size
        for polygon in polygons:  # Iterate through the list
            if polygon.get('label') == 'a':
                # Extract points
                points = np.array([[point['x'], point['y']] for point in polygon['points']], dtype=np.int32)
                cv2.fillPoly(mask, [points], 1)

        train_images.append(image)
        train_masks.append(mask)

    return np.array(train_images), np.array(train_masks)

# Load training data
train_images, train_masks = load_data(train_folders, root_dir)

# Load testing data
test_images, test_masks = load_data(test_folders, root_dir)

# Normalize the images (values between 0 and 1)
train_images = train_images / 255.0
test_images = test_images / 255.0

# Resize images and masks to (32, 32) for the CNN model
train_images = tf.image.resize(train_images, (32, 32))
train_masks = tf.image.resize(train_masks[..., np.newaxis], (32, 32))

test_images = tf.image.resize(test_images, (32, 32))
test_masks = tf.image.resize(test_masks[..., np.newaxis], (32, 32))

print("Train Images Shape:", train_images.shape)  # Expected: (num_samples, 32, 32, 3)
print("Train Masks Shape:", train_masks.shape)    # Expected: (num_samples, 32, 32, 1)
print("Test Images Shape:", test_images.shape)
print("Test Masks Shape:", test_masks.shape)

# model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),  # Downsamples to (16, 16)
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),  # Downsamples to (8, 8)
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),  # Downsamples to (4, 4)

    # Upsampling back to (32, 32)
    layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', activation='relu'),  # Upsamples to (8, 8)
    layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', activation='relu'),   # Upsamples to (16, 16)
    layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same', activation='relu'),   # Upsamples to (32, 32)

    layers.Conv2D(1, (1, 1), activation='sigmoid')  # Final output: (32, 32, 1)
])

# Compile the model
model.compile(optimizer='sgd',
              loss='binary_crossentropy',
              metrics=[
                  metrics.MeanIoU(num_classes=2),  # IoU
                  metrics.Accuracy(),  # Accuracy
                  metrics.Precision(),  # Precision
                  metrics.Recall(),  # Recall (TPR)
              ])

# Train the model
history = model.fit(train_images, train_masks, epochs=10)


# Train the model
history = model.fit(train_images, train_masks, epochs=10)



test_loss, test_iou, test_accuracy, test_precision, test_tpr, test_fpr = model.evaluate(test_images, test_masks)

print(f"Test IoU: {test_iou}")
print(f"Test Accuracy: {test_accuracy}")
print(f"Test Precision: {test_precision}")
print(f"Test TPR (Recall): {test_tpr}")
print(f"Test FPR: {test_fpr}")

# Save the model
model_dir = os.path.join(static_path, 'model_CNN.h5')
os.makedirs(os.path.dirname(model_dir), exist_ok=True)
model.save(model_dir)


