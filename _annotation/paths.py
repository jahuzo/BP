
## UNIVERSAL IMPORTS

# Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import os


def find_directory(target_static):
    """
    Recursively searches for the target directory starting from the current working directory.
    :param target_dir: The name of the target directory (e.g., '001')
    :return: Full path to the target directory, if found.
    """
    # Start searching from the current working directory
    cwd = os.getcwd()
    print(cwd)
    for dirpath, dirnames, filenames in os.walk(cwd):
        if 'static' in dirnames:
            static_path = os.path.join(dirpath, target_static)
            
            return static_path
            
            
            ## This part is for searching at one subdir level deeper, currently deprecated
            

            #target_path = os.path.join(static_path, target_static)
            #if os.path.isdir(target_path):
            #   return target_path  # Return the full path if found
            
    return None

# Specify the target directory name
target_static = "static"
target_00x = "001" #deprecated

# Search for the directory
result_dir = find_directory(target_static)

if result_dir:
    print(f"Directory found: {result_dir}")
else:
    print(f"Directory '{target_static}' not found")



def load_image(image_path):
    """Load an image from the given path."""
    return Image.open(image_path)

def load_polygons(json_path):
    """Load polygons from a JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

import os
import json

def delete_detected_labels(base_path):
    # Loop through folder names from 001 to 009
    for folder_num in range(1, 10):
        folder_name = f"{folder_num:03d}"  # Format as 001, 002, ..., 009
        folder_path = os.path.join(base_path, folder_name)
        json_path = os.path.join(folder_path, 'polygons.json')

        # Check if the polygons.json file exists
        if os.path.exists(json_path):
            # Load the JSON data
            with open(json_path, 'r') as f:
                polygons = json.load(f)
            
            # Remove items with the label "detected"
            filtered_polygons = [polygon for polygon in polygons if polygon.get('label') != 'detected']

            # Write the modified data back to polygons.json
            with open(json_path, 'w') as f:
                json.dump(filtered_polygons, f, indent=4)

            print(f"Processed: {json_path}")
        else:
            print(f"File not found: {json_path}")

# deletes "detected" labels

#delete_detected_labels(result_dir)



def calculate_accuracy(detected_polygons, ground_truth_polygons):
    """Calculate accuracy by comparing detected polygons to ground truth."""
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    # Count true positives and false positives
    for detected in detected_polygons:
        if detected in ground_truth_polygons:
            true_positives += 1
        else:
            false_positives += 1
            
    # Count false negatives
    for truth in ground_truth_polygons:
        if truth not in detected_polygons:
            false_negatives += 1
            
    # Calculate accuracy
    accuracy = (
        true_positives / (true_positives + false_positives + false_negatives)
        if (true_positives + false_positives + false_negatives) > 0 else 0
    )
    
    return accuracy

def universal_accuracy_function(model, main_dir):
    """Universal function to calculate average accuracy of detected polygons in multiple folders."""
    accuracies = []
    
    # Iterate over each folder (assuming they are named '001', '002', ..., '007')
    for folder in sorted(os.listdir(main_dir)):
        folder_path = os.path.join(main_dir, folder)
        
        if os.path.isdir(folder_path):
            image_path = os.path.join(folder_path, 'image.jpg')
            json_path = os.path.join(folder_path, 'polygons.json')
            
            # Check if image and JSON exist
            if os.path.exists(image_path) and os.path.exists(json_path):
                # Load image and polygons
                load_image(image_path)  # Image loading can be omitted if not needed
                polygons = load_polygons(json_path)

                # Extract polygons based on labels
                detected_polygons = [p for p in polygons if p['label'] == 'detected']
                ground_truth_polygons = [p for p in polygons if p['label'] == 'a']
                
                # Calculate accuracy for the current folder
                accuracy = calculate_accuracy(detected_polygons, ground_truth_polygons)
                accuracies.append(accuracy)
            else:
                print(f"Warning: Missing image or polygons.json in {folder_path}")
    
    # Calculate the average accuracy across all folders
    average_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0
    
    return {
        'average_accuracy': average_accuracy,
        'folder_count': len(accuracies)
    }
    


