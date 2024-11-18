
## UNIVERSAL IMPORTS

# Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
from torchvision.ops import box_iou
from torchsummary import summary
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import json
import numpy as np
from PIL import Image, ImageDraw
import cv2
import matplotlib.pyplot as plt
import os
import random
from shapely.geometry import Polygon

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

def delete_FP_labels(base_path):
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
            
            # Remove items with the label "FP"
            filtered_polygons = [polygon for polygon in polygons if polygon.get('label') != 'FP']

            # Write the modified data back to polygons.json
            with open(json_path, 'w') as f:
                json.dump(filtered_polygons, f, indent=4)

            print(f"Processed: {json_path}")
        else:
            print(f"File not found: {json_path}")

# deletes "detected" labels, currently implemented in simple_image.py

#delete_detected_labels(result_dir)

    


