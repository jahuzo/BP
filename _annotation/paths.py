
## UNIVERSAL IMPORTS

# Imports
import torch
import torch.nn as nn
import torch.optim as optim
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


