# Author: Marek Vacula
# Bachelor thesis  

# package imports
from operator import index
from flask import redirect, url_for
from flask import Flask, render_template
from flask import request, jsonify
from flask import Response
#from prettytable import PrettyTable
import logging
import json
import cv2
import numpy as np
from PIL import Image, ImageDraw
import os

logging.basicConfig(level=logging.DEBUG)

# script imports
from paths import cur_dir, dir_path
from methods.baseline import *

# simple function to get the current working directory 
def jump_to_script_directory():
    # Get the directory of the current script
    script_directory = os.path.dirname(os.path.abspath(__file__))
    
    try:
        # Change the current working directory to the script's directory
        os.chdir(script_directory)
        print(f"Successfully changed the directory to the script's directory: {os.getcwd()}")
    except Exception as e:
        print(f"An error occurred while changing the directory: {e}")

jump_to_script_directory()

# detection migrated to baseline.py

def print_structure(var, indent=0):
    """
    Recursively prints the structure of the given Python variable.
    """
    prefix = " " * indent  # Indentation
    if isinstance(var, dict):
        print(f"{prefix}{type(var)} containing:")
        for key, value in var.items():
            print(f"{prefix}  Key: {key} ->", end=" ")
            print_structure(value, indent + 4)
    elif isinstance(var, list):
        print(f"{prefix}{type(var)} with {len(var)} elements:")
        for i, item in enumerate(var):
            print(f"{prefix}  Index {i} ->", end=" ")
            print_structure(item, indent + 4)
    elif isinstance(var, tuple):
        print(f"{prefix}{type(var)} with {len(var)} elements:")
        for i, item in enumerate(var):
            print(f"{prefix}  Element {i} ->", end=" ")
            print_structure(item, indent + 4)
    elif isinstance(var, set):
        print(f"{prefix}{type(var)} with {len(var)} elements:")
        for item in var:
            print(f"{prefix}  Value ->", end=" ")
            print_structure(item, indent + 4)
    else:
        print(f"{prefix}{type(var)} with value {var}")

os.chdir(cur_dir)

def load_polygons(dir_path):
    json_path = os.path.join(app.static_folder, dir_path, 'polygons.json')
    
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            polygons = json.load(f)
    else:
        polygons = []  # Return an empty list if the file does not exist

    return polygons


app = Flask(__name__, static_folder='static', static_url_path= '/static'  , template_folder='templates')
@app.route('/')
def index():

    polygons = load_polygons(dir_path)
    return render_template('index.html', polygons=polygons)

@app.route('/predict')
def predict():
    os.chdir(cur_dir)
    polygons = load_polygons(dir_path)    

    # Open the source image
    source_img_path = os.path.join(app.static_folder, dir_path, 'image.jpg')
    source_img = Image.open(source_img_path)

    baseline(polygons, source_img)
    
    
@app.route('/submit-polygons', methods=['POST'])
def submit_polygons():
      # Parse the JSON data
      polygons = request.get_json()
      
      # Filter out polygons with fewer than 3 points
      polygons = [polygon for polygon in polygons if len(polygon["points"]) >= 3]
      
      os.chdir(cur_dir)
      
      with open('polygons.json', 'w') as f:
          json.dump(polygons, f, indent=4)
  
      #return redirect('/')  
      return redirect(url_for('index'))


if __name__ == '__main__':
    app.run(debug=True)

