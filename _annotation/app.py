from operator import index
from flask import redirect, url_for
from flask import Flask, render_template
from flask import request, jsonify
from flask import Response
from prettytable import PrettyTable
import logging
import json
         
import cv2
import numpy as np
from PIL import Image, ImageDraw
import os

logging.basicConfig(level=logging.DEBUG)

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

doc_dir = 'static'
doc_index = os.path.join(doc_dir, 'index.json')

def load_index():
    with open(doc_index, 'r') as f:
        return json.load(f)

def get_doc(doc_id):
    index = load_index()
    document = next((doc for doc in index['documents'] if doc['id'] == doc_id), None)
    
    if document:
        # Serve the image and polygon data for the selected document
        image_path = document['image_path']
        json_path = document['json_path']
        
        with open(json_path, 'r') as f:
            polygons = json.load(f)
        
        return jsonify({
            'image': image_path,
            'polygons': polygons,
            'current_page': document['current_page']
        })
    else:
        return jsonify({'error': 'Document not found'}), 404

def optimize_polygon(polygons):
    optimized_polygons = []

    for polygon in polygons:
        # Calculate the bounding box for the polygon
        min_x = min(point['x'] for point in polygon['points'])
        min_y = min(point['y'] for point in polygon['points'])
        max_x = max(point['x'] for point in polygon['points'])
        max_y = max(point['y'] for point in polygon['points'])

        # Adjust polygon points relative to the bounding box's top-left corner
        adjusted_points = [{'x': point['x'] - min_x, 'y': point['y'] - min_y} for point in polygon['points']]

        # Construct new polygon with adjusted points
        optimized_polygon = {
            'label': polygon.get('label', ''),  # Preserve label if it exists
            'points': adjusted_points,
            # Optionally include the bounding box dimensions for reference or further processing
            'bounding_box': {'min_x': min_x, 'min_y': min_y, 'max_x': max_x, 'max_y': max_y}
        }

        optimized_polygons.append(optimized_polygon)

    return optimized_polygons


def calculate_centroid(polygon):
    """Calculate the centroid of a polygon given its vertices."""
    x_coords = [point['x'] for point in polygon['points']]
    y_coords = [point['y'] for point in polygon['points']]
    centroid_x = sum(x_coords) / len(polygon['points'])
    centroid_y = sum(y_coords) / len(polygon['points'])
    return {'x': centroid_x, 'y': centroid_y}

def calculate_distance(centroid1, centroid2):
    """Calculate the Euclidean distance between two centroids."""
    return ((centroid1['x'] - centroid2['x'])**2 + (centroid1['y'] - centroid2['y'])**2) ** 0.5

def filter_polygons(polygons, distance):
    """Filter out polygons that are too close to each other, based on centroid distance."""
    filtered_polygons = []
    for polygon in polygons:
        polygon_centroid = calculate_centroid(polygon)
        too_close = False
        for existing_polygon in filtered_polygons:
            existing_polygon_centroid = calculate_centroid(existing_polygon)
            if calculate_distance(polygon_centroid, existing_polygon_centroid) < distance:
                too_close = True
                break
        if not too_close:
            filtered_polygons.append(polygon)
    return filtered_polygons

def is_far_enough(new_detection, existing_detections, min_distance):
    """Check if new_detection is at least min_distance away from all existing_detections."""
    for existing_detection in existing_detections:
        for new_point in new_detection['points']:
            for existing_point in existing_detection['points']:
                distance = ((new_point['x'] - existing_point['x'])**2 + (new_point['y'] - existing_point['y'])**2) ** 0.5
                if distance < min_distance:
                    return False
    return True

def add_distant_detections(matchesAll, matches, min_distance):
    """Add new detections to matchesAll if they are sufficiently distant from existing detections."""
    filtered_matches = [match for match in matches if is_far_enough(match, matchesAll, min_distance)]
    matchesAll.extend(filtered_matches)
    return matchesAll


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


def find_matches(img, template, polygons, min_distance):
    # Convert PIL images to NumPy arrays for OpenCV processing
    img_np = np.array(img)
    template_np = np.array(template)

    # Convert to grayscale
    img_gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template_np, cv2.COLOR_BGR2GRAY)
    
    # Perform template matching using cross-correlation
    res = cv2.matchTemplate(img_gray, template_gray, cv2.TM_CCORR_NORMED)
    
    # Set a threshold for detecting matches
    threshold = 0.  # Adjusted threshold for better accuracy
    loc = np.where(res >= threshold)
    
    new_polygons = []
    for pt in zip(*loc[::-1]):  # loc gives us the top-left corner of the match
        match_center = (pt[0] + template_np.shape[1]//2, pt[1] + template_np.shape[0]//2)  # Calculate match center
        # If far enough, add to new polygons list
        new_polygons.append({
            "label": "detected",
            "points": [
                {"x": int(pt[0]), "y": int(pt[1])},
                {"x": int(pt[0]) + int(template_np.shape[1]), "y": int(pt[1])},
                {"x": int(pt[0]) + int(template_np.shape[1]), "y": int(pt[1] + template_np.shape[0])},
                {"x": int(pt[0]), "y": int(pt[1] + template_np.shape[0])}
            ]
        })
    
def load_polygons(folder_name, filename='polygons.json'):
    file_path = os.path.join('static', folder_name, filename)
    with open(file_path, 'r') as f:
        polygons = json.load(f)
    return polygons


app = Flask(__name__)
@app.route('/')
def hello_world():

    cur_dir = request.args.get('folder', 'default_folder')  
    polygons = load_polygons(cur_dir)
    return render_template('index.html', polygons=polygons, folder_name=cur_dir)

@app.route('/predict')
def predict():
    
    with open('polygons.json', 'r') as f:
        polygons = json.load(f)
        #print_structure(polygons)
        
        # Open the source image
        source_img_path = os.path.join("static", "001.jpg")
        source_img = Image.open(source_img_path)
    
        matchesAll = []    
        matchesAll.extend(polygons)
                
        for index, polygon in enumerate(polygons):
        
            if polygon["label"] != "a":
                continue
                
            # Convert polygon points to a format suitable for PIL (list of tuples)
            print(polygon)
            polygon_points = [(int(point["x"]), int(point["y"])) for point in polygon['points']]
            
            # Determine the bounding box of the polygon
            min_x = min(point[0] for point in polygon_points)
            min_y = min(point[1] for point in polygon_points)
            max_x = max(point[0] for point in polygon_points)
            max_y = max(point[1] for point in polygon_points)
            
            # Create a new image with white background
            img_size = (max_x - min_x, max_y - min_y)
            img = Image.new("RGB", img_size, "white")
            
            # Calculate the offset to crop the source image correctly
            offset = (min_x, min_y)
            
            # Crop the source image to the bounding box of the polygon
            source_img_cropped = source_img.crop((min_x, min_y, max_x, max_y))
            
            # Paste the cropped source image onto the new image
            img.paste(source_img_cropped, (0, 0))
            
            # Draw the polygon on the new image
            draw = ImageDraw.Draw(img)
            draw.polygon(polygon_points, outline="red")
            
            # Load the image from the path
            image_path = "static/001.jpg"
            reference = Image.open(image_path)
            
            matches = find_matches(reference, img, polygons, 50)
            #print(matches)
            #matchesAll = merge_if_far_enough(matchesAll, matches, min_distance)
            #print_structure(matches)
            #matchesAll.extend(matches)
            min_distance = 500  # Define the minimum distance
            matchesAll = add_distant_detections(matchesAll, matches, min_distance)

    distance_threshold = 10  # Define the minimum distance between polygon centers
    matchesAll = filter_polygons(matchesAll, distance_threshold)
    #matchesAll = optimize_polygon(matchesAll)
    

@app.route('/submit-polygons', methods=['POST'])
def submit_polygons():
      # Parse the JSON data
      polygons = request.get_json()
      
      # Filter out polygons with fewer than 3 points
      polygons = [polygon for polygon in polygons if len(polygon["points"]) >= 3]
      
      # Save the data
      with open('polygons.json', 'w') as f:
          json.dump(polygons, f, indent=4)
  
      return redirect('/')  
      #return redirect(url_for('/'))

@app.route('/change', methods=['POST'])
def change():
    # Get the current filename from the request
    filename = request.form.get('filename')

    # Get the directory of the current file
    current_directory = os.path.dirname(filename)

    # Get the list of files in the current directory
    files = [f for f in os.listdir(current_directory) if f.endswith('.json')]  # Only consider JSON files

    # Find the index of the current file
    current_index = files.index(os.path.basename(filename))

    # Calculate the index of the next file
    next_index = (current_index + 1) % len(files)

    # Get the path of the next file
    next_file = os.path.join(current_directory, files[next_index])

    return redirect(url_for('hello_world', filename=next_file))  # Pass the next filename when redirecting


if __name__ == '__main__':
    app.run(debug=True)



