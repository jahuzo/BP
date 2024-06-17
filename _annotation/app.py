from flask import redirect, url_for
from flask import Flask, render_template
from flask import request, jsonify
from flask import Response
import logging
import json
         
import cv2
import numpy as np
from PIL import Image, ImageDraw
import os

logging.basicConfig(level=logging.DEBUG)

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
    
    # Perform template matching
    res = cv2.matchTemplate(img_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    
    # Set a threshold for detecting matches
    threshold = 0.5
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
                {"x": int(pt[0]) + int(template_np.shape[1]), "y": int(pt[1] +template_np.shape[0])},
                {"x": int(pt[0]), "y": int(pt[1] + template_np.shape[0])}
            ]
            
            
        })
    
    return new_polygons

def compute_iou(polygon1, polygon2):
    # Calculate the intersection area
    intersection_area = 0
    for i in range(len(polygon1['points'])):
        for j in range(len(polygon2['points'])):
            x1, y1 = polygon1['points'][i]['x'], polygon1['points'][i]['y']
            x2, y2 = polygon1['points'][(i + 1) % len(polygon1['points'])]['x'], polygon1['points'][(i + 1) % len(polygon1['points'])]['y']
            x3, y3 = polygon2['points'][j]['x'], polygon2['points'][j]['y']
            x4, y4 = polygon2['points'][(j + 1) % len(polygon2['points'])]['x'], polygon2['points'][(j + 1) % len(polygon2['points'])]['y']
            
            intersection_area += compute_intersection_area(x1, y1, x2, y2, x3, y3, x4, y4)
    
    # Calculate the union area
    polygon1_area = calculate_polygon_area(polygon1)
    polygon2_area = calculate_polygon_area(polygon2)
    union_area = polygon1_area + polygon2_area - intersection_area
    
    # Calculate the IoU
    iou = intersection_area / union_area if union_area != 0 else 0
    return iou

def compute_intersection_area(x1, y1, x2, y2, x3, y3, x4, y4):
    # Calculate the intersection area between two line segments
    x = max(min(x1, x2), min(x3, x4))
    y = max(min(y1, y2), min(y3, y4))
    w = min(max(x1, x2), max(x3, x4)) - x
    h = min(max(y1, y2), max(y3, y4)) - y
    if w < 0 or h < 0:
        return 0
    return w * h

def calculate_polygon_area(polygon):
    # Calculate the area of a polygon using the shoelace formula
    area = 0
    for i in range(len(polygon['points'])):
        x1, y1 = polygon['points'][i]['x'], polygon['points'][i]['y']
        x2, y2 = polygon['points'][(i + 1) % len(polygon['points'])]['x'], polygon['points'][(i + 1) % len(polygon['points'])]['y']
        area += (x1 * y2 - x2 * y1)
    return abs(area) / 2

def aggregate_iou(detected_polygons, labeled_polygons): # compares each detected polygon with each labeled polygon
    iou_values = []
    for d_polygon in detected_polygons:
        best_iou = 0
        for l_polygon in labeled_polygons:
            iou = compute_iou(d_polygon, l_polygon)
            if iou > best_iou:
                best_iou = iou
        iou_values.append(best_iou)
    
    # Calculate the average IoU if there are any IoU values, else return 0
    return np.mean(iou_values) if iou_values else 0

app = Flask(__name__)

@app.route('/')
def hello_world():

    with open('polygons.json', 'r') as f:
        polygons = json.load(f)
    

    #polygons = [
    #  {"points": [{"x": 100, "y": 10}, {"x": 60, "y": 10}, {"x": 60, "y": 60}, {"x": 10, "y": 60}], "label": "a"},
    #  {"points": [{"x": 700, "y": 70}, {"x": 120, "y": 70}, {"x": 120, "y": 120}, {"x": 70, "y": 120}], "label": "n"},
    #  {"points": [{"x": 130, "y": 130}, {"x": 180, "y": 130}, {"x": 180, "y": 180}, {"x": 130, "y": 180}], "label": "a"},
    #  {"points": [], "label": ""}
    #]
    
    

    # Render the hello.html template
    return render_template('index.html', polygons=polygons)

@app.route('/predict')
def predict():
    
    with open('polygons.json', 'r') as f:
        polygons = json.load(f)
        #print_structure(polygons)
        
        # Open the source image
        source_img_path = os.path.join("static", "1.jpg")
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
            image_path = "static/1.jpg"
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
           
    # Save the data
    with open('polygons.json', 'w') as f:
        json.dump(matchesAll, f, indent=4)
        
    # Compute IoU and store it in results.json
    detected_polygons = [polygon for polygon in matchesAll if polygon["label"] == "detected"]
    labeled_polygons = [polygon for polygon in matchesAll if polygon["label"] == "a"]
    iou = aggregate_iou(detected_polygons, labeled_polygons)
    results = {"IoU": iou}
    app.logger.info(f"IoU: {results}") # does not work...
    try:
        with open('results.json', 'w') as f:
            json.dump(results, f, indent=4)
    except Exception as e:
        print(f"Error writing to JSON: {e}")

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
    

if __name__ == '__main__':
    app.run(debug=True)



