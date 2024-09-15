import cv2
import numpy as np
import json
import os
from PIL import Image, ImageDraw

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


def find_matches(img, template, polygons, min_distance, scale_percent=50, threshold=0.8):  # migrate to baseline.py
    # Convert PIL images to NumPy arrays for OpenCV processing
    img_np = np.array(img)
    template_np = np.array(template)

    # Convert to grayscale
    img_gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template_np, cv2.COLOR_BGR2GRAY)
    
    # Resize images for faster processing
    width = int(img_gray.shape[1] * scale_percent / 100)
    height = int(img_gray.shape[0] * scale_percent / 100)
    dim = (width, height)

    img_resized = cv2.resize(img_gray, dim, interpolation=cv2.INTER_AREA)
    template_resized = cv2.resize(template_gray, dim, interpolation=cv2.INTER_AREA)

    # Perform template matching using cross-correlation
    res = cv2.matchTemplate(img_resized, template_resized, cv2.TM_CCORR_NORMED)
    
    # Set a threshold for detecting matches
    loc = np.where(res >= threshold)
    
    new_polygons = []
    for pt in zip(*loc[::-1]):  # loc gives us the top-left corner of the match
        match_center = (pt[0] + template_resized.shape[1]//2, pt[1] + template_resized.shape[0]//2)  # Calculate match center
        # If far enough, add to new polygons list
        new_polygons.append({
            "label": "detected",
            "points": [
                {"x": int(pt[0]), "y": int(pt[1])},
                {"x": int(pt[0]) + int(template_resized.shape[1]), "y": int(pt[1])},
                {"x": int(pt[0]) + int(template_resized.shape[1]), "y": int(pt[1] + template_resized.shape[0])},
                {"x": int(pt[0]), "y": int(pt[1] + template_resized.shape[0])}
            ]
        })
        
def baseline(polygons, source_img ):
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
        
        matches = find_matches(source_img, img, polygons, 50)
        #print(matches)
        #matchesAll = merge_if_far_enough(matchesAll, matches, min_distance)
        #print_structure(matches)
        #matchesAll.extend(matches)
        min_distance = 500  # Define the minimum distance
        matchesAll = add_distant_detections(matchesAll, matches, min_distance)

    distance_threshold = 10  # Define the minimum distance between polygon centers
    matchesAll = filter_polygons(matchesAll, distance_threshold)