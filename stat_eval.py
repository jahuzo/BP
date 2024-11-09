import sys

# Paths import (update the path accordingly)
sys.path.append(r'/mnt/c/Users/jahuz/Links/BP/_annotation')

from paths import *

import json
from shapely.geometry import Polygon


def calculate_iou(polygon1_points, polygon2_points):
    """
    Calculate the Intersection over Union (IoU) between two polygons.
    Each polygon is defined by a list of points, where each point is a dictionary with 'x' and 'y' keys.
    """
    # Convert points to (x, y) tuples for shapely
    poly1 = Polygon([(point['x'], point['y']) for point in polygon1_points])
    poly2 = Polygon([(point['x'], point['y']) for point in polygon2_points])
    
    # Calculate intersection and union areas
    intersection_area = poly1.intersection(poly2).area
    union_area = poly1.union(poly2).area
    
    # Calculate IoU
    iou = intersection_area / union_area if union_area > 0 else 0
    return iou

def count_true_detections(json_file_path, threshold=0.3, false_positive_threshold=0.175):
    """
    Load the JSON file, calculate IoU between every "a" and "detected" label,
    print the highest IoU for each detected polygon, count true detections, 
    calculate precision and recall based on the threshold, and save blatant false positives.
    """
    # Load the JSON file
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    
    # Separate "a" and "detected" polygons
    a_polygons = [item['points'] for item in data if item['label'] == "a"]
    detected_polygons = [item['polygon'] for item in data if item['label'] == "detected"]
    
    # Initialize counts
    true_positives = 0
    false_positives = 0

    # Keep track of which ground truth polygons have been matched
    matched_gt_indices = set()

    # List to store blatant false positives
    blatant_false_positives = []

    # Loop through each detected polygon
    for det_index, detected_polygon in enumerate(detected_polygons):
        max_iou = 0  # Track the maximum IoU for this detected polygon
        matched_gt_index = -1  # Index of the matched ground truth polygon

        # Compare with each "a" polygon
        for a_index, a_polygon in enumerate(a_polygons):
            if a_index in matched_gt_indices:
                continue  # Skip already matched ground truth polygons

            # Calculate IoU between this detected and "a" polygon
            iou = calculate_iou(a_polygon, detected_polygon)
            if iou > max_iou:
                max_iou = iou
                matched_gt_index = a_index  # Potential match

        # Print IoU for this detection
        print(f"Detection {det_index + 1}: Highest IoU = {max_iou:.4f}")
        
        # Determine if it is a true or false positive
        if max_iou >= threshold and matched_gt_index != -1:
            print(f"Detection {det_index + 1} is a True Detection")
            true_positives += 1
            matched_gt_indices.add(matched_gt_index)  # Mark this ground truth as matched
        else:
            print(f"Detection {det_index + 1} is a False Detection")
            false_positives += 1
            # Check for blatant false positive (IoU below 0.2)
            if max_iou < false_positive_threshold:
                print(f"Detection {det_index + 1} is a Blatant False Positive")
                blatant_false_positives.append({"label": "FP", "polygon": detected_polygon})

    # Calculate False Negatives
    false_negatives = len(a_polygons) - len(matched_gt_indices)
    
    # Calculate and print precision and recall
    total_detections = true_positives + false_positives
    precision = (true_positives / total_detections) * 100 if total_detections > 0 else 0
    recall = (true_positives / (true_positives + false_negatives)) * 100 if (true_positives + false_negatives) > 0 else 0
    print(f"Total number of true detections (TP): {true_positives}")
    print(f"Total number of false detections (FP): {false_positives}")
    print(f"Total number of false negatives (FN): {false_negatives}")
    print(f"Precision: {precision:.2f}%")
    print(f"Recall: {recall:.2f}%")

    # Save blatant false positives back to the JSON file
    data.extend(blatant_false_positives)
    with open(json_file_path, 'w') as file:
        json.dump(data, file, indent=4)
    print(f"Blatant false positives saved to {json_file_path}")
    
    return precision, recall

# Example usage
json_file_path = os.path.join(result_dir, '009', 'polygons.json')    
print(json_file_path)
result = count_true_detections(json_file_path, threshold=0.3)
#print("Number of true detections:", result)
