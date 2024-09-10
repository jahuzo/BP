import cv2
import numpy as np
import json
import os

def load_polygons(folder_name, filename='polygons.json'):
    file_path = os.path.join('static', folder_name, filename)
    with open(file_path, 'r') as f:
        polygons = json.load(f)
    return polygons


def extract_region(img, polygon):
    # Create a mask for the polygon
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    points = np.array([[point['x'], point['y']] for point in polygon['points']], dtype=np.int32)
    cv2.fillPoly(mask, [points], 255)
    
    # Extract the region using the mask
    region = cv2.bitwise_and(img, img, mask=mask)
    return region

def non_max_suppression(boxes, overlapThresh):
    if len(boxes) == 0:
        return []

    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    pick = []
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs[:last]]

        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

    return boxes[pick].astype("int")

def find_matches(img, template, json_path, min_distance):
    # Load user-created polygons from JSON file
    polygons = load_polygons(json_path)
    
    # Convert PIL images to NumPy arrays for OpenCV processing
    img_np = np.array(img)
    template_np = np.array(template)

    # Convert to grayscale
    img_gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template_np, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise and improve matching accuracy
    img_gray = cv2.GaussianBlur(img_gray, (5, 5), 0)
    template_gray = cv2.GaussianBlur(template_gray, (5, 5), 0)
    
    new_polygons = []
    boxes = []
    
    for polygon in polygons:
        # Extract the region defined by the polygon
        region = extract_region(img_np, polygon)
        region_gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        
        # Perform template matching within the region
        res = cv2.matchTemplate(region_gray, template_gray, cv2.TM_CCORR_NORMED)
        
        # Set a threshold for detecting matches
        threshold = 0.7  # Adjusted threshold for better accuracy
        loc = np.where(res >= threshold)
        
        for pt in zip(*loc[::-1]):  # loc gives us the top-left corner of the match
            match_center = (pt[0] + template_np.shape[1]//2, pt[1] + template_np.shape[0]//2)  # Calculate match center
            # Add to boxes list for non-maximum suppression
            boxes.append([pt[0], pt[1], pt[0] + template_np.shape[1], pt[1] + template_np.shape[0]])
    
    # Apply non-maximum suppression to filter out overlapping detections
    boxes = np.array(boxes)
    boxes = non_max_suppression(boxes, 0.3)
    
    for (x1, y1, x2, y2) in boxes:
        new_polygons.append({
            "label": "detected",
            "points": [
                {"x": int(x1), "y": int(y1)},
                {"x": int(x2), "y": int(y1)},
                {"x": int(x2), "y": int(y2)},
                {"x": int(x1), "y": int(y2)}
            ]
        })
    
    return new_polygons