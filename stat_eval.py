import numpy as np

def compute_iou(polygon1, polygon2):
    # Calculate the intersection area
    intersection_area = 0
    for i in range(len(polygon1['points'])):
        for j in range(len(polygon2['points'])):
            x1, y1 = polygon1['points'][i]['x'], polygon1['points'][i]['y']
            x2, y2 = polygon2['points'][j]['x'], polygon2['points'][j]['y']
            # Implement intersection area calculation here
            # This is a placeholder for the actual intersection logic
            pass
    # Calculate the union area
    union_area = 0  # Placeholder for union area calculation
    iou = intersection_area / union_area if union_area != 0 else 0
    return iou

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


def stat_calc(matchesAll):
    #IoU
    detected_polygons = [polygon for polygon in matchesAll if polygon["label"] == "detected"]
    labeled_polygons = [polygon for polygon in matchesAll if polygon["label"] == "a"]
    iou = aggregate_iou(detected_polygons, labeled_polygons)
    #FPR
    #fpr = fp/(fp+tn)

    #TPR
    #tpr = tp/(tp+fn)

    #Precision
    #prec = tp/(tp+fp)
    
    #Accuracy
    #acc = (tp+tn)/(tp+tn+fp+fn)
    
    #Recall
    #rec = tp/(tp+fn)
    
    #return iou, fpr, tpr, prec, acc, rec
    return iou
