# mandatory for paths import
import sys
sys.path.append(r'/mnt/c/Users/jahuz/Links/BP/_annotation')

# header file basically
from paths import *

def load_polygons(json_file):
    with open(json_file) as f:
        data = json.load(f)
    return data  # Return the entire JSON content for processing

def plot_image_with_polygons(image_file, polygons):
    # Load the image
    image = cv2.imread(image_file)
    # Convert BGR (OpenCV) to RGB (Matplotlib)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    plt.imshow(image)
    
    # Plot each polygon
    for polygon in polygons:
        label = polygon.get('label', None)

        # Handle 'points' structure
        if 'points' in polygon:
            points = polygon['points']
            x = [point['x'] for point in points]
            y = [point['y'] for point in points]
        
        # Handle 'polygon' structure (can be either 4 values or a list of points)
        elif 'polygon' in polygon:
            bbox = polygon['polygon']

            # Check if the polygon is a list of points
            if isinstance(bbox, list) and all(isinstance(p, dict) and 'x' in p and 'y' in p for p in bbox):
                x = [point['x'] for point in bbox]
                y = [point['y'] for point in bbox]
            else:
                # Treat as a bounding box and expect 4 values
                if len(bbox) == 4:
                    x = [bbox[0], bbox[2], bbox[2], bbox[0]]  # x-coordinates of the bounding box
                    y = [bbox[1], bbox[1], bbox[3], bbox[3]]  # y-coordinates of the bounding box
                else:
                    print(f"Warning: Expected 4 values or list of points in 'polygon' for label '{label}', but got {bbox}")
                    continue
        else:
            continue  # Skip if neither structure is found

        # Determine color based on label
        if label == "a":
            color = 'green'
        elif label == "detected":
            color = 'red'
        else:
            continue  # Skip other labels
            
        # Create a polygon patch
        plt.fill(x, y, alpha=0.5, color=color)  # Fill the polygon with specified color
        plt.text(x[0], y[0], label, fontsize=10, color='black')  # Label at the first point

    plt.get_current_fig_manager().window.state('normal')  # For Windows
    plt.axis('off')  # Turn off axis numbers and ticks
    plt.show()

def main(directory):
    image_file = os.path.join(directory, 'image.jpg')
    json_file = os.path.join(directory, 'polygons.json')
    
    polygons = load_polygons(json_file)
    plot_image_with_polygons(image_file, polygons)

if __name__ == "__main__":
    # Change the directory here (e.g., '001', '002', ... '007')
    directory = os.path.join(result_dir, '009') # Change the directory here
    main(directory)
    delete_detected_labels(result_dir) # added to delete past labels
