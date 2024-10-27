import sys

# Paths import (update the path accordingly)
sys.path.append(r'/mnt/c/Users/jahuz/Links/BP/_annotation')
from paths import *

# Debugging and checks
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

root_dir = result_dir

# Define label map
label_to_index = {"a": 0}  # Only detecting letter 'a'

# Transform for input images
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),  # Reduce rotation range to make task easier
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),  # Reduce translation
    transforms.RandomApply([transforms.GaussianBlur(3)], p=0.3),  # Add Gaussian Blur
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize to match pre-trained model
])

def calculate_bounding_box(polygon):
    points = polygon['points']
    if len(points) < 3:
        raise ValueError("Polygon does not have enough points to form a bounding box.")
    
    x_coords = [point['x'] for point in points]
    y_coords = [point['y'] for point in points]
    
    # Calculate bounding box
    xmin = min(x_coords)
    xmax = max(x_coords)
    ymin = min(y_coords)
    ymax = max(y_coords)
    
    return [xmin, ymin, xmax, ymax]

def preprocess_polygons_to_bboxes(polygons):
    bboxes = []
    for polygon in polygons:
        if 'points' in polygon:
            try:
                bbox = calculate_bounding_box(polygon)
                bboxes.append(bbox)
            except ValueError as e:
                print(f"Error processing polygon: {e}")
    return bboxes

def load_data(data_dir):
    train_images = []
    train_labels = []

    for folder in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder)
        if os.path.isdir(folder_path):
            image_path = os.path.join(folder_path, 'image.jpg')
            json_path = os.path.join(folder_path, 'polygons.json')
            
            if os.path.isfile(image_path):
                input_image = Image.open(image_path)
                input_image = transform(input_image)
                train_images.append(input_image)

                if os.path.isfile(json_path):
                    with open(json_path, 'r') as f:
                        existing_data = json.load(f)
                    
                    # Only use polygons labeled as "a"
                    polygons = [polygon for polygon in existing_data if polygon['label'] == 'a']
                    if polygons:
                        # Convert polygons to bounding boxes
                        bboxes = preprocess_polygons_to_bboxes(polygons)

                        # Assuming each image has only one bounding box for simplicity
                        if bboxes:
                            train_labels.append(torch.tensor(bboxes[0]).float())  # Only the first box for each image
                        else:
                            print(f"No bounding boxes created for {json_path}")
                    else:
                        print(f"No 'a' polygons found in {json_path}")
                else:
                    print(f"Polygons file not found: {json_path}")

    return torch.stack(train_images), torch.stack(train_labels)

# Load training data
train_images, train_labels = load_data(result_dir)

# Load a pre-trained Faster R-CNN model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

# Adjust the classifier to detect only one class ('a' + background)
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
    model.roi_heads.box_predictor.cls_score.in_features, 2  # background + 'a'
)

# Freeze backbone layers initially to fine-tune the head
for param in model.backbone.parameters():
    param.requires_grad = False

model.to(device)

# Training configuration
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.00001, weight_decay=1e-5)  # Lower learning rate
criterion = nn.SmoothL1Loss()  # Use Smooth L1 loss for bounding box regression
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

# Create DataLoader
train_dataset = TensorDataset(train_images, train_labels)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)  # Reduce batch size

def train_model(model, train_loader, epochs=10):
    for epoch in range(epochs):
        running_loss = 0.0
        model.train()
        
        for i, (images, labels) in enumerate(train_loader):
            images = list(image.to(device) for image in images)
            labels = [{"boxes": label.unsqueeze(0).to(device), "labels": torch.tensor([1], dtype=torch.int64).to(device)} for label in labels]

            optimizer.zero_grad()
            
            # Forward pass
            loss_dict = model(images, labels)
            losses = sum(loss for loss in loss_dict.values())
            
            # Backward pass
            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)  # Gradient clipping
            optimizer.step()
            
            running_loss += losses.item()

        # Learning rate adjustment
        scheduler.step(running_loss)
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")

        # Unfreeze backbone after a few epochs
        if epoch == 2:  # Unfreeze after epoch 2
            for param in model.backbone.parameters():
                param.requires_grad = True

    print("Training complete!")

# Train the model
train_model(model, train_loader, epochs=10)

# Inference section
def infer_and_update_polygons(model, data_dir):
    model.eval()
    model.to(device)

    for folder in os.listdir(data_dir):
        if folder not in ['008', '009']:
            continue

        folder_path = os.path.join(data_dir, folder)
        
        if os.path.isdir(folder_path):
            image_path = os.path.join(folder_path, 'image.jpg')
            json_path = os.path.join(folder_path, 'polygons.json')
            
            if not os.path.isfile(image_path):
                print(f"Image not found: {image_path}")
                continue

            input_image = Image.open(image_path)
            input_tensor = transforms.ToTensor()(input_image).unsqueeze(0).to(device)

            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    existing_data = json.load(f)
            else:
                existing_data = []

            with torch.no_grad():
                predictions = model(input_tensor)
                if len(predictions) > 0:
                    prediction = predictions[0]
                    if len(prediction['boxes']) > 0:
                        # Take the first bounding box (assuming only one 'a' per image)
                        box = prediction['boxes'][0].cpu().numpy()
                        xmin, ymin, xmax, ymax = box

                        detected_box = {
                            "label": "detected",
                            "polygon": [
                                {"x": int(xmin), "y": int(ymin)},
                                {"x": int(xmax), "y": int(ymin)},
                                {"x": int(xmax), "y": int(ymax)},
                                {"x": int(xmin), "y": int(ymax)}
                            ]
                        }

                        if detected_box not in existing_data:
                            existing_data.append(detected_box)
                            with open(json_path, 'w') as f:
                                json.dump(existing_data, f, indent=4)
                            print(f"Updated polygons saved in {json_path}")
                        else:
                            print(f"No new detected polygons in {json_path}")


# Train the model and infer 
train_model(model, train_loader, epochs=10)
infer_and_update_polygons(model, result_dir)