# mandatory for paths import
import sys
sys.path.append(r'/mnt/c/Users/jahuz/Links/BP/_annotation')

# header file basically
from paths import *

# Debugging and checks
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

input("Press Enter to continue...")
print("OK...moving on")

root_dir = result_dir

# Define label map
label_to_index = {"a": 0}  # You can expand this as needed

# Transform for input images
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),  # Add vertical flip
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Wider range
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Random translation
    transforms.ToTensor()
])


def iou_loss(pred_boxes, target_boxes):
    # Compute IoU loss between predicted and target boxes
    pred_xmin, pred_ymin, pred_xmax, pred_ymax = pred_boxes.T
    target_xmin, target_ymin, target_xmax, target_ymax = target_boxes.T

    # Compute intersection
    inter_xmin = torch.max(pred_xmin, target_xmin)
    inter_ymin = torch.max(pred_ymin, target_ymin)
    inter_xmax = torch.min(pred_xmax, target_xmax)
    inter_ymax = torch.min(pred_ymax, target_ymax)
    inter_area = torch.clamp(inter_xmax - inter_xmin, min=0) * torch.clamp(inter_ymax - inter_ymin, min=0)

    # Compute areas
    pred_area = (pred_xmax - pred_xmin) * (pred_ymax - pred_ymin)
    target_area = (target_xmax - target_xmin) * (target_ymax - target_ymin)
    union_area = pred_area + target_area - inter_area

    # IoU
    iou = inter_area / union_area
    
    # Return a scalar loss by averaging the IoU
    return 1 - iou.mean()  # Loss is 1 - IoU (so higher IoU leads to lower loss)

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
    train_polygons = []  # Store polygons associated with each image

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
                            # Take the first bounding box, or you can aggregate them based on your use case
                            train_labels.append(torch.tensor(bboxes[0]).float())  # Only the first box for each image
                        else:
                            print(f"No bounding boxes created for {json_path}")
                    else:
                        print(f"No 'a' polygons found in {json_path}")
                else:
                    print(f"Polygons file not found: {json_path}")

    return torch.stack(train_images), torch.stack(train_labels)  # Make sure to stack train_labels


class BackboneCNN(nn.Module):
    def __init__(self):
        super(BackboneCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        c1 = F.relu(self.conv1(x))
        p1 = self.pool(c1)
        c2 = F.relu(self.conv2(p1))
        p2 = self.pool(c2)
        c3 = F.relu(self.conv3(p2))
        p3 = self.pool(c3)
        c4 = F.relu(self.conv4(p3))
        p4 = self.pool(c4)
        return p1, p2, p3, p4

class FPN(nn.Module):
    def __init__(self):
        super(FPN, self).__init__()
        self.lateral_p4 = nn.Conv2d(256, 128, kernel_size=1)
        self.lateral_p3 = nn.Conv2d(128, 128, kernel_size=1)
        self.lateral_p2 = nn.Conv2d(64, 128, kernel_size=1)
        self.lateral_p1 = nn.Conv2d(32, 128, kernel_size=1)
        
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.final_conv = nn.Conv2d(128, 128, kernel_size=3, padding=1)

    def forward(self, p1, p2, p3, p4):
        p4_ = self.lateral_p4(p4)
        p3_fused = F.relu(self.lateral_p3(p3) + self.upsample(p4_))
        p2_fused = F.relu(self.lateral_p2(p2) + self.upsample(p3_fused))
        p1_fused = F.relu(self.lateral_p1(p1) + self.upsample(p2_fused))
        final_feature = self.final_conv(p1_fused)
        return final_feature

class FPNModel(nn.Module):
    def __init__(self):
        super(FPNModel, self).__init__()
        self.backbone = BackboneCNN()
        self.fpn = FPN()
        self.classifier = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 4),
            nn.Sigmoid()  # Constrain output to [0, 1]
        )
    
    def forward(self, x):
        p1, p2, p3, p4 = self.backbone(x)
        fpn_features = self.fpn(p1, p2, p3, p4)
        output = self.classifier(fpn_features)
        return output


def train_model(model, train_images, train_labels, epochs=10, batch_size=16):
    model.to(device)

    criterion = nn.SmoothL1Loss()  # Still using Smooth L1 Loss for bounding box regression
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Optional learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    train_dataset = torch.utils.data.TensorDataset(train_images, train_labels)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        running_loss = 0.0
        model.train()
        
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            
            # Scale outputs to image dimensions
            outputs_scaled = outputs * torch.tensor([128, 128, 128, 128]).to(device)
            
            # Compute loss
            loss = criterion(outputs_scaled, labels.float())
            loss = loss.mean()  # Ensure it's a scalar

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Learning rate adjustment
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
        for epoch in range(epochs):
            # Training code
            scheduler.step(running_loss)

        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")

    print("Training complete!")


def infer_and_update_polygons(model, data_dir):
    model.eval()
    model.to(device)

    for folder in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder)
        
        if os.path.isdir(folder_path):
            image_path = os.path.join(folder_path, 'image.jpg')
            json_path = os.path.join(folder_path, 'polygons.json')
            
            if not os.path.isfile(image_path):
                print(f"Image not found: {image_path}")
                continue

            input_image = Image.open(image_path)

            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    existing_data = json.load(f)
            else:
                existing_data = []
            
            preprocess = transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
            ])
            input_tensor = preprocess(input_image).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(input_tensor)
                bounding_box = output[0].cpu().numpy()
                xmin, ymin, xmax, ymax = bounding_box

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

# Load training data
train_images, train_labels = load_data(result_dir)

# Initialize and train model
model = FPNModel()
train_model(model, train_images, train_labels, epochs=10)
infer_and_update_polygons(model, result_dir)

