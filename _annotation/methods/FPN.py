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
    transforms.Resize((128, 128)),  # Resize images to 128x128 (adjust as needed)
    transforms.ToTensor()
])

# Function to load training data
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
                        train_labels.append(label_to_index['a'])
                        train_polygons.append(polygons)
                    else:
                        print(f"No 'a' polygons found in {json_path}")
                else:
                    print(f"Polygons file not found: {json_path}")

    return torch.stack(train_images), torch.tensor(train_labels), train_polygons


# Define the Backbone (Simple CNN)
class BackboneCNN(nn.Module):
    def __init__(self):
        super(BackboneCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
    
    def forward(self, x):
        # Extract feature maps at each layer
        c1 = F.relu(self.conv1(x))  # 32 channels
        p1 = self.pool(c1)          # Downsample
        c2 = F.relu(self.conv2(p1)) # 64 channels
        p2 = self.pool(c2)          # Downsample
        c3 = F.relu(self.conv3(p2)) # 128 channels
        p3 = self.pool(c3)          # Downsample
        c4 = F.relu(self.conv4(p3)) # 256 channels
        p4 = self.pool(c4)          # Downsample
        return [p1, p2, p3, p4]     # Return feature maps from multiple layers

# Define FPN
class FPN(nn.Module):
    def __init__(self):
        super(FPN, self).__init__()
        
        self.lateral_p3 = nn.Conv2d(128, 128, kernel_size=1)
        self.lateral_p2 = nn.Conv2d(64, 64, kernel_size=1)
        self.lateral_p1 = nn.Conv2d(32, 32, kernel_size=1)
        
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.final_conv = nn.Conv2d(32, 128, kernel_size=3, padding=1)

    def forward(self, p1, p2, p3, p4):
        p3_fused = F.relu(self.lateral_p3(p3) + self.upsample(p4))
        p2_fused = F.relu(self.lateral_p2(p2) + self.upsample(p3_fused))
        p1_fused = F.relu(self.lateral_p1(p1) + self.upsample(p2_fused))
        final_feature = self.final_conv(p1_fused)
        return final_feature


# Combine the 2 models + classifier
class FPNModel(nn.Module):
    def __init__(self):
        super(FPNModel, self).__init__()
        self.backbone = BackboneCNN()
        self.fpn = FPN()
        self.classifier = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  # Global average pooling
            nn.Flatten(),
            nn.Linear(64, 2)  # Two classes: "a" (0) and "background" (1)
        )
    
    def forward(self, x):
        p1, p2, p3, p4 = self.backbone(x)
        fpn_features = self.fpn(p1, p2, p3, p4)
        output = self.classifier(fpn_features)
        return output


def train_model(model, train_images, train_labels, epochs=10, batch_size=16):

    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_dataset = torch.utils.data.TensorDataset(train_images, train_labels)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")

    print("Training complete!")
