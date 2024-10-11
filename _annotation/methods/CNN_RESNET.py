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

# Load ResNet model
class ResNetModel(nn.Module):
    def __init__(self):
        super(ResNetModel, self).__init__()
        self.model = models.resnet18(pretrained=True)  # Using ResNet18 as an example
        self.model.fc = nn.Linear(self.model.fc.in_features, 26)  # Assuming 26 classes (a-z)

    def forward(self, x):
        return self.model(x)

# Data loading function
import os
import json
from PIL import Image

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize all images to 224x224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_data(data_dir):
    """
    Load images and their corresponding polygons from a directory.
    
    :param data_dir: Directory containing subfolders with image and polygons.json
    :return: A tensor of images and a list of filtered polygons
    """
    train_images = []
    all_filtered_polygons = []

    # Iterate through each folder in the specified directory
    for folder in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder)

        # Check if the path is a directory
        if os.path.isdir(folder_path):
            image_path = os.path.join(folder_path, 'image.jpg')  # Path to the image
            json_path = os.path.join(folder_path, 'polygons.json')  # Path to polygons.json
            
            # Check if the image file exists
            if os.path.isfile(image_path):
                input_image = Image.open(image_path)
                
                # Apply transformations directly here
                input_image = transform(input_image)
                train_images.append(input_image)

                # Load existing polygons from JSON file
                if os.path.isfile(json_path):
                    with open(json_path, 'r') as f:
                        existing_data = json.load(f)

                    # Filter for polygons labeled "a"
                    filtered_polygons = [
                        item for item in existing_data if item.get('label') == 'a'
                    ]
                    all_filtered_polygons.append(filtered_polygons)
                else:
                    print(f"Polygons file not found: {json_path}")

    # Stack all images into a single tensor
    return torch.stack(train_images), all_filtered_polygons


# Prepare dataset class
class LetterDataset(Dataset):
    def __init__(self, image_list, label_list, transform=None):
        self.images = image_list
        self.labels = label_list
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

# Image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def collate_fn(batch):
    images, labels = zip(*batch)  # Unzip the batch
    images = [image.unsqueeze(0) for image in images]  # Add batch dimension
    return torch.cat(images, dim=0), labels

# Load training data
train_images, train_labels = load_data(result_dir)

# Create dataset and dataloader
train_dataset = LetterDataset(train_images, train_labels, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

# Initialize model, loss function, and optimizer
model = ResNetModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 5  # Adjust number of epochs as needed
for epoch in range(num_epochs):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

def infer_and_update_polygons(model, data_dir):
    # Process each folder in the data directory
    for folder in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder)
        if os.path.isdir(folder_path):
            image_path = os.path.join(folder_path, 'image.jpg')
            json_path = os.path.join(folder_path, 'polygons.json')

            # Load image and polygons
            input_image, filtered_polygons, existing_data = load_data(data_dir)

            # Preprocess the image
            preprocess = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            input_tensor = preprocess(input_image)
            input_batch = input_tensor.unsqueeze(0)  # Create a mini-batch

            # Move the input and model to GPU if available
            if torch.cuda.is_available():
                input_batch = input_batch.to('cuda')
                model.to('cuda')

            # Run the model to get predictions
            with torch.no_grad():
                output = model(input_batch)

            # Process output to find new detected polygons
            detected_polygons = []
            for polygon in filtered_polygons:
                # This is a placeholder for your actual detection logic
                detected_polygons.append({
                    "label": "detected",  # Mark detected polygons
                    "polygon": polygon["polygon"]  # Use your detected coordinates here
                })

            # Combine existing and newly detected polygons
            existing_data.extend(detected_polygons)

            # Write back to the JSON file
            with open(json_path, 'w') as f:
                json.dump(existing_data, f, indent=4)

# Example usage
infer_and_update_polygons(model, result_dir)

# Optional: Show a sample image
plt.imshow(train_images[0])
plt.title(train_labels[0])
plt.show()