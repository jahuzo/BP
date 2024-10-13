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

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize all images to 224x224 as defined by torch resnet docs
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

label_to_index = {'a': 0}  # Only mapping for 'a'

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
                    
                    # Assuming you're interested in the first polygon's label
                    label = existing_data[0].get('label', None)
                    if label in label_to_index:
                        train_labels.append(label_to_index[label])
                    else:
                        print(f"Unknown label {label} in {json_path}")
                else:
                    print(f"Polygons file not found: {json_path}")

    return torch.stack(train_images), torch.tensor(train_labels)

# Prepare dataset class
class LetterDataset(Dataset):
    def __init__(self, image_list, label_list, transform=None):
        self.images = image_list
        self.labels = label_list
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]  # This will be a tensor already transformed
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)  # This line can be omitted if images are already tensors

        return image, label  # No additional transformations needed here

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

# Training loop
batch_size = 16
num_samples = train_images.size(0)
num_batches = (num_samples + batch_size - 1) // batch_size

# Initialize model, loss function, and optimizer
model = ResNetModel().to(device)  # Use 'weights' parameter
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
batch_size = 16
num_samples = train_images.size(0)
num_batches = (num_samples + batch_size - 1) // batch_size
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, num_samples)

        images = train_images[start_idx:end_idx].to(device)
        labels = train_labels[start_idx:end_idx].to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

def infer_and_update_polygons(model, data_dir):
    """
    Run inference on each image in the directory, and update the polygons.json with detected polygons.
    
    :param model: Trained model to run inference.
    :param data_dir: Directory where images and polygons.json are stored.
    """
    # Iterate through each folder in the directory
    for folder in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder)
        
        if os.path.isdir(folder_path):
            image_path = os.path.join(folder_path, 'image.jpg')
            json_path = os.path.join(folder_path, 'polygons.json')
            
            # Load image
            input_image = Image.open(image_path)

            # Load polygons.json data
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    existing_data = json.load(f)
            else:
                print(f"No polygons.json found in {folder}")
                continue
            
            # Preprocess the image for the model
            preprocess = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            input_tensor = preprocess(input_image).unsqueeze(0)  # Add batch dimension

            # Move to device (GPU or CPU)
            input_tensor = input_tensor.to(device)
            model.to(device)

            # Run inference to get predictions
            model.eval()
            with torch.no_grad():
                output = model(input_tensor)
                # You will need to add actual logic here for processing the output to get "detected" polygons.
                # Placeholder: Let's assume `detected_polygons` is obtained from your model's output
                detected_polygons = [
                    {
                        "label": "detected",
                        "polygon": [100, 200, 150, 250]  # Placeholder polygon coordinates
                    }
                ]

            # Update polygons.json by adding the new "detected" polygons
            for polygon in detected_polygons:
                existing_data.append(polygon)

            # Write updated data back to polygons.json
            with open(json_path, 'w') as f:
                json.dump(existing_data, f, indent=4)

            print(f"Updated polygons saved in {json_path}")


# Example usage
infer_and_update_polygons(model, result_dir)

