import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from PIL import Image

# Transformation pipeline (resize and normalize images)
transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Folders for training and testing images
folder1 = 'data/+'
folder2 = 'data/g'


# Dataset class for loading image folders
class SymbolDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []

        # load images and labels
        for label, folder_name in enumerate([folder1, folder2]):
            folder_path = os.path.join(root_dir, folder_name)
            for img_file in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img_file)
                self.images.append(img_path)
                self.labels.append(label)

    # return length of dataset
    def __len__(self):
        return len(self.images)

    # return image and label at index
    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return img, label


# Load dataset
dataset = SymbolDataset('.', transform=transform)

# Ensure at least 30 images per class
assert sum(label == 0 for label in dataset.labels) >= 30 and sum(label == 1 for label in dataset.labels) >= 30, \
    "There must be at least 30 images per symbol."

# Split into train and test sets (20 train + 10 test per class)
plus_indices = [i for i, label in enumerate(dataset.labels) if label == 0][:30]
g_indices = [i for i, label in enumerate(dataset.labels) if label == 1][:30]
train_indices = plus_indices[:20] + g_indices[:20]
test_indices = plus_indices[20:] + g_indices[20:]

train_dataset = Subset(dataset, train_indices)
test_dataset = Subset(dataset, test_indices)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

train_dataset_plus = Subset(dataset, plus_indices[:20])
train_dataset_g = Subset(dataset, g_indices[:20])

train_loader_plus = DataLoader(train_dataset_plus, batch_size=8, shuffle=False)
train_loader_g = DataLoader(train_dataset_g, batch_size=8, shuffle=False)


# Define CNN models
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(64 * 36 * 36, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 36 * 36)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class CNNModel1(nn.Module):
    def __init__(self):
        super(CNNModel1, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = None
        self.fc2 = None

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = self.pool(torch.relu(self.conv2(x)))

        # Dynamic flattening
        x = x.view(x.size(0), -1)

        # Dynamically adjust fc1 size
        if not hasattr(self, "fc1") or self.fc1.in_features != x.size(1):
            self.fc1 = nn.Linear(x.size(1), 128).to(x.device)

        x = torch.relu(self.fc1(x))

        if not hasattr(self, "fc2"):
            self.fc2 = nn.Linear(128, 2).to(x.device)

        x = self.fc2(x)

        return x


class CNNModel2(nn.Module):
    def __init__(self):
        super(CNNModel2, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3)
        self.fc1 = None
        self.fc2 = None

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = self.pool(torch.relu(self.conv4(x)))

        # Dynamic flattening
        x = x.view(x.size(0), -1)

        # Dynamically adjust fc1 size
        if not hasattr(self, "fc1") or self.fc1.in_features != x.size(1):
            self.fc1 = nn.Linear(x.size(1), 128).to(x.device)

        x = torch.relu(self.fc1(x))

        if not hasattr(self, "fc2"):
            self.fc2 = nn.Linear(128, 2).to(x.device)

        x = self.fc2(x)

        return x


# Initialize models and optimizers
model1 = CNNModel()
model2 = CNNModel()
model3 = CNNModel1()
model4 = CNNModel2()

criterion = nn.CrossEntropyLoss()
optimizer1 = optim.Adam(model1.parameters(), lr=0.001)
optimizer2 = optim.Adam(model2.parameters(), lr=0.001)
optimizer3 = optim.Adam(model3.parameters(), lr=0.001)
optimizer4 = optim.Adam(model4.parameters(), lr=0.001)


# Training function
def train_model(modelv, optimizerv, loader, epochs=50):
    modelv.train()  # Set model to training mode
    for epoch in range(epochs):  # Loop over the dataset each epoch
        running_loss = 0.0  # Initialize running loss
        for inputs, labels in loader:  # Loop over the dataset
            optimizerv.zero_grad()  # Zero the parameter gradients
            outputs = modelv(inputs)  # Forward pass through the network to get predictions
            loss = criterion(outputs, labels)  # Compute loss based on predictions and true labels
            loss.backward()  # Backward pass to compute gradients
            optimizerv.step()  # Update network weights based on calculated gradients
            running_loss += loss.item()  # Accumulate loss over batches within an epoch
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(loader)}')


# Train the models
train_model(model1, optimizer1, train_loader)
train_model(model2, optimizer2, train_loader_plus)
train_model(model3, optimizer3, train_loader_g)
train_model(model4, optimizer4, train_loader)


# Testing function
def test_model(loader, modelv):
    modelv.eval()  # Set model to evaluation mode
    correct = total = 0  # Initialize counters for correct and total predictions
    with torch.no_grad():  # Disable gradient computation
        for inputs, labels in loader:  # Loop over the dataset
            outputs = modelv(inputs)  # Forward pass through the network to get predictions
            _, predicted = torch.max(outputs.data, 1)  # Get the class index with the highest probability
            total += labels.size(0)  # Increment total count by batch size
            correct += (predicted == labels).sum().item()  # Increment correct count by number of correct predictions
    print(f'Test Accuracy: {100 * correct / total:.2f}%')


# Test the models on the test dataset
test_model(test_loader, model1)
test_model(test_loader, model2)
test_model(test_loader, model3)
test_model(test_loader, model4)
