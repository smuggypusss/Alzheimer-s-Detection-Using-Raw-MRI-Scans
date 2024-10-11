import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from efficientnet_pytorch import EfficientNet
import nibabel as nib
import numpy as np
import pandas as pd
import random
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Dataset class to load 2D slices from 3D MRI images
class MRIDataset(Dataset):
    def __init__(self, df, transform=None):
        self.data_frame = df
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_path = self.data_frame.iloc[idx, 1]
        print(f"Loading image from: {img_path}")
        img = nib.load(img_path).get_fdata()

        if img.ndim == 4 and img.shape[3] == 1:  # (D, H, W, C) where C=1
            img = img.squeeze(-1)  # Remove the channel dimension

        elif img.ndim == 3:  # (D, H, W)
            pass  # Already in the correct shape
        else:
            raise ValueError(f"Unexpected image dimensions: {img.shape}")

        # Extract the middle slice
        slice_idx = img.shape[2] // 2
        img_slice = img[:, :, slice_idx]

        # Normalize the image slice
        img_slice = (img_slice - np.min(img_slice)) / (np.max(img_slice) - np.min(img_slice))

        # Convert to 3 channels (RGB-like format)
        img_slice = np.stack([img_slice] * 3, axis=-1)  # Convert 1-channel to 3-channel

        # Convert to tensor and ensure it's float32
        img_slice = self.transform(img_slice).float()  # Ensure float32 type here

        # Get the label
        label = 1 if self.data_frame.iloc[idx, 2] == 'Demented' else 0
        return img_slice, torch.tensor(label, dtype=torch.float32)


# Define model with EfficientNet
class AlzheimerEfficientNet(nn.Module):
    def __init__(self):
        super(AlzheimerEfficientNet, self).__init__()
        # Load EfficientNet
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b0')
        self.efficientnet._fc = nn.Linear(self.efficientnet._fc.in_features, 1)  # Output a single logit

    def forward(self, x):
        return self.efficientnet(x)

# Training function
def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Calculate accuracy
            predicted = torch.sigmoid(outputs).round()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            epoch_loss += loss.item()

        accuracy = 100 * correct / total
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%')

# Load CSV and create dataset
df = pd.read_csv('labelled.csv')
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Adjust as needed
])

# Loaders
train_dataset = MRIDataset(df, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# Model, criterion, and optimizer
model = AlzheimerEfficientNet().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

# Train the model
train_model(model, train_loader, criterion, optimizer, num_epochs=10)
