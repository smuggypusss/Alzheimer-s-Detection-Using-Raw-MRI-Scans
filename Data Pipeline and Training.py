import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import nibabel as nib
import numpy as np
import pandas as pd
import scipy.ndimage
import torch.nn.functional as F


# Custom Dataset Class for Loading MRI Images
class MRIDataset3D(Dataset):
    def __init__(self, df):
        self.data_frame = df

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_path = str(self.data_frame.iloc[idx, 1])
        print(f"Loading image from: {img_path}")  # Debugging print1
        img = nib.load(img_path).get_fdata()

        if img.ndim == 4 and img.shape[3] == 1:  # (D, H, W, C) where C=1
            img = img.squeeze(-1)  # Remove the channel dimension

        
        elif img.ndim == 3:  # (D, H, W)
            pass  # No action needed; shape is already correct
        else:
            raise ValueError(f"Unexpected image dimensions: {img.shape}")

        
        img = self.resize_image(np.expand_dims(img, axis=0), new_size=(64, 64, 64))
        img = img.astype(np.float32)  # PyTorch should have this type

       
        label = 1 if self.data_frame.iloc[idx, 2] == 'Demented' else 0 #binary classification between demented and nondemented 
        return img, torch.tensor(label, dtype=torch.float32)

    def resize_image(self, img, new_size):
        """ Resize the image to the new size. """
        if img.ndim == 5:
            img = img.squeeze(0)  # Remove batch dimension if exists
        if img.ndim != 4:
            raise ValueError(f"Expected 4D input but got {img.ndim}D input.")

        # Prepare an empty array for the resized image
        img_resized = np.zeros((img.shape[0], *new_size))
        for i in range(img.shape[0]):  # Loop over each channel
            # Calculate zoom factors for each dimension
            zoom_factors = [
                new_size[0] / img.shape[1],  # D
                new_size[1] / img.shape[2],  # H
                new_size[2] / img.shape[3]  # W
            ]
            img_resized[i] = scipy.ndimage.zoom(img[i], zoom_factors, order=1)  # Resize with interpolation

        return img_resized


# CNN Model Definition
class CNN3D(nn.Module):
    def __init__(self):
        super(CNN3D, self).__init__()
        self.conv1 = nn.Conv3d(1, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(16)
        self.bn2 = nn.BatchNorm3d(32)
        self.bn3 = nn.BatchNorm3d(64)

        self.fc1 = nn.Linear(32 * 16 * 16 * 16, 128)  # Adjust this dimension based on output from conv layers
        self.fc2 = nn.Linear(128, 1)  # Output a single logit for binary classification

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 16 * 16 * 16)  # Flattening the output
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # Single logit output
        return x


# Training Function
def train_model(model, train_loader, criterion, optimizer, num_epochs):
    model.train()  # Set the model to training mode
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            print(f"Processing batch {batch_idx + 1}/{len(train_loader)}")  # Progress indicator
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # Calculate loss
            loss = criterion(outputs.squeeze(), labels)  # Squeeze to match label shape
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights

            # Calculate accuracy
            predicted = torch.sigmoid(outputs).round()  # Convert logit to binary
            correct += (predicted.squeeze() == labels).sum().item()
            total += labels.size(0)

            epoch_loss += loss.item()

        # Print epoch statistics
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {100 * correct / total:.2f}%")


# Main Function
def main():
    # Load the CSV
    df = pd.read_csv('labelled.csv')

    # Create dataset and DataLoader
    train_dataset = MRIDataset3D(df)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    # Instantiate your model, loss function, and optimizer
    model = CNN3D()  # Initialize the model
    criterion = nn.BCEWithLogitsLoss()  # Binary Cross Entropy with logits
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # Start training the model
    train_model(model, train_loader, criterion, optimizer, num_epochs=4)


# Add the multiprocessing protection
if __name__ == '__main__':
    main()
