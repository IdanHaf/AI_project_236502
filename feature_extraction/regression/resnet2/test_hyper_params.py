import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import os
import pandas as pd
import numpy as np

# loading bar
from tqdm import tqdm

# geodesic loss
class GeoLoss(nn.Module):
    def __init__(self):
        super(GeoLoss, self).__init__()

    def forward(self, output, target):
        # Assuming output and target are tensors with shape (batch_size, 2)
        olon, olat = output[:, 0], output[:, 1]
        tlon, tlat = target[:, 0], target[:, 1]

        # Convert degrees to radians
        olon = torch.deg2rad(olon)
        olat = torch.deg2rad(olat)
        tlon = torch.deg2rad(tlon)
        tlat = torch.deg2rad(tlat)

        # Calculate deltas
        delta_lon = olon - tlon
        delta_lat = olat - tlat

        # Haversine formula
        a = torch.sin(delta_lat / 2) ** 2 + torch.cos(olat) * torch.cos(tlat) * torch.sin(delta_lon / 2) ** 2
        c = 2 * torch.asin(torch.sqrt(a))

        # Radius of the Earth in kilometers
        radius_earth = 6371.0
        res = c * radius_earth

        # Return the mean loss over the batch
        return torch.mean(res)

# Handle the dataset.
class CustomImageDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.img_labels = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        # Images are saved as: idx.png
        img_name = str(idx) + ".png"
        image_path = os.path.join(self.img_dir, img_name)
        image = Image.open(image_path)

        # Activate transform.
        if self.transform:
            image = self.transform(image)

        # Data saves as: lat, lng.
        lat = self.img_labels.iloc[idx, 0]
        lng = self.img_labels.iloc[idx, 1]
        img_labels = torch.tensor([lat, lng], dtype=torch.float32)

        return image, img_labels


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Gpu is available: " + str(torch.cuda.is_available()))

num_epochs = 20
batch_size = 64

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

idan_dataset_path = './datasets/archive/dataset'
idan_csv_file_path = './datasets/archive/dataset/coords.csv'
# dataset = CustomImageDataset(idan_csv_file_path, idan_dataset_path, transform)  # Idan Dataset loading

daniel_dataset_path = '../../../../Datasets/10K_Dataset'
daniel_csv_file_path = '../../../../Datasets/10K_Dataset/coords.csv'
dataset = CustomImageDataset(daniel_csv_file_path, daniel_dataset_path, transform)  # Daniel Dataset loading

train_size = int(0.6 * len(dataset))
val_size = int(0.2 * len(dataset))
test_size = len(dataset) - train_size - val_size

generator = torch.Generator()
generator.manual_seed(387642706252)

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size], generator=generator)

# Merging train and validation datasets
merged_dataset = ConcatDataset([train_dataset, val_dataset])

train_val_loader = DataLoader(merged_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size,
                         shuffle=False)

# Using the pre-trained ResNet-50 model
lr = 10**-3
loss_func = GeoLoss()

model = torchvision.models.resnet50(weights=None)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)

model = nn.DataParallel(model)
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=lr)

# List to store loss values
epoch_losses = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    train_bar = tqdm(train_val_loader, total=len(train_val_loader), desc=f"Epoch {epoch + 1}/{num_epochs}")

    for inputs, labels in train_bar:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss_value = loss_func(outputs, labels)
        loss_value.backward()
        optimizer.step()

        running_loss += loss_value.item()
        train_bar.set_postfix(loss=running_loss / len(train_bar))

    avg_train_loss = running_loss / len(train_val_loader)
    epoch_losses.append(avg_train_loss)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_train_loss:.4f}")

# Save the model weights after training
torch.save(model.state_dict(), './10K_dataset_best_model_weights_random.pth')

# Save the loss values to a .npy file
np.save('epoch_losses_rand.npy', np.array(epoch_losses))
