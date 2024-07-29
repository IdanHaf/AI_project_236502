import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import os
import pandas as pd
import numpy as np

# loading bar
from tqdm import tqdm

# Defining the loss classes:

# geodesic-logarithmic (geo-logic) loss
class GeoLogLoss(nn.Module):
    def __init__(self):
        super(GeoLogLoss, self).__init__()

    def forward(self, output, target):
        olon, olat = output[:, 0], output[:, 1]
        tlon, tlat = target[:, 0], target[:, 1]

        olon = torch.deg2rad(olon)
        olat = torch.deg2rad(olat)
        tlon = torch.deg2rad(tlon)
        tlat = torch.deg2rad(tlat)

        delta_lon = olon - tlon
        delta_lat = olat - tlat

        a = torch.sin(delta_lat / 2) ** 2 + torch.cos(olat) * torch.cos(tlat) * torch.sin(delta_lon / 2) ** 2
        c = 2 * torch.asin(torch.sqrt(a))

        radius_earth = 6371.0
        res = c * radius_earth

        res = torch.log10(res)

        return torch.mean(res)

# geodesic loss
class GeoLoss(nn.Module):
    def __init__(self):
        super(GeoLoss, self).__init__()

    def forward(self, output, target):
        olon, olat = output[:, 0], output[:, 1]
        tlon, tlat = target[:, 0], target[:, 1]

        olon = torch.deg2rad(olon)
        olat = torch.deg2rad(olat)
        tlon = torch.deg2rad(tlon)
        tlat = torch.deg2rad(tlat)

        delta_lon = olon - tlon
        delta_lat = olat - tlat

        a = torch.sin(delta_lat / 2) ** 2 + torch.cos(olat) * torch.cos(tlat) * torch.sin(delta_lon / 2) ** 2
        c = 2 * torch.asin(torch.sqrt(a))

        radius_earth = 6371.0
        res = c * radius_earth

        return torch.mean(res)

# Log of MSE loss
class MSELogLoss(nn.Module):
    def __init__(self):
        super(MSELogLoss, self).__init__()

    def forward(self, output, target):
        mse = torch.mean((output - target) ** 2)
        log_mse = torch.log10(mse)
        return log_mse

# Handle the dataset.
class CustomImageDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.img_labels = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_name = str(idx) + ".png"
        image_path = os.path.join(self.img_dir, img_name)
        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)

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

daniel_dataset_path = '../../../../../Datasets/10K_Dataset'
daniel_csv_file_path = '../../../../../Datasets/10K_Dataset/coords.csv'
dataset = CustomImageDataset(daniel_csv_file_path, daniel_dataset_path, transform)  # Daniel Dataset loading

train_size = int(0.6 * len(dataset))
val_size = int(0.2 * len(dataset))
test_size = len(dataset) - train_size - val_size

generator = torch.Generator()
generator.manual_seed(387642706252)

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size], generator=generator)

train_loader = DataLoader(train_dataset, batch_size=batch_size,
                          shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size,
                        shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size,
                         shuffle=False)

# Using the pre-trained ResNet-50 model

best_lr = None
best_loss_func = None
best_geo_dist = float('inf')
lrs = np.logspace(-1, -6, 6)
losses = [GeoLogLoss(), GeoLoss(), MSELogLoss(), nn.MSELoss()]

# Folder for all loss files
losses_save_dir = './losses_rand'
if not os.path.exists(losses_save_dir):
    os.makedirs(losses_save_dir)

# Directory for saving model weights
weights_save_dir = './model_checkpoints_rand'
if not os.path.exists(weights_save_dir):
    os.makedirs(weights_save_dir)

geo_loss_func = GeoLoss()

for lr_index, lr in enumerate(lrs):
    if lr_index == 0: # remove when finished
        continue
    for loss_index, loss in enumerate(losses):
        model = torchvision.models.resnet50(weights=None)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 2)

        model = nn.DataParallel(model)
        model = model.to(device)

        loss_func = loss
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # Early stopping
        patience = 5
        best_model_val_loss = float('inf')
        epochs_no_improve = 0 

        validation_losses = []

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            train_bar = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs}")

            for inputs, labels in train_bar:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss_value = loss_func(outputs, labels)
                loss_value.backward()
                optimizer.step()

                running_loss += loss_value.item()
                train_bar.set_postfix(loss=running_loss / len(train_bar))

            avg_train_loss = running_loss / len(train_loader)

            # Validation
            model.eval()
            running_val_loss = 0.0
            running_geo_dist = 0.0

            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    val_outputs = model(inputs)
                    val_loss = loss_func(val_outputs, labels)
                    geo_dist = geo_loss_func(val_outputs, labels)
                    running_val_loss += val_loss.item()
                    running_geo_dist += geo_dist.item()

            avg_val_loss = running_val_loss / len(val_loader)
            avg_geo_dist = running_geo_dist / len(val_loader)
            validation_losses.append(avg_geo_dist)

            print(f'Epoch {epoch + 1}/{num_epochs}, avg training loss RMSE: {np.sqrt(avg_train_loss):.4f}, '
                  f'avg validation loss RMSE: {np.sqrt(avg_val_loss):.4f}')

            # Early stopping check
            if avg_geo_dist < best_model_val_loss:
                best_model_val_loss = avg_geo_dist
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print(f'Early stopping at epoch {epoch + 1}')
                break

        # Save the model weights in separate directories
        model_combination_dir = os.path.join(weights_save_dir, f'lr{lr}_loss{loss}')
        if not os.path.exists(model_combination_dir):
            os.makedirs(model_combination_dir)

        model_save_path = os.path.join(model_combination_dir, 'model.pth')
        torch.save(model.state_dict(), model_save_path)

        # Save validation losses with names as lr<x>loss<y>
        loss_file_name = f'lr{lr_index}loss{loss_index}_rand.npy'
        val_loss_log_path = os.path.join(losses_save_dir, loss_file_name)
        np.save(val_loss_log_path, validation_losses)

        # Update the best combination if the current one is better
        if best_model_val_loss < best_geo_dist:
            best_geo_dist = best_model_val_loss
            best_lr = lr
            best_loss_func = loss_func

# Test best combo on test_loader
best_model = torchvision.models.resnet50(weights=None)
num_features = best_model.fc.in_features
best_model.fc = nn.Linear(num_features, 2)

best_model = nn.DataParallel(best_model)
best_model = best_model.to(device)

best_model_path = os.path.join(weights_save_dir, f'lr{lrs.tolist().index(best_lr)}_loss{losses.index(best_loss_func)}', 'model.pth')
best_model.load_state_dict(torch.load(best_model_path))

# Evaluate the best model on the test set
best_model.eval()
running_test_loss = 0.0

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        test_outputs = best_model(inputs)
        test_loss = best_loss_func(test_outputs, labels)
        running_test_loss += test_loss.item()

avg_test_loss = running_test_loss / len(test_loader)
print(f'Average test loss RMSE: {np.sqrt(avg_test_loss):.4f}')
