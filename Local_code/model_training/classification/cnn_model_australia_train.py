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


# Handle the dataset.
class CustomImageDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):

        self.img_labels = pd.read_csv(csv_file, header=None)

        ausralia_cords_lat = [-14.248949, -38.675074]
        ausralia_cords_lng = [114.182947, 152.420742]

        filtered_df = self.img_labels[(self.img_labels[0] <= ausralia_cords_lat[0]) &
                                 (self.img_labels[0] >= ausralia_cords_lat[1]) &
                                 (self.img_labels[1] >= ausralia_cords_lng[0]) &
                                 (self.img_labels[1] <= ausralia_cords_lng[1])]

        result_df = pd.DataFrame(0, index=self.img_labels.index, columns=[0])
        result_df.loc[filtered_df.index, 0] = 1
        self.img_labels = result_df

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
        is_australia = self.img_labels.iloc[idx, 0]
        img_labels = torch.tensor([is_australia], dtype=torch.float32)

        return image, img_labels


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Gpu is available: " + str(torch.cuda.is_available()))

num_epochs = 10
batch_size = 64
learning_rate = 0.1

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

dataset_path = './datasets/archive/dataset'
csv_file_path = './datasets/archive/dataset/coords.csv'
dataset = CustomImageDataset(csv_file_path, dataset_path, transform)  # Idan Dataset loading

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
model = torchvision.models.resnet50(weights='IMAGENET1K_V1')
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 1)

model = nn.DataParallel(model)
model = model.to(device)

loss_func = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop.
for epoch in range(num_epochs):
    train_bar = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs}")

    running_loss = 0.0
    model.train()
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward propogation
        outputs = model(inputs)
        loss = loss_func(outputs, labels)

        # Backward propogation
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        train_bar.set_postfix(loss=running_loss / len(train_bar))

    # Print the loss for every epoch
    avg_train_loss = running_loss / len(train_loader)

    # Validation on val_loader.
    model.eval()
    running_val_loss = 0.0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            val_outputs = model(inputs)
            val_loss = loss_func(val_outputs, labels)
            running_val_loss += val_loss.item()

    avg_val_loss = running_val_loss / len(val_loader)

    print(f'Epoch {epoch + 1}/{num_epochs}, avg training loss: {avg_train_loss:.4f}, '
          f'avg validation loss: {avg_val_loss:.4f}')

# Test on test_loader.
model.eval()
running_test_loss = 0.0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = loss_func(outputs, labels.float())
        running_test_loss += loss.item()

test_loss = running_test_loss / len(test_loader)

print(f'Finished test, Loss: {test_loss:.4f}')

torch.save(model.state_dict(), 'resnet50module_australia.pth')

print("Module saved")


with torch.no_grad():
    images, labels = next(iter(test_loader))

    outputs = model(images)

    probabilities = torch.sigmoid(outputs)
    predicted_labels = (probabilities > 0.5).int()

    print(f'Image label: {labels}')
    print(f'model output: {predicted_labels}')

