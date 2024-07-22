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

        # Data saves as: class_num.
        class_num = self.img_labels.iloc[idx, 0]
        img_label = torch.tensor(class_num, dtype=torch.long)

        return image, img_label


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Gpu is available: " + str(torch.cuda.is_available()))

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

dataset_path = './datasets/archive/dataset'
csv_file_path = '../coords_labels.csv'
dataset = CustomImageDataset(csv_file_path, dataset_path, transform)

batch_size = 64

train_size = int(0.6 * len(dataset))
val_size = int(0.2 * len(dataset))
test_size = len(dataset) - train_size - val_size

# Creating a seed.
generator = torch.Generator()
generator.manual_seed(387642706252)

# Splitting the data.
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size], generator = generator  )

train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                           shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size,
                                           shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size,
                                           shuffle=False)

# Using the ResNet-50 model
model = torchvision.models.resnet50(weights=None)

# Changing his last layer.
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 72)

model = nn.DataParallel(model)

# Loading the model.
model_dict_path = '../modle_weights/resnet50module_australia.pth'
model.load_state_dict(torch.load(model_dict_path))

loss_func = nn.CrossEntropyLoss()

running_test_loss = 0.0
predicted_labels_list = []
true_labels_list = []

print("Calculate accuracy on test.")

with torch.no_grad():
    images, labels = next(iter(test_loader))
    images = images.to(device)
    labels = labels.to(device)

    outputs = model(images)

    probabilities = nn.functional.softmax(outputs, dim=1)
    _, predicted_labels = torch.max(probabilities, 1)

    predicted_labels_list.extend(predicted_labels.cpu().numpy())
    true_labels_list.extend(labels.cpu().numpy())

    loss = loss_func(outputs, labels)
    running_test_loss += loss.item()

test_loss = running_test_loss / len(test_loader)

predicted_labels_array = np.array(predicted_labels_list)
true_labels_array = np.array(true_labels_list)

accuracy = np.mean(predicted_labels_array == true_labels_array)

print(predicted_labels_array)

print(f'Finished test, Loss: {test_loss:.4f}')
print(f'Accuracy: {accuracy}.')

print(true_labels_array)