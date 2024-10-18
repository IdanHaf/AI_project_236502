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

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

daniel_dataset_path = '../../../../Datasets/10K_Dataset'
daniel_csv_file_path = '../../../../Datasets/10K_Dataset/coords.csv'
dataset = CustomImageDataset(daniel_csv_file_path, daniel_dataset_path, transform)  # Daniel Dataset loading

batch_size = 64

train_size = int(0.6 * len(dataset))
val_size = int(0.2 * len(dataset))
test_size = len(dataset) - train_size - val_size

generator = torch.Generator()
generator.manual_seed(387642706252)

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size], generator=generator)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Gpu is available: " + str(torch.cuda.is_available()))

model = torchvision.models.resnet50(weights=None)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)

model = nn.DataParallel(model)  # Add DataParallel here
model.load_state_dict(torch.load('./10K_dataset_best_model_weights_random.pth'))

model.to(device)
model.eval()

all_targets = []
all_outputs = []

with torch.no_grad():
    for images, targets in test_loader:
        images = images.to(device)
        targets = targets.to(device)
        
        outputs = model(images)
        
        all_targets.append(targets.cpu())
        all_outputs.append(outputs.cpu())

all_targets = torch.cat(all_targets, dim=0).numpy()
all_outputs = torch.cat(all_outputs, dim=0).numpy()

# Save to .npy files
np.save('targets_rand.npy', all_targets)
np.save('outputs_rand.npy', all_outputs)

print("Targets and outputs saved to targets.npy and outputs.npy")
