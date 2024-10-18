import torch
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import transforms
from classifier import Classifier

from test_custom_dataset import CustomImageDataset

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

batch_size = 64

city_dataset_path = './Images'
city_csv_file_path = './city_images_dataset.csv'
big_dataset_path = './big_dataset'
big_csv_file_path = './big_dataset_labeled.csv'
dataset = CustomImageDataset(city_csv_file_path, city_dataset_path, big_csv_file_path, big_dataset_path, transform,
                             True)
generator = torch.Generator()
generator.manual_seed(387642706252)

train_size = int(0.6 * len(dataset))
val_size = int(0.2 * len(dataset))
test_size = len(dataset) - train_size - val_size

# Splitting the data.
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size], generator=generator)

train_loader = DataLoader(train_dataset, batch_size=batch_size,
                          shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size,
                        shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size,
                         shuffle=False)

classifier = Classifier(transform=transform)
classifier.export(train_loader, 'train.csv')
classifier.export(val_loader, 'val.csv')
classifier.export(test_loader, 'test.csv')
