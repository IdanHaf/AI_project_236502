import pandas as pd
import torch
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm

from Pipeline import Atlas
from moco import utils
from moco.test_custom_dataset import CustomImageDataset
from torchvision.transforms import transforms

transform = transforms.Compose([
    transforms.ToTensor()
])

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

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size],
                                                        generator=generator)
test_dataset, _ = random_split(test_dataset, [test_size // 2, test_size // 2],
                               generator=generator)
batch_size = 4

test_loader = DataLoader(test_dataset, batch_size=batch_size,
                         shuffle=False)

pipeline = Atlas("reg.pth", 'lang.pth', 'moco.pth', 'refine.pth', 15, 'baseset.csv')

errors = []
count = 0
for images, labels, lats, lngs, idxs in tqdm(test_loader):
    predictions = pipeline.predict_from_images(images)
    e = [utils.haversine(float(lat), float(lng), pred[0], pred[1]) for lat, lng, pred in zip(lats, lngs, predictions)]
    errors += e
    print(errors)
    if count == 2:
        exit(0)
    count += 1

df_errors = pd.DataFrame(errors)
df_errors.to_csv('test_errors.csv', index=False)
