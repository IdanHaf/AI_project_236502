import torch
from torch.utils.data import Dataset
import os
import pandas as pd
from PIL import Image


# Handle the dataset.
class CustomImageDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.img_labels = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        # Images are saved as: cityId_placeID_year_month_northdeg_latitude_longitude_panoid.jpg
        row = self.img_labels.iloc[idx]
        city_name = row['city_id']
        place_id = str(row['place_id']).zfill(7)
        year = str(row['year']).zfill(4)
        month = str(row['month']).zfill(2)
        northdeg = str(row['northdeg']).zfill(3)
        lat = str(row['lat'])
        lon = str(row['lon'])
        panoid = str(row['panoid'])

        img_name = f"{city_name}_{place_id}_{year}_{month}_{northdeg}_{lat}_{lon}_{panoid}.jpg"
        img_path = os.path.join(self.img_dir, city_name, img_name)
        image = Image.open(img_path)

        # Activate transform.
        if self.transform:
            image = self.transform(image)

        # Data saves as: class_num.
        class_num = row['cluster_label']
        img_label = torch.tensor(class_num, dtype=torch.long)

        return image, img_label
