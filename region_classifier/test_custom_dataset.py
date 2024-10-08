import torch
from torch.utils.data import Dataset
import os
import pandas as pd
from PIL import Image


# Handle the dataset.
class CustomImageDataset(Dataset):
    def __init__(self, city_csv_file, city_img_dir, big_csv_file, big_img_dir, transform=None, get_id=False):
        self.city_img_labels = pd.read_csv(city_csv_file)
        self.city_img_dir = city_img_dir
        self.big_csv_file = pd.read_csv(big_csv_file)
        self.big_img_dir = big_img_dir
        self.transform = transform
        self.get_id = get_id

    def __len__(self):
        total_length = len(self.city_img_labels) + len(self.big_csv_file)
        return total_length

    def __getitem__(self, idx):
        if idx >= len(self.city_img_labels):
            relative_idx = idx - len(self.city_img_labels)
            row = self.big_csv_file.iloc[relative_idx]
            image_id = row['id']
            class_num = row['cluster_label']
            lat = str(row['lat'])
            lng = str(row['lng'])

            img_path = os.path.join(self.big_img_dir, image_id)
            image = Image.open(img_path)

            # Activate transform.
            if self.transform:
                image = self.transform(image)

            img_label = torch.tensor(class_num, dtype=torch.long)
            if self.get_id:
                return image, img_label, lat, lng, idx
            return image, img_label, lat, lng

        # Images are saved as: cityId_placeID_year_month_northdeg_latitude_longitude_panoid.jpg
        row = self.city_img_labels.iloc[idx]
        city_name = row['city_id']
        place_id = str(row['place_id']).zfill(7)
        year = str(row['year']).zfill(4)
        month = str(row['month']).zfill(2)
        northdeg = str(row['northdeg']).zfill(3)
        lat = str(row['lat'])
        lon = str(row['lon'])
        panoid = str(row['panoid'])

        img_name = f"{city_name}_{place_id}_{year}_{month}_{northdeg}_{lat}_{lon}_{panoid}.jpg"
        img_path = os.path.join(self.city_img_dir, city_name, img_name)
        image = Image.open(img_path)

        # Activate transform.
        if self.transform:
            image = self.transform(image)

        # Data saves as: class_num.
        class_num = row['cluster_label']
        img_label = torch.tensor(class_num, dtype=torch.long)
        if self.get_id:
            return image, img_label, lat, lon, idx
        return image, img_label, lat, lon
