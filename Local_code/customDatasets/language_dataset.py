import torch
from torch.utils.data import Dataset
import os
import pandas as pd
from PIL import Image


# Handle the dataset.
class CustomLanguageDataset(Dataset):
    def __init__(self, language_img_dir, transform=None):
        self.language_img_dir = language_img_dir
        self.transform = transform

    def __len__(self):
        # TODO:: add real length by accessing folder.
        total_length = 9970
        return total_length

    def __getitem__(self, idx):
        img_name = f"{idx}.png"
        img_path = os.path.join(self.city_img_dir, img_name)
        image = Image.open(img_path)

        # Activate transform.
        if self.transform:
            image = self.transform(image)

        img_label = torch.tensor(class_num, dtype=torch.long)

        # Activate transform.
        if self.transform:
            image = self.transform(image)

        img_label = torch.tensor(class_num, dtype=torch.long)

        return image, img_label
