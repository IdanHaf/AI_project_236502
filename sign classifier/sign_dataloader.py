import os
import numpy as np
import json
import torch
from torch.utils.data import Dataset

from torchvision.io import read_image
from torchvision.ops.boxes import masks_to_boxes, box_convert
from torchvision import tv_tensors
from torchvision.tv_tensors import BoundingBoxes


class SignImageDataset(Dataset):

    def __init__(self, root=os.path.join("dataset", "annotations"), img_dir=os.path.join("dataset", "images"),
                 transform=None):
        self.root = root
        self.img_dir = img_dir
        self.transform = transform

        # The list of the names of our images
        self.images = [line.strip() for line in open('annotated_images.txt', 'r').readlines()]
        self.labels = json.load(open("labels.json", 'r'))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        json_path = os.path.join(self.root, self.images[idx])
        json_file = open(json_path, 'r')
        image_data = json.load(json_file)
        img_path = os.path.join(self.img_dir, self.images[idx])
        image = read_image(img_path)

        # Activate transform.
        if self.transform:
            image = self.transform(image)
        image = tv_tensors.Image(image)

        num_objs = len(image_data['objects'])
        # Init labels
        labels = torch.ones((num_objs,), dtype=torch.int64)
        areas = torch.ones((num_objs,), dtype=torch.int64)
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        bboxes = []
        # Finish preparing the data
        for idx, obj in enumerate(image_data['objects']):
            labels[idx] = obj['label']
            bbox = obj['bbox']
            bboxes.append(bbox)

            areas[idx] = (bbox['xmax'] - bbox['xmin']) * (bbox['ymax'] - bbox['ymin'])
        bboxes = torch.tensor(bboxes)
        bboxes = BoundingBoxes(bboxes, format='XYXY', canvas_size=(image_data['height'], image_data['width']))

        target = {"boxes": bboxes, "labels": labels, "image_id": idx, "area": areas, "iscrowd": iscrowd}

        json_file.close()

        return image, target
