import os
import numpy as np
import json
import torch
from PIL import Image
from torch.utils.data import Dataset

from torchvision.io import read_image
from torchvision.ops.boxes import masks_to_boxes, box_convert
from torchvision import tv_tensors
from torchvision.tv_tensors import BoundingBoxes


class SignImageDataset(Dataset):

    def __init__(self, root=os.path.join("dataset", "annotations"), img_dir=os.path.join("images"),
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
        image_name = os.path.splitext(self.images[idx])[0] + ".jpg"

        img_path = os.path.join(self.img_dir, image_name)
        image = Image.open(img_path)

        width_ratio = height_ratio = 1
        # Activate transform.
        if self.transform:
            image = self.transform(image)
            width_ratio = image.shape[0] / image_data['width']
            height_ratio = image.shape[1] / image_data['height']

        num_objs = len(image_data['objects'])
        # Init labels
        labels = torch.ones((num_objs,), dtype=torch.int64)
        areas = torch.ones((num_objs,), dtype=torch.float64)
        iscrowd = torch.zeros((num_objs,), dtype=torch.int8)

        bboxes = torch.zeros((num_objs, 4), dtype=torch.float64)
        # Finish preparing the data
        for idx, obj in enumerate(image_data['objects']):
            labels[idx] = self.labels[obj['label']]
            bbox = obj['bbox']

            xmin = int(bbox["xmin"])
            ymin = int(bbox["ymin"])
            xmax = int(bbox["xmax"])
            ymax = int(bbox["ymax"])
            xmin = min(xmin, xmax)
            ymin = min(ymin, ymax)
            xmax = max(int(bbox["xmin"]), xmax)
            ymax = max(int(bbox["ymin"]), ymax)

            areas[idx] = (xmax - xmin) * (ymax - ymin)
            bboxes[idx] = torch.tensor(
                [width_ratio * xmin, height_ratio * ymin, width_ratio * xmax, height_ratio * ymax], dtype=torch.float64)

        bboxes = BoundingBoxes(bboxes, format='XYXY', canvas_size=image.shape)

        if num_objs == 0:
            bboxes = torch.empty((0, 4), dtype=torch.float32)

        target = {"boxes": bboxes, "labels": labels, "image_id": torch.tensor(idx), "area": areas, "iscrowd": iscrowd}

        json_file.close()

        return image, target
