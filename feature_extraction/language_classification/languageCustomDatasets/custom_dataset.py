import cv2
from torchvision import datasets, transforms
from PIL import Image
import numpy as np


def create_grid(img):
    """
        Scaling the size of each image without loosing data.

        :param img: The PIL image.
        :returns: new grid image with scaled size.
    """
    # Convert img to numpy.
    img = np.array(img)

    cols = round(448 / img.shape[1])
    rows = round(448 / img.shape[0])

    if rows == 0:
        rows = 1
    if cols == 0:
        cols = 1

    row_stack = np.hstack([img] * cols)
    grid_image = np.vstack([row_stack] * rows)

    # Converting back to PIL.
    grid_image = Image.fromarray(grid_image)

    return grid_image


class CustomImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root, transform)

    def __getitem__(self, index):
        path, target = self.samples[index]
        img = self.loader(path)
        img = create_grid(img)

        if self.transform:
            img = self.transform(img)

        return img, target
