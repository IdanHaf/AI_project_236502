import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
from languageCustomDatasets.custom_dataset import CustomImageFolder
from language_model import LanguageModel
from PIL import Image
from tqdm import tqdm


def test_on_dataset(lang_model, transform, dataset_path):
    """
        results:
        Label Recall: [0.935, 0.9, 0.72653061, 0.81773399, 0.92771084, 0.89711934,
                       0.95512821, 0.84466019, 0.95045045]
    """
    test_dataset = CustomImageFolder(root=dataset_path, transform=transform)
    print(f"labels: {test_dataset.classes}")

    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    lang_model.test_model(test_loader)


def test_on_single_image(lang_model):
    """
        results:
        [0.         0.         0.13441712 0.         0.         0.60172933
        0.         0.2638536  0.        ]
    """
    #   Change with actual image path.
    img = Image.open('./1362657911122507.jpg')
    width, height = img.size
    # Crop the bottom - possible timeline part
    img = img.crop((0, 0, width, height - 35))
    probs = lang_model.detect_language(img)
    print(probs)


def export_model(lang_model, dataset_path):
    dataset = datasets.ImageFolder(root=dataset_path, transform=None)

    print(f"labels: {dataset.classes}")

    lang_model.export(dataset, "prob_dataset_language.csv", dataset.classes)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Gpu is available: " + str(torch.cuda.is_available()))

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    model_dict_path = './b3_language_best_lr0.002_batch32.pth'

    lang_model = LanguageModel(model_dict_path, transform)
    export_model(lang_model, "../prob_dataset")
