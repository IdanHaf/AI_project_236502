import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import os
import math
import pandas as pd
import numpy as np
from customDatasets.test_custom_dataset import CustomImageDataset
from test_classification_models import TestModels


#   TODO:: This needed to be added to the test file.
def add_predicted_labels_to_csv(predicted_labels_arr, true_labels, lat_arr, lng_arr, probabilities):
    test_data = {
        'predicted_labels': predicted_labels_arr,
        'true_labels': true_labels,
        'lat': lat_arr,
        'lng': lng_arr,
    }
    df = pd.DataFrame(test_data)

    for i in range(120):
        df[f'Probability_Class_{i}'] = probabilities[:, i]

    output_file_path = './model_lr0.0005_predictions_with_prob.csv'
    df.to_csv(output_file_path, index=False)
    print("file saved")


def plot_label_histogram(data_loader, title):
    labels = []
    for _, img_label, _, _ in data_loader:
        labels.extend(img_label.tolist())

    plt.hist(labels, bins=120, range=(0, 119), alpha=0.7, color='blue')
    plt.title(title)
    plt.xlabel('Label')
    plt.ylabel('number of images')

    plt.savefig(f"{title}.png")
    plt.close()


if __name__ == "__main__":
    print("start testing:")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Gpu is available: " + str(torch.cuda.is_available()))

    batch_size = 64

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    city_dataset_path = './Images'
    city_csv_file_path = './city_images_dataset.csv'
    big_dataset_path = './results'
    big_csv_file_path = './big_dataset_labeled.csv'
    dataset = CustomImageDataset(city_csv_file_path, city_dataset_path, big_csv_file_path, big_dataset_path,
                                 transform)  # Idan Dataset loading

    print(f"Number of samples in the dataset: {len(dataset)}")

    train_size = int(0.6 * len(dataset))
    val_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    # Creating a seed.
    generator = torch.Generator()
    generator.manual_seed(387642706252)

    # Splitting the data.
    _, _, test_dataset = random_split(dataset, [train_size, val_size, test_size],
                                      generator=generator)

    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False)

    #   Plots histograms of our dataset split.
    # plot_label_histogram(train_loader, 'Train dataset label distribution')
    # plot_label_histogram(val_loader, 'Validation dataset label distribution')
    # plot_label_histogram(test_loader, 'Test dataset label distribution')

    model_dict_path = './classification_best_lr0.0005_batch64.pth'

    model = torchvision.models.resnet50(weights=None)

    # Changing the last layer.
    num_classes = 120
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    model = nn.DataParallel(model)

    # Loading the model.
    model.load_state_dict(torch.load(model_dict_path))

    model_tester = TestModels(batch_size, transform, model)
    model_tester.test_model(test_loader, device, add_predicted_labels_to_csv)
