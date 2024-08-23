import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import os
import pandas as pd
import numpy as np
from customDatasets.test_custom_dataset import CustomImageDataset


#   TODO:: This needed to be added to the test file.
def add_predicted_labels_to_csv(predicted_labels, true_labels, lat_array, lng_array):
    test_data = {
        'predicted_labels': predicted_labels,
        'true_labels': true_labels,
        'lat': lat_array,
        'lng': lng_array
    }
    df = pd.DataFrame(test_data)

    output_file_path = './model_predictions.csv'
    df.to_csv(output_file_path, index=False)
    print("file saved")


def create_centers_array():
    df = pd.read_csv('big_dataset_labeled.csv')

    result_array = [None] * 120
    labels_visited = np.zeros(120)
    num_of_labels_visited = 0

    for _, row in df.iterrows():
        lat, lng = row['cluster_center']
        label = row['cluster_label']

        if labels_visited[label] == 0:
            result_array[label] = [lat, lng]
            labels_visited[label] += 1
            num_of_labels_visited += 1

        if num_of_labels_visited == 120:
            break

    return result_array


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
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False)

    # Using the ResNet-50 model
    model = torchvision.models.resnet50(weights=None)

    # Changing the last layer.
    num_classes = 120
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    model = nn.DataParallel(model)

    # Loading the model.
    model_dict_path = './classification_best_lr0.03_batch64.pth'
    model.load_state_dict(torch.load(model_dict_path))

    loss_func = nn.CrossEntropyLoss()

    running_test_loss = 0.0
    predicted_labels_list = []
    true_labels_list = []
    lat_list = []
    lng_list = []
    expected_values = []

    just_to_know = 1

    print("Calculating accuracy on test.")

    with torch.no_grad():
        for images, labels, lat, lng in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            probabilities = nn.functional.softmax(outputs, dim=1)
            _, predicted_labels = torch.max(probabilities, 1)

            predicted_labels_list.extend(predicted_labels.cpu().numpy())
            true_labels_list.extend(labels.cpu().numpy())
            lat_list.extend(lat)
            lng_list.extend(lng)

            # TODO:: remove.
            if just_to_know == 1:
                print(f"true_labels_list: {true_labels_list}, predicted_labels_list: {predicted_labels_list}")
                just_to_know = 0

            loss = loss_func(outputs, labels)
            running_test_loss += loss.item()

    test_loss = running_test_loss / len(test_loader)

    predicted_labels_array = np.array(predicted_labels_list)
    true_labels_array = np.array(true_labels_list)
    lat_array = np.array(lat_list)
    lng_array = np.array(lng_list)

    accuracy = np.mean(predicted_labels_array == true_labels_array)

    add_predicted_labels_to_csv(predicted_labels_array, true_labels_array, lat_array, lng_array)

    print(f'Finished test, Loss: {test_loss:.4f}')
    print(f'Accuracy: {accuracy}.')
