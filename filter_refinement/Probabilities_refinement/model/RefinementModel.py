import numpy as np
import math
import pandas as pd
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import copy
from tqdm import tqdm


class RefinementModel(nn.Module):
    def __init__(self):
        super().__init__()
        # input size = 120 (1st model) + 9 (language model)
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(129, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 120),
        )

    def forward(self, x):
        x = self.linear_relu_stack(x)
        return x


def validation_test_loop(data_loader, model, device, criterion):
    """
    Validate or test model on the dataloader.
    :param data_loader: validation or test dataloader.
    :param model: model to train.
    :param device: device to use.
    :param criterion: loss function to use.

    :return: validation or test loss and accuracy, recall for each class.
    """
    model.eval()
    running_val_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    # To avoid dividing by 0.
    classes_correct = np.ones(120)
    classes_counter = np.ones(120)

    with torch.no_grad():
        for inputs, labels, _ in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            val_outputs = model(inputs)
            val_loss = criterion(val_outputs, labels)
            running_val_loss += val_loss.item()

            probabilities = nn.functional.softmax(val_outputs, dim=1)
            _, predicted_labels = torch.max(probabilities, 1)

            correct_predictions += (predicted_labels == labels).sum().item()
            total_predictions += labels.size(0)

            # Calc recall.
            labels_np = labels.cpu().numpy()
            predicted_labels_np = predicted_labels.cpu().numpy()

            for label, prediction in zip(labels_np, predicted_labels_np):
                classes_counter[label] += 1
                if label == prediction:
                    classes_correct[label] += 1

    avg_val_loss = running_val_loss / len(data_loader)
    val_accuracy = correct_predictions / total_predictions
    classes_recall = classes_correct / classes_counter
    classes_recall = np.where(classes_recall == 1, 0, classes_recall)

    print(f'avg validation loss: {avg_val_loss:.4f}, validation accuracy: {val_accuracy:.4f}')
    return avg_val_loss, val_accuracy, classes_recall


def train_model(dataloaders, model, device, criterion, optimizer, num_epochs, model_name, patience=10):
    """
    Train model.
    :param dataloaders: tuple of train and validation dataloaders.
    :param model: model to train.
    :param device: device to use.
    :param criterion: loss function to use.
    :param optimizer: optimizer to use.
    :param num_epochs: number of epochs.
    :param model_name: name of the saved model.
    :param patience: how many epochs to wait without improvement in validation loss.

    :return: training loss over epochs, validation loss, accuracy over epochs
    """
    train_loader, val_loader = dataloaders
    train_losses = []
    validation_loss_results = []
    validation_accuracy_results = []
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = patience
    best_model_weights = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        running_loss = 0.0
        model.train()

        for inputs, labels, _ in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        print(f'Epoch {epoch + 1}/{num_epochs}, avg training loss: {avg_train_loss:.4f} ')
        epoch_val_loss, epoch_val_accuracy, _ = validation_test_loop(val_loader, model, device, criterion)
        validation_loss_results.append(epoch_val_loss)
        validation_accuracy_results.append(epoch_val_accuracy)

        # Early stopping according to val loss.
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            patience_counter = patience
        else:
            patience_counter -= 1
            if patience_counter == 0:
                break

    torch.save(best_model_weights, f"{model_name}.pth")
    print(f"Module saved, best epoch: {best_epoch}")

    return train_losses, validation_loss_results, validation_accuracy_results


def expected_val(df_clusters, prob_vector, lat):
    """
    Calculate the expected value of giving prob vector on clusters values.
    :param df_clusters: The df clusters contain the clusters coordinates.
    :param prob_vector: The probability vector.
    :param lat: True if calc lat, False for lng.
    :return: expected value of giving prob vector on clusters values.
    """
    res = 0
    idx = 0 if lat else 1

    for i in range(len(prob_vector)):
        condition = df_clusters['cluster_label'] == i
        c = df_clusters[condition].cluster_center.to_numpy()[0]
        res += prob_vector[i] * c[idx]
    return res


def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points
    :param lat1: The latitude of the first point.
    :param lon1: The longitude of the first point.
    :param lat2: The latitude of the second point.
    :param lon2: The longitude of the second point.
    :return: The great circle distance.
    """
    R = 6371.0

    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = (math.sin(dlat / 2) ** 2 +
         math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance = R * c
    return distance


def Q_graph(arr, x_title, y_title, title, labels, max_perc=100):
    for element, label in zip(arr, labels):
        sorted_arr = np.sort(element)
        percentiles = np.arange(1, len(element) + 1) / len(element) * 100
        if max_perc != 100:
            limit = percentiles <= max_perc
            percentiles = percentiles[limit]
            sorted_arr = sorted_arr[limit]

        plt.plot(percentiles, sorted_arr, label=label)

    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{title}.png")
    plt.show()


def plot_test_results(data_loader, model, device, df_clusters_center):
    """
    Train model.
    :param data_loader: validation or test dataloader.
    :param model: model to train.
    :param device: device to use.
    :param df_clusters_center: df of clusters center lat and lng coordinates.

    :return: validation or test loss and accuracy, recall for each class.
    """
    model.eval()
    before_dist = np.array([])
    dist = np.array([])

    with torch.no_grad():
        for inputs, labels, coords in tqdm(data_loader, desc="Batch"):
            before_lat_values = np.array([expected_val(df_clusters_center, prediction[:120], True)
                                          for prediction in inputs])
            before_lng_values = np.array([expected_val(df_clusters_center, prediction[:120], False)
                                          for prediction in inputs])

            before_batch_dist = np.array([haversine(e_lat, before_lng_values[idx], coords[idx][0], coords[idx][1])
                                          for idx, e_lat in enumerate(before_lat_values)])

            before_dist = np.concatenate((before_dist, before_batch_dist))

            inputs = inputs.to(device)
            val_outputs = model(inputs)
            probabilities = nn.functional.softmax(val_outputs, dim=1)
            probabilities_np = probabilities.cpu().numpy()

            batch_lat_values = np.array([expected_val(df_clusters_center, prediction, True)
                                         for prediction in probabilities_np])
            batch_lng_values = np.array([expected_val(df_clusters_center, prediction, False)
                                         for prediction in probabilities_np])

            batch_dist = np.array([haversine(e_lat, batch_lng_values[idx], coords[idx][0], coords[idx][1])
                                   for idx, e_lat in enumerate(batch_lat_values)])

            dist = np.concatenate((dist, batch_dist))

    Q_graph([before_dist, dist], 'percentage of the dataset', 'distance in KM',
            'Quantile_graph_of_predicted_error_after_expected_value',
            labels=["Cnn model", "After refinement model"])
