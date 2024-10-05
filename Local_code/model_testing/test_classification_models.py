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


class TestModels(nn.Module):

    def __init__(self, batch_size, transform, model_path):
        self.batch_size = batch_size
        self.transform = transform

        self.model = torchvision.models.resnet50(weights=None)

        # Changing the last layer.
        num_classes = 120
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)

        self.model = nn.DataParallel(self.model)

        # Loading the model.
        model_dict_path = model_path
        self.model.load_state_dict(torch.load(model_dict_path))

        self.loss_func = nn.CrossEntropyLoss()

    def test_model(self, test_loader, device, handle_func=None):
        running_test_loss = 0.0
        predicted_labels_list = []
        true_labels_list = []
        lat_list = []
        lng_list = []
        all_probabilities = []

        just_to_know = 1

        print("Calculating accuracy on test.")

        with torch.no_grad():
            for images, labels, lat, lng in test_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = self.model(images)

                probabilities = nn.functional.softmax(outputs, dim=1).cpu().numpy()
                all_probabilities.extend(probabilities)

                _, predicted_labels = torch.max(torch.tensor(probabilities), 1)

                predicted_labels_list.extend(predicted_labels.cpu().numpy())
                true_labels_list.extend(labels.cpu().numpy())
                lat_list.extend(lat)
                lng_list.extend(lng)

                # TODO:: remove.
                if just_to_know == 1:
                    print(f"true_labels_list: {true_labels_list}, predicted_labels_list: {predicted_labels_list}")
                    just_to_know = 0

                loss = self.loss_func(outputs, labels)
                running_test_loss += loss.item()

        test_loss = running_test_loss / len(test_loader)

        predicted_labels_array = np.array(predicted_labels_list)
        true_labels_array = np.array(true_labels_list)
        lat_array = np.array(lat_list)
        lng_array = np.array(lng_list)

        accuracy = np.mean(predicted_labels_array == true_labels_array)

        if handle_func is not None:
            handle_func(predicted_labels_array, true_labels_array, lat_array, lng_array,
                        np.array(all_probabilities))

        print(f'Finished test, Loss: {test_loss:.4f}')
        print(f'Accuracy: {accuracy}.')
