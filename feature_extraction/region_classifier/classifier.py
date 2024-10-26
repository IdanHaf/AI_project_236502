import torch
import torchvision
from torch import nn
import numpy as np
import pandas as pd
from tqdm import tqdm
# from test_custom_dataset import CustomImageDataset


class Classifier:
    def __init__(self, model_weights_path, transform=None):
        self.transform = transform
        model = torchvision.models.resnet50(weights=None)

        # Changing the last layer.
        num_classes = 120
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)

        model = nn.DataParallel(model)

        # Loading the model.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Gpu is available: " + str(torch.cuda.is_available()))
        model_dict_path = model_weights_path
        model.load_state_dict(torch.load(model_dict_path, map_location=self.device))
        model.eval()
        self.model = model

    def predict_list_images(self, images):
        """
            Give a prediction for list of images.
            :param images: list of images.

            :return: list of probabilities predicted for each image.
        """
        self.model.to(self.device)
        self.model.eval()
        # If no images were given.
        if len(images) == 0:
            raise ValueError("Array of images was empty")

        images_transformed = [self.transform(img) for img in images]

        images_batch = torch.stack(images_transformed, dim=0)
        images_batch = images_batch.to(self.device)

        with torch.no_grad():
            outputs = self.model(images_batch)

        probabilities = nn.functional.softmax(outputs, dim=1).cpu().numpy()
        probabilities = [prob.tolist() for prob in probabilities]

        return probabilities

    def predict_loader_batch(self, images):
        self.model.eval()
        with torch.no_grad():
            images = images.to(self.device)
            outputs = self.model(images)
            probabilities = nn.functional.softmax(outputs, dim=1)
            return probabilities

    def export(self, dataloader, result_file):
        df = pd.DataFrame(columns=['id', 'lat', 'lng', 'label', 'prob_vector'])
        for images, labels, lat, lng, ids in dataloader:
            probabilities = self.predict_loader_batch(images)
            probabilities = probabilities.cpu().numpy()
            probabilities = [prob.tolist() for prob in probabilities]
            new_df = pd.DataFrame({'id': ids, 'lat': lat, 'lng': lng, 'label': labels, 'prob_vector': probabilities})
            df = pd.concat([df, new_df])

        df.to_csv(result_file, index=False)

    def minimal_export(self, dataloader, result_file, convert_label):
        """
            Same as export for data loader with only images and labels.
            :param dataloader: dataloader with images and labels.
            :param result_file: path to save.
            :param convert_label: np.arr that holds at each idx the corresponding label.

            Saves the model results to a CSV file.
            Prints the model accuracy.
        """
        accuracy = 0
        counter = 0
        df = pd.DataFrame(columns=['label', 'prob_vector'])
        for images, labels in tqdm(dataloader, desc="Processing batches"):
            probabilities = self.predict_loader_batch(images).cpu().numpy()
            _, predicted_labels = torch.max(torch.tensor(probabilities), 1)

            probabilities = [prob.tolist() for prob in probabilities]

            labels = labels.numpy()
            labels = [int(convert_label[label]) for label in labels]

            accuracy += np.sum(predicted_labels == labels)
            counter += len(labels)

            new_df = pd.DataFrame({'label': labels, 'prob_vector': probabilities})
            df = pd.concat([df, new_df], ignore_index=True)

        df.to_csv(result_file, index=False)
        print(f"Accuracy: {accuracy / counter}")
