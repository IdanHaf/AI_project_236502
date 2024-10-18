import torch
import torch.nn as nn
import torchvision
import numpy as np
from PIL import Image
import easyocr
import cv2


class LanguageModel:
    def __init__(self, transform, model, device):
        self.transform = transform

        self.model = model
        self.device = device
        self.model.to(self.device)
        self.model.eval()

        self.loss_func = nn.CrossEntropyLoss()
        self.reader = easyocr.Reader(['en'], gpu=True)

    def create_grid(self, cropped_img):
        """
            Scaling the size of each image without loosing data.

            :param cropped_img: The cropped image.
            :returns: new grid image with scaled size.
        """
        cols = round(448 / cropped_img.shape[1])
        rows = round(448 / cropped_img.shape[0])

        if rows == 0:
            rows = 1
        if cols == 0:
            cols = 1

        row_stack = np.hstack([cropped_img] * cols)
        grid_image = np.vstack([row_stack] * rows)

        return grid_image

    def detect_text_from_image(self, batch_images):
        """
            Using EasyOCR to detect text in an image.
            EasyOCR GitHub - 'https://github.com/JaidedAI/EasyOCR'

            :param batch_images: The image batch to detect text.
            :returns: vector containing lists of detected and cropped text images from each image
            in batch_images.
        """
        text_img_batch = []

        for img in batch_images:
            img = np.array(img)

            text_img_lst = []
            results = self.reader.readtext(img)

            for bbox, text, prob in results:
                top_left = tuple(map(int, bbox[0]))
                bottom_right = tuple(map(int, bbox[2]))

                h, w, _ = img.shape
                top_left = (max(0, top_left[0]), max(0, top_left[1]))
                bottom_right = (min(w, bottom_right[0]), min(h, bottom_right[1]))

                cropped_img = img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

                if cropped_img is None or cropped_img.size == 0:
                    print(f"invalid image skipping...")
                    continue

                # Defined as "too low resolution".
                if cropped_img.shape[0] < 10 and cropped_img.shape[1] < 10:
                    continue

                # Resize by 40 low resolution images (not resizing too much).
                if cropped_img.shape[0] < 60 or cropped_img.shape[1] < 60:
                    cropped_img = cv2.resize(cropped_img, (cropped_img.shape[1] + 40, cropped_img.shape[0] + 40))

                # Try to resize image using rowxcol grid of the image.
                grid_image = self.create_grid(cropped_img)
                grid_image = Image.fromarray(grid_image)
                text_img_lst.append(grid_image)

            text_img_batch.append(text_img_lst)

        return text_img_batch

    def detect_language(self, img):
        """
            Apply the text detection model, and then using the script detection.

            :param img: Single image to detect language.
            :returns: language probability vector.
        """
        text_img_lst = self.detect_text_from_image([img])[0]

        # If no text was found return 0 vector.
        if len(text_img_lst) == 0:
            return np.zeros(9)

        images_transformed = [self.transform(text_img) for text_img in text_img_lst]

        images_batch = torch.stack(images_transformed, dim=0)
        images_batch = images_batch.to(self.device)

        with torch.no_grad():
            outputs = self.model(images_batch)

        probabilities = nn.functional.softmax(outputs, dim=1).cpu().numpy()
        # Filter probabilities with max < 0.6.
        max_probs = np.max(probabilities, axis=1)
        mask = max_probs > 0.6
        probabilities = probabilities[mask]
        mean_prob = np.mean(probabilities, axis=0)
        # From mean vector, Filter low probabilities.
        mean_prob[mean_prob < 0.1] = 0
        # Normalize remaining probabilities.
        sum_probs = np.sum(mean_prob)

        if sum_probs > 0:
            mean_prob /= sum_probs

        return mean_prob

    def test_model(self, test_loader):
        """
            Testing the model on test dataset.
            prints the recall of each class.
        """
        label_accuracy = np.zeros(9)
        count_labels = np.zeros(9)

        with torch.no_grad():

            for batch_images, batch_labels in test_loader:
                batch_images = batch_images.to(self.device)

                outputs = self.model(batch_images)
                probabilities = nn.functional.softmax(outputs, dim=1).cpu()
                _, predicted_labels = torch.max(probabilities, 1)

                for predicted_label, batch_label in zip(predicted_labels, batch_labels):
                    if predicted_label == batch_label:
                        label_accuracy[predicted_label] += 1

                    count_labels[batch_label] += 1

        label_accuracy = label_accuracy / count_labels

        print(f"Label Recall: {label_accuracy}")

    def loader_detect_language(self, loader):
        """
            Apply the text detection model, and then using the script detection.

            :param loader: The loader to detect languages.
            :returns: language probability vector.
        """
        predicted_probabilities = []
        with torch.no_grad():
            self.model.to(self.device)

            for batch_images, batch_labels in loader:
                text_img_batch = self.detect_text_from_image(batch_images)
                same_image_counter = []
                images_transformed = []

                for text_img_lst in text_img_batch:
                    image_lst_length = len(text_img_lst)
                    same_image_counter.append(image_lst_length)

                    if image_lst_length == 0:
                        continue

                    images_transformed += [self.transform(text_img) for text_img in text_img_lst]

                image_counter = 0
                batch_probabilities = []
                for idx, count in enumerate(same_image_counter):
                    image_counter += count

                    if image_counter >= 30 or idx == len(same_image_counter) - 1:
                        images_batch = torch.stack(images_transformed[:image_counter], dim=0)
                        del images_transformed[:image_counter]
                        images_batch = images_batch.to(self.device)

                        outputs = self.model(images_batch)

                        probabilities = nn.functional.softmax(outputs, dim=1).cpu().numpy()
                        batch_probabilities += probabilities
                        image_counter = 0

                for count in same_image_counter:
                    if count == 0:
                        predicted_probabilities.append(np.zeros(9))

                    else:
                        mean_prob = np.mean(probabilities[:count], axis=0)
                        del probabilities[:count]
                        predicted_probabilities.append(mean_prob)

        return predicted_probabilities
