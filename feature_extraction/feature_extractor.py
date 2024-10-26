import numpy as np
import torchvision.transforms as transforms
from feature_extraction.language_classification.language_model import LanguageModel
from feature_extraction.region_classifier.classifier import Classifier
from PIL import Image


class FeatureExtractor:
    def __init__(self, region_model_weights_path, language_model_weights_path):
        self.region_model_path = region_model_weights_path
        self.language_model_path = language_model_weights_path
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.region_classifier = Classifier(self.region_model_path, transform)
        self.lang_model = LanguageModel(self.language_model_path, transform)

    def extract_features(self, images_path_list):
        """
        Extract features from images.
        :param images_path_list: list of image paths to extract features from.

        :return: list of extracted probabilities for each image.
        """

        images_to_predict = [Image.open(img_path) for img_path in images_path_list]

        # Get probabilities vectors for images from region classifier and language.
        regions_probabilities = self.region_classifier.predict_list_images(images_to_predict)
        lang_probabilities = self.lang_model.list_detect_language(images_to_predict)

        combine_probs = [(reg_prob + lang_prob) if not np.all(lang_prob == np.zeros(9)) else reg_prob for reg_prob, lang_prob in
                         zip(regions_probabilities, lang_probabilities)]

        return combine_probs

    def extract_features_from_images(self, images):
        """
        Extract features from images
        :param images: PIL images
        :return: list of extracted probabilities for each image.
        """

        regions_probabilities = self.region_classifier.predict_list_images(images)
        lang_probabilities = self.lang_model.list_detect_language(images)

        combine_probs = [(reg_prob + lang_prob) if not np.all(lang_prob == np.zeros(9)) else reg_prob for reg_prob, lang_prob in
                         zip(regions_probabilities, lang_probabilities)]

        return combine_probs
