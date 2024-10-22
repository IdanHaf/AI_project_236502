import torchvision.transforms as transforms
from feature_extraction.language_classification.language_model import LanguageModel
from feature_extraction.region_classifier.classifier import Classifier
from PIL import Image


class FeatureExtractor:
    def __init__(self, region_model_weights_path, language_model_weights_path):
        self.region_model_path = region_model_weights_path
        self.language_model_path = language_model_weights_path

    def extract_features(self, images_path_list):
        """
        Extract features from images.
        :param images_path_list: list of image paths to extract features from.

        :return: list of extracted probabilities for each image.
        """
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        region_classifier = Classifier(transform, self.region_model_path)
        lang_model = LanguageModel(self.language_model_path, transform)

        images_to_predict = [Image.open(img_path) for img_path in images_path_list]

        # Get probabilities vectors for images from region classifier and language.
        regions_probabilities = region_classifier.predict_list_images(images_to_predict)
        lang_probabilities = lang_model.list_detect_language(images_to_predict)

        combine_probs = [(reg_prob + lang_prob) for reg_prob, lang_prob in zip(regions_probabilities, lang_probabilities)]

        return combine_probs
