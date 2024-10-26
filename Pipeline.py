import torch
import torchvision
from PIL import Image
from lightly.transforms import MoCoV2Transform
from torch import nn
from torchvision.transforms import transforms
from torchvision.transforms import ToPILImage

from feature_extraction.feature_extractor import FeatureExtractor
from filter_refinement.Probabilities_refinement.model.RefinementModel import RefinementModel, refine_probability_list
from moco import utils
from moco.embedder import Embedder


class Atlas:
    def __init__(self, region_model_weights, langauge_model_weights, moco_model_weights, refinements_model_weights, k, baseset):
        self.extractor = FeatureExtractor(region_model_weights, langauge_model_weights)
        resnet = torchvision.models.resnet18()
        net = nn.Sequential(*list(resnet.children())[:-1])
        self.embedder = Embedder(net)
        self.embedder.load_csv(moco_model_weights)
        self.refiner = RefinementModel()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.refiner.load_state_dict(torch.load(refinements_model_weights, map_location = device))
        self.moco_transform = torchvision.transforms.Compose([transforms.Resize((224, 224)),
                                                              MoCoV2Transform(input_size=224, cj_prob=0.2,
                                                                              cj_bright=0.1, cj_contrast=0.1,
                                                                              cj_hue=0.1, cj_sat=0.1, min_scale=0.5,
                                                                              random_gray_scale=0.0),
                                                              ])
        self.k = k
        self.baseset = utils.read_csv_with_tensor(baseset, 'query')

    def predict_from_vectors(self, vectors, images):
        refined_vectors = []
        for vec in vectors:
            if len(vec) == 120:
                refined_vectors.append(vec)
            else:
                refined_vectors.append(refine_probability_list(self.refiner, torch.tensor(vec).unsqueeze(0))[0])
        filtered_vectors = [utils.cluster_filter(utils.distances_matrix, vec) for vec in refined_vectors]
        predictions = [(utils.expected_val(f_vec, lat=True), utils.expected_val(f_vec, lat=False)) for f_vec in
                       filtered_vectors]

        queries = [self.embedder(self.moco_transform(img)[0].unsqueeze(0)) for img in images]
        final_predictions = [utils.predict_location(q, co[0], co[1], self.baseset, K=self.k) for q, co in
                             zip(queries, predictions)]
        return final_predictions

    def predict(self, images_path):
        vectors = self.extractor.extract_features(images_path)
        images = [Image.open(image_path) for image_path in images_path]

        return self.predict_from_vectors(vectors, images)

    def predict_from_images(self, images):
        with torch.no_grad():
            images = [ToPILImage()(image) for image in images]
            vectors = self.extractor.extract_features_from_images(images)

        return self.predict_from_vectors(vectors, images)
