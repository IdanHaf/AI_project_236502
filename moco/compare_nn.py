import numpy as np
import torch
import torchvision
from lightly.transforms import MoCoV2Transform
from matplotlib import pyplot as plt
from torch import nn
from torchvision.transforms import transforms
import swifter

import utils
from embedder import Embedder
from test_custom_dataset import CustomImageDataset

validation = 'val.csv'
model_weights = 'model.pth'

validation_df = utils.read_vector_csv(validation, 'prob_vector')

city_dataset_path = './Images'
city_csv_file_path = './city_images_dataset.csv'
big_dataset_path = './big_dataset'
big_csv_file_path = './big_dataset_labeled.csv'

transform = torchvision.transforms.Compose([transforms.Resize((224, 224)),
                                            MoCoV2Transform(input_size=224, cj_prob=0.2, cj_bright=0.1, cj_contrast=0.1,
                                                            cj_hue=0.1, cj_sat=0.1, min_scale=0.5,
                                                            random_gray_scale=0.0),
                                            ])
device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = CustomImageDataset(city_csv_file_path, city_dataset_path, big_csv_file_path, big_dataset_path, transform)


def apply_row(n_errors, filtered_errors, original_errors, baseset, embedder, row):
    o_lat, o_lng = utils.get_cluster_center(np.argmax(row['prob_vector']))
    lat, lng = row['lat'], row['lng']
    o_err = utils.haversine(o_lat, o_lng, lat, lng)
    original_errors.append(o_err)
    filtered_vec = utils.cluster_filter(utils.distances_matrix, row['prob_vector'])
    f_lat = utils.expected_val(filtered_vec, lat=True)
    f_lng = utils.expected_val(filtered_vec, lat=False)
    f_err = utils.haversine(f_lat, f_lng, lat, lng)
    filtered_errors.append(f_err)

    idx = row['id']
    img, _ = dataset[idx][0]
    img = img.unsqueeze(0).to(device)

    with torch.no_grad():
        q = embedder(img).cpu().detach()

    for k in range(3, 9, 2):
        n_lat, n_lng = utils.predict_location(q, f_lat, f_lng, baseset, K=k)
        err = utils.haversine(n_lat, n_lng, lat, lng)
        if k not in n_errors:
            n_errors[k] = []
        n_errors[k].append(err)


if __name__ == '__main__':
    baseset = utils.read_csv_with_tensor('baseset.csv', 'query')
    resnet = torchvision.models.resnet18()
    net = nn.Sequential(*list(resnet.children())[:-1])
    embedder = Embedder(net)
    embedder.load_csv(model_weights)
    embedder.to(device)
    embedder.eval()
    original_errors = []
    filtered_errors = []
    n_errors = {}

    print("start validating")
    validation_df.swifter.apply(
        lambda row: apply_row(n_errors, filtered_errors, original_errors, baseset, embedder, row), axis=1)

    # plot the graphs
    plt.figure()
    utils.Q_graph(original_errors, 'percentage of the dataset', 'Distance in KM', 'Quantile graph of the error',
                  'Classifier')
    utils.Q_graph(filtered_errors, 'percentage of the dataset', 'Distance in KM', 'Quantile graph of the error',
                  'Filter')
    for k in range(3, 9, 2):
        utils.Q_graph(n_errors[k], 'percentage of the dataset', 'Distance in KM', 'Quantile graph of the error',
                      f'{k} neighbors comparison')
    # Plot the first 70%
    plt.figure()
    utils.Q_graph(original_errors, 'percentage of the dataset', 'Distance in KM', 'Quantile graph of the error to 70%',
                  'Classifier', max_perc=70)
    utils.Q_graph(filtered_errors, 'percentage of the dataset', 'Distance in KM', 'Quantile graph of the error to 70%',
                  'Filter', max_perc=70)
    for k in range(1, 10, 2):
        utils.Q_graph(n_errors[k], 'percentage of the dataset', 'Distance in KM', 'Quantile graph of the error to 70%',
                      f'{k} neighbors comparison', max_perc=70)
