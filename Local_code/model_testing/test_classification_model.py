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


'''
    :param - two couples of (lat, lng).
    :return - distance in km between the coordinates points.
'''


def calc_distance(lat1, lon1, lat2, lon2):
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


def calc_expected_values(cluster_probabilities, real_labels):
    lat_centers = [-26.3633653, 27.61790643, 41.71007335, 50.78678275,
                   -29.84909793, 10.15807816, 55.7225008, -27.99005751,
                   39.41694472, 34.21583426, 4.19290006, 45.4710219,
                   39.90257948, 43.73714566, 40.05460995, 34.00003998,
                   -32.23191646, 47.55329067, -6.00054232, 48.15725426,
                   51.50309588, 16.45250807, 8.18357377, 29.21081863,
                   62.55102876, 52.79545913, -6.91185322, -36.10031501,
                   60.70094222, -0.45027543, -45.07963399, -34.560591,
                   40.29788564, 42.2112062, -3.93428088, 53.28342709,
                   15.2741314, 63.27994421, 39.95257763, 58.56615608,
                   42.29616268, 48.55555214, 40.99227843, 48.88271608,
                   37.56601173, 20.05979166, -27.95558053, 65.41527794,
                   50.86615048, 20.2614738, 23.43897974, 35.61970007,
                   45.7313998, 43.07312505, -37.52450927, 34.22805196,
                   26.56558704, 53.24071265, 19.18009577, 33.64679816,
                   55.35831273, 32.72561153, 59.8935473, -21.36007501,
                   22.65918682, 54.22316696, 59.46505194, -6.07532882,
                   -14.40312252, 19.9742151, -14.95094727, -34.15811507,
                   60.915795, 56.33441673, 54.64562798, -1.24850016,
                   43.22221331, 15.16453573, 39.89287373, 32.14947681,
                   46.32263775, 38.19125959, 60.17000663, 6.09999949,
                   43.39465304, 43.9380493, 66.99828821, 45.20577389,
                   45.74160595, 58.12084934, 49.04843861, -21.07855508,
                   56.674006, 56.04950503, 18.01940104, 33.93723187,
                   53.34970877, 40.15281404, 36.14267634, 47.05737437,
                   25.26248082, -25.0377829, 52.31692669, -45.78335098,
                   20.03904734, -32.05994983, 39.52008986, 34.83850484,
                   45.8667753, 52.04062807, 60.17518067, 10.43582607,
                   -22.62003749, 42.01443917, 37.69218985, -5.01643933,
                   -18.06397627, -16.90975639, -38.41903142, 25.56747439]

    lng_centers = [-51.64814878, 77.34279755, -88.82833792, 5.10699578,
                   150.94230312, 122.98814364, 49.55616946, 30.18056371,
                   -121.58353409, 130.54908928, 100.84648755, 26.20881699,
                   9.19462345, -72.05609367, -3.2681433, -88.19027555,
                   117.21111347, -1.23915278, -78.36378132, 20.44712603,
                   -111.19647894, 79.22600802, 5.59776208, -97.94558581,
                   30.22070842, 94.24227974, -38.1116018, -69.78439763,
                   16.08221254, 36.40921449, 170.59775248, 138.82743964,
                   -76.20352555, -97.16766369, 119.34930316, 36.44356618,
                   103.72148438, -148.29162045, 15.55087507, 73.07949147,
                   141.29266908, 2.1311728, -105.07833495, 15.11738119,
                   37.23876026, -99.8160219, -64.71978676, -20.4631883,
                   108.21897633, 120.89549711, 87.71481285, 137.31423922,
                   9.34196014, -112.71366292, 145.19071256, -97.37565733,
                   -80.72215349, -56.15080015, -15.73656411, -111.67977449,
                   -120.35821525, -83.21308594, 25.30459648, -45.20908235,
                   72.8684829, 83.91641453, 124.96562438, 109.05881937,
                   -41.57979957, -156.71003578, -69.00763279, -58.79607332,
                   9.47884452, -3.5425472, 23.23677712, -58.82894054,
                   1.11216878, 100.37293534, -8.25436141, -92.34703387,
                   -65.44291928, -94.53379903, 150.6395856, -74.78302875,
                   12.69166303, -79.01775168, 21.00620538, -93.04293323,
                   41.90879507, 41.6750446, -100.93630539, 145.12680448,
                   11.92569823, 59.83430319, -91.32379331, -103.23951736,
                   -7.6928285, 22.85019758, -79.3551794, -120.69906374,
                   55.455717, 27.12623732, 18.25868495, -69.90795264,
                   76.77362443, 22.38183473, 30.09853352, -117.99637926,
                   5.01616734, -1.13295088, -133.44720929, 78.55806647,
                   117.44180788, -83.67156214, -85.22132137, -47.11575991,
                   131.94163753, -51.83107417, 175.18257043, -106.14740366]

    expected_values_arr = []

    for probs, real_label in zip(cluster_probabilities, real_labels):
        expected_value = 0
        lat2, lng2 = lat_centers[real_label], lng_centers[real_label]

        for idx, prob in enumerate(probs):
            lat1, lng1 = lat_centers[idx], lng_centers[idx]

            expected_value += prob * calc_distance(lat1, lng1, lat2, lng2)

        expected_values_arr.append(expected_value)

    return expected_values_arr


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
    model_tester = TestModels(batch_size, transform, model_dict_path)

    model_tester.test_model(test_loader, device, add_predicted_labels_to_csv)
