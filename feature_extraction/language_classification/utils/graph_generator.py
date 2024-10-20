import math
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import swifter
from tqdm import tqdm

df_clusters = pd.read_csv("./clusters.csv")
df_dataset = pd.read_csv("./probs_dataset.csv")

df_clusters['cluster_center'] = df_clusters['cluster_center'].apply(
    lambda x: np.fromstring(x.strip('[]'), sep=' '))

print(df_clusters.head())


def expected_val(prob_vector, lat):
    res = 0
    for i in range(len(prob_vector)):
        condition = df_clusters['cluster_label'] == i
        idx = 0 if lat else 1
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


def Q_graph(arr, x_title, y_title, title, label, max_perc=100):
    sorted_arr = np.sort(arr)
    percentiles = np.arange(1, len(arr) + 1) / len(arr) * 100
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


probs = df_dataset['prob_vector'].apply(
    lambda x: np.fromstring(x.strip('[]'), sep=', '))

coords = df_dataset['coords'].apply(
    lambda x: np.fromstring(x.strip('[]'), sep=', '))


lat_values = np.array([expected_val(prob, True) for prob in tqdm(probs, desc='Probability vector')])
lng_values = np.array([expected_val(prob, False) for prob in tqdm(probs, desc='Probability vector')])

dist = np.array([haversine(e_lat, lng_values[idx], coords[idx][0], coords[idx][1])
                 for idx, e_lat in enumerate(lat_values)])

Q_graph(dist, 'percentage of the dataset', 'distance in KM',
        'Quantile_graph_of_predicted_error_after_expected_value', label="resnet50")




