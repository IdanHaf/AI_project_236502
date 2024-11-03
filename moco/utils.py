import os
import csv
import math
import ast
import numpy as np
import pandas as pd
import torch
from torch import tensor
from matplotlib import pyplot as plt
from scipy.special import softmax
import torch.nn.functional as F
from sklearn.metrics.pairwise import haversine_distances
import swifter

MAX_CLUSTER_DISTANCE = 5000
clusters_file_path = os.path.join('model_resources', 'clusters.csv')
clusters_df = pd.read_csv(clusters_file_path)
clusters_df['cluster_center'] = clusters_df['cluster_center'].apply(
    lambda x: np.fromstring(x.strip('[]'), sep=' '))

cluster_centers = np.radians(clusters_df['cluster_center'].to_list())
distances_matrix = haversine_distances(cluster_centers) * 6371
MAX_COMP_DISTANCE = 250


def expected_val(prob_vector, lat):
    """
    Predict coordinates according to expected value of the vector
    :param prob_vector: The cluster probability vector
    :param lat: whether we do it to the latitude
    :return: The predicted coordinates
    """
    res = 0
    vector = prob_vector
    for i in range(len(vector)):
        condition = clusters_df['cluster_label'] == i
        idx = 0 if lat else 1
        c = clusters_df[condition].cluster_center.to_numpy()[0]
        res += vector[i] * c[idx]
    return res


def Q_graph(arr, x_title, y_title, title, label, max_perc=100):
    """
    This creates a quantile graph of the result
    :param arr: The array to be plotted
    :param x_title: the title of the x-axis
    :param y_title: the title of the y-axis
    :param title: the title of the graph
    :param label: the label of the plot
    :param max_perc: The percentage to up we plot.
    """
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
    plt.savefig(f'plots/{title}.png')
    plt.show()


def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two coordinates
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


def relative_probs(x):
    """
    Computes the relative probabilities of each element in x, we use this instead of softmax
    Due to the original array having small values which becomes noisy
    :param x: The original array
    :return: Updated probability array
    """
    s = np.sum(x)
    x /= s
    return x


def get_neighbors(elements, idx, n):
    """
    Given a cluster index and a number of clusters, return the neighbors of specified cluster.
    :param elements: The clusters
    :param idx: The cluster index
    :param n: The number of clusters
    :return: The indices of the neighbors
    """
    sorted_indices = np.argsort(elements[idx])
    return sorted_indices[1:math.ceil(n / 8)]


def read_vector_csv(filename, vector_name):
    """
    Read csv and convert a saved array into a numpy array
    :param filename: The file we want to read
    :param vector_name: The vector field
    :return: The output data frame
    """
    df = pd.read_csv(filename)
    df[vector_name] = df[vector_name].swifter.apply(
        lambda x: np.fromstring(x.strip('[]'), sep=', '))
    return df


def get_cluster_center(cluster):
    """
    Returns the cluster center
    :param cluster: The cluster we want his center
    :return: The center of the cluster
    """
    return clusters_df[clusters_df['cluster_label'] == cluster]['cluster_center'].values[0]


def max_cluster(df, arr):
    """
    Computes the maximum cluster for the max element in arr
    :param df: The ndarray of the distances
    :param arr: The probability array
    :return: cluster of probabilities around the max element
    """
    n = arr.shape[0]
    prob = 0
    max_idx = np.argmax(arr)
    result = []

    neighbors = (get_neighbors(df, max_idx, n))
    for n_idx in neighbors:
        if arr[n_idx] != 0 and (not n_idx in result) and (df[max_idx][n_idx] <= MAX_CLUSTER_DISTANCE):
            result.append(n_idx)
            prob += arr[n_idx]
    prob += arr[max_idx]
    result.append(max_idx)

    return result, prob


def cluster_point(df, arr, idx, used=[]):
    """
    Get the indices of the cluster of a certain element
    :param df: The ndarray of the distances
    :param arr: The probability array
    :param idx: The element index
    :param used: Array of used indices
    :return: The indices of the cluster
    """
    n = arr.shape[0]
    prob = 0
    q = [idx]
    result = []
    if arr[idx] == 0:
        return result, prob

    for idx in q:
        neighbors = (get_neighbors(df, idx, n))
        for n_idx in neighbors:
            if arr[n_idx] != 0 and (not n_idx in result) and (not n_idx in used) and (
                    df[idx][n_idx] <= MAX_CLUSTER_DISTANCE):
                result.append(n_idx)
                prob += arr[n_idx]
        prob += arr[idx]
        result.append(idx)
    return result, prob


def cluster_vector(df, arr):
    """
    Get the cluster with the highest probability among all clusters
    :param df: The ndarray of the distances
    :param arr: The probability array
    :return: The best cluster
    """
    sorted_indexes = np.argsort(arr)[::-1]
    i = 0
    total_prob = 0
    best_cluster = []
    best_prob = 0
    used = []
    while best_prob < 1 - total_prob and i < arr.shape[0]:  # If the probability of the best
        cluster, prob = cluster_point(df, arr, sorted_indexes[i], used)
        used += cluster
        i += 1
        if prob > best_prob:
            best_prob = prob
            best_cluster = cluster

    return best_cluster, best_prob


def cluster_filter(df, arr):
    """
    Filter the element which are not in the best cluster
    :param df: The ndarray of the distances
    :param arr: The probability array
    :return: The filtered array
    """
    arr = np.copy(arr)
    cluster, _ = cluster_vector(df, arr)
    mask = np.ones_like(arr, dtype=bool)
    mask[cluster] = False
    arr[mask] = 0
    arr[cluster] = relative_probs(arr[cluster])
    return arr


def read_csv_with_tensor(filename, tensor_field):
    """
    Read csv with a tensor column
    :param filename: The csv name
    :param tensor_field: the column name
    :return: Dataframe of the csv
    """
    df = pd.read_csv(filename)
    df[tensor_field] = df[tensor_field].apply(
        lambda x: (eval(x).squeeze(0)))
    return df


def predict_location(q, lat, lng, baseset_df, K=1):
    """
    This function predict the latitude and longitude based on the embedding of the image and its rough coordinates
    :param q: The embedding
    :param lat: The image rough latitude
    :param lng: The image longitude
    :param baseset_df: Dataframe of samples for comparison
    :param K: The amount of neighbors to account for
    :return: predicted latitude, predicted longitude.
    """
    df = baseset_df.copy()
    df['dists'] = df.apply(lambda row: haversine(lat, lng, row['lat'], row['lng']), axis=1)
    if len(df[df['dists'] <= MAX_COMP_DISTANCE].index) <= 5 * K:
        # In this case there isn't enough data around the image
        return lat, lng
    else:
        df = df[df['dists'] <= MAX_COMP_DISTANCE]

    df['similarity'] = df.apply(lambda row: F.cosine_similarity(q.squeeze(0), row["query"], dim=0).item(), axis=1)
    df = df.nlargest(K, 'similarity')
    predicted_lat = df['lat'].mean()
    predicted_lng = df['lng'].mean()
    return predicted_lat, predicted_lng
