import csv
import math

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.special import softmax
from sklearn.metrics.pairwise import haversine_distances
import swifter

MAX_CLUSTER_DISTANCE = 5000


def expected_val(prob_vector, lat):
    res = 0
    vector = prob_vector[0]
    for i in range(len(vector)):
        condition = clusters_df['cluster_label'] == i
        idx = 0 if lat else 1
        c = clusters_df[condition].cluster_center.to_numpy()[0]
        res += vector[i] * c[idx]
    return res


def Q_graph(arr, x_title, y_title, title):
    sorted_arr = np.sort(arr)
    percentiles = np.arange(1, len(arr) + 1) / len(arr) * 100
    plt.plot(percentiles, sorted_arr)

    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.title(title)
    plt.grid(True)
    plt.savefig(f'plots/{title}.png')
    plt.show()


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


def deviation_filter(arr, mode=0):
    """
    Filter probabilities which are too far from the max.
    :param arr: The probability array
    :return: Updated probability array
    """
    arr = np.copy(arr)
    condition = None
    std = np.std(arr)
    max_prob = np.max(arr)
    mean_prob = np.mean(arr)
    if mode == 0:
        condition = (arr < (max_prob - 2 * std))
    elif mode == 1:
        condition = (arr < (mean_prob - 2 * std))
    elif mode == 2:
        condition = (arr < mean_prob)
    elif mode == 3:
        condition = (arr < (max_prob - 4 * std))

    arr[condition] = 0
    arr[~condition] = softmax(arr[~condition])
    return arr


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


def check_filters(prediction_df, dist_matrix):
    original_correct = prediction_df.swifter.apply(
        lambda row: row['predicted_labels'] == row['true_labels'], axis=1)

    deviation_correct = [prediction_df.swifter.apply(
        lambda row: row['true_labels'] == np.argsort(deviation_filter(row['prob_vector'], i))[-1], axis=1) for i in
        range(4)]

    cluster_correct = prediction_df.swifter.apply(
        lambda row: row['true_labels'] == np.argsort(cluster_filter(dist_matrix, row['prob_vector']))[-1], axis=1)

    full_filter_correct = [prediction_df.swifter.apply(
        lambda row: row['true_labels'] ==
                    np.argsort(cluster_filter(dist_matrix, deviation_filter(row['prob_vector'], i)))[
                        -1], axis=1) for i in range(4)]

    original_correct = prediction_df[original_correct].shape[0]
    for i in range(4):
        deviation_correct_res = prediction_df[deviation_correct[i]].shape[0]
        print(f'deviation_correct: {deviation_correct_res} i {i}')

        full_filter_correct_res = prediction_df[full_filter_correct[i]].shape[0]
        print(f'full_filter_correct: {full_filter_correct_res} i {i}')

    cluster_correct = prediction_df[cluster_correct].shape[0]
    print(f'original_correct: {original_correct}')
    print(f'cluster_correct: {cluster_correct}')


# Note: the filters improved the prediction rate by a small margin but they didn't harm which is helpful for next stages
# It will require tuning of some variables on the complete csv.
if __name__ == '__main__':

    # Parse the clusters csv
    clusters_df = pd.read_csv('clusters.csv')
    # convert to ndarray
    clusters_df['cluster_center'] = clusters_df['cluster_center'].swifter.apply(
        lambda x: np.fromstring(x.strip('[]'), sep=' '))
    # convert to Radians
    cluster_centers = np.radians(clusters_df['cluster_center'].to_list())
    distances_matrix = haversine_distances(cluster_centers) * 6371

    # Parse the prediction csv
    columns = [f'Probability_Class_{i}' for i in range(120)]
    prediction_df = pd.read_csv('model_lr0.0005_predictions_with_prob.csv')
    # Create the 'prob_vector' column, combining the Probability_Class_i values into a list for each row
    prediction_df['prob_vector'] = prediction_df[columns].swifter.apply(lambda row: np.array(row), axis=1)
    prediction_df = prediction_df[['predicted_labels', 'true_labels', 'lat', 'lng', 'prob_vector']]
    # check_filters(prediction_df, distances_matrix)

    prediction_df = prediction_df.merge(clusters_df, left_on='predicted_labels', right_on='cluster_label', how='left')

    predicted_error = prediction_df[['lat', 'lng', 'cluster_center']].swifter.apply(
        lambda row: haversine(row[0], row[1], row[2][0], row[2][1]), axis=1)
    Q_graph(predicted_error, 'percentage of the dataset', 'distance in KM',
            'Quantile graph of predicted error after classifier')

    prediction_df['expected_lat'] = prediction_df[['prob_vector']].swifter.apply(
        lambda row: expected_val(row, True), axis=1)
    prediction_df['expected_lng'] = prediction_df[['prob_vector']].swifter.apply(
        lambda row: expected_val(row, False), axis=1)
    predicted_error_on_exp = prediction_df[['expected_lat', 'expected_lng', 'lat', 'lng']].swifter.apply(
        lambda row: haversine(row[0], row[1], row[2], row[3]), axis=1)
    Q_graph(predicted_error_on_exp, 'percentage of the dataset', 'distance in KM',
            'Quantile graph of predicted error after expected value')

    filter_errors = []
    for i in range(4):
        prediction_df['filtered_vector'] = prediction_df.swifter.apply(
            lambda row: (cluster_filter(distances_matrix, deviation_filter(row['prob_vector'], i))), axis=1)
        prediction_df['expected_lat'] = prediction_df[['filtered_vector']].swifter.apply(
            lambda row: expected_val(row, True), axis=1)
        prediction_df['expected_lng'] = prediction_df[['filtered_vector']].swifter.apply(
            lambda row: expected_val(row, False), axis=1)
        predicted_error_on_filter = prediction_df[['expected_lat', 'expected_lng', 'lat', 'lng']].swifter.apply(
            lambda row: haversine(row[0], row[1], row[2], row[3]), axis=1)
        filter_errors.append(predicted_error_on_filter)

        Q_graph(predicted_error_on_filter, 'percentage of the dataset', 'distance in KM',
                f'Quantile graph of predicted error after filter {i}')

    prediction_df['filtered_vector'] = prediction_df.swifter.apply(
        lambda row: (cluster_filter(distances_matrix, row['prob_vector'])), axis=1)

    prediction_df['expected_lat'] = prediction_df[['filtered_vector']].swifter.apply(
        lambda row: expected_val(row, True), axis=1)
    prediction_df['expected_lng'] = prediction_df[['filtered_vector']].swifter.apply(
        lambda row: expected_val(row, False), axis=1)
    predicted_error_on_filter = prediction_df[['expected_lat', 'expected_lng', 'lat', 'lng']].swifter.apply(
        lambda row: haversine(row[0], row[1], row[2], row[3]), axis=1)

    Q_graph(predicted_error_on_filter, 'percentage of the dataset', 'distance in KM',
            f'Quantile graph of predicted error after cluster filter ')

    sorted_arr = np.sort(predicted_error)
    percentiles = np.arange(1, len(predicted_error) + 1) / len(predicted_error) * 100
    plt.plot(percentiles, sorted_arr, label='classifier')

    sorted_arr = np.sort(predicted_error_on_exp)
    percentiles = np.arange(1, len(predicted_error_on_exp) + 1) / len(predicted_error_on_exp) * 100
    plt.plot(percentiles, sorted_arr, label='expected value on cluster')

    sorted_arr = np.sort(predicted_error_on_filter)
    percentiles = np.arange(1, len(predicted_error_on_filter) + 1) / len(predicted_error_on_filter) * 100
    plt.plot(percentiles, sorted_arr, label='cluster filter')

    for i in range(4):
        sorted_arr = np.sort(filter_errors[i])
        percentiles = np.arange(1, len(filter_errors[i]) + 1) / len(filter_errors[i]) * 100
        plt.plot(percentiles, sorted_arr, label=f'full filter {i}')

    plt.xlabel('percentage')
    plt.ylabel('distance in KM')
    plt.title("compare quantile")
    plt.grid(True)
    plt.legend()
    plt.savefig('plots/compare.png')
    plt.show()
