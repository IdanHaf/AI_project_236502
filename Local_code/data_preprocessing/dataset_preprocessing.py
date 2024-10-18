import pandas as pd
import numpy as np
import sklearn.cluster as cluster
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import os

"""
    Creates csv file with images names, lat, lng and classes.
"""


def create_csv_from_data(dir_path):
    """
    Create the big dataset csv
    :param dir_path: The dir of the augmented images
    :return: None
    """
    img_files = os.listdir(dir_path)
    df = pd.DataFrame(img_files, columns=['id'])

    print(f"df length: {len(df)}")
    output_file_path = './big_dataset_id.csv'
    df.to_csv(output_file_path, index=False)
    print("file saved")


def add_coords_data(csv_file, extracted_csv):
    """
    Add coordinates to csv file
    :param csv_file:  The original csv file
    :param extracted_csv: The extracted csv file
    """
    df1 = pd.read_csv(extracted_csv)
    df1['temp'] = df1['id'].apply(lambda x: x.rsplit('.', 1)[0][:-1])
    # file with id, lat, and lng columns
    df2 = pd.read_csv(csv_file)

    merged_df = pd.merge(df1, df2[['id', 'lat', 'lng']].rename(columns={"id":'temp'}), on='temp', how='left')
    merged_df = merged_df[['id', 'lat', 'lng']]
    output_file_path = './merged_dataset.csv'
    merged_df.to_csv(output_file_path, index=False)


def add_city_to_dataset(city_file, big_dataset_file):
    """
    Merge the city dataset with the big dataset
    :param city_file: The city dataset
    :param big_dataset_file: The big dataset file
    """
    city_df = pd.read_csv(city_file)
    big_data_df = pd.read_csv(big_dataset_file)

    city_df_renamed = city_df[['place_id', 'lat', 'lon']].rename(columns={
        'place_id': 'id',
        'lon': 'lng'
    })

    combined_df = pd.concat([big_data_df, city_df_renamed], ignore_index=True)

    output_file_path = './combined_city_and_big_dataset.csv'
    combined_df.to_csv(output_file_path, index=False)


def add_cluster_column(file_name, num_clusters):
    df = pd.read_csv(file_name, dtype={'id': str})
    coordinates_df = df[['lat', 'lng']]

    kmeans = cluster.KMeans(n_clusters=num_clusters, init="k-means++", n_init=10)
    kmeans = kmeans.fit(coordinates_df)

    cluster_labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_

    df['cluster_label'] = cluster_labels
    df['cluster_center'] = df['cluster_label'].apply(lambda label: cluster_centers[label])

    # output_file_path = './images_dataset_with_clusters.csv'
    # df.to_csv(output_file_path, index=False)
    # print("file saved")

    # Count number of samples in each cluster
    cls_hist = np.bincount(cluster_labels, minlength=num_clusters)
    filtered_cluster_indices = np.where(cls_hist < 1000)[0]

    print("Clusters with fewer than 1000 samples:")
    for idx in filtered_cluster_indices:
        print(f"Cluster {idx}: {cls_hist[idx]} samples")

    # Plot the number of samples in each cluster
    plt.figure(figsize=(12, 6))
    plt.bar(range(num_clusters), cls_hist)
    plt.xlabel('Cluster')
    plt.ylabel('Number of Image Samples')
    plt.title('Number of Images in Each Cluster')

    num_images_cluster_file = f"./images_per_clusters_{num_clusters}.png"
    plt.savefig(num_images_cluster_file)
    plt.close()

    # Plot clusters on map.
    latitude = coordinates_df['lat'].values
    longitude = coordinates_df['lng'].values

    plt.figure(figsize=(12, 8))

    m = Basemap(projection='mill', llcrnrlat=-60, urcrnrlat=90,
                llcrnrlon=-180, urcrnrlon=180, resolution='c')

    m.drawcoastlines()
    m.drawcountries()
    m.drawstates()

    x, y = m(longitude, latitude)

    m.scatter(x, y, c=cluster_labels, cmap='viridis', alpha=0.7, s=50)

    x_centers, y_centers = m(cluster_centers[:, 1], cluster_centers[:, 0])
    m.scatter(x_centers, y_centers, marker='x', color='red', s=100, label='Cluster Centers')

    output_file_path = f"./map_clusters_{num_clusters}.png"
    plt.savefig(output_file_path, dpi=300)

    plt.close()

    return df


def add_cluster_data_to_cities(df_with_clusters, city_csv):
    df1 = df_with_clusters

    # file with id, lat, and lng columns
    df2 = pd.read_csv(city_csv, dtype={'place_id': str})

    df1_renamed = df1.rename(columns={'id': 'place_id', 'lng': 'lon'})
    merged_df = pd.merge(df2, df1_renamed[['place_id', 'lat', 'lon', 'cluster_label', 'cluster_center']],
                         on=['place_id', 'lat', 'lon'],
                         how='left')

    output_file_path = './city_images_dataset.csv'
    merged_df.to_csv(output_file_path, index=False)


def remove_city_data(df_with_clusters):
    df = df_with_clusters

    filtered_df = df.iloc[:747932]

    output_file_path = './big_dataset_labeled.csv'
    filtered_df.to_csv(output_file_path, index=False)


# Switch cluster column for future training.
def drop_cluster_col(csv_file):
    df = pd.read_csv(csv_file, dtype={'id': str})
    df = df.drop(columns=['cluster_label'])

    output_file_path = './combined_city_and_big_dataset.csv'
    df.to_csv(output_file_path, index=False)


def preprocess_datasets(combined_dataset_file_path, cities_dataset_file_path):
    df_clustered = add_cluster_column(combined_dataset_file_path, 120)
    add_cluster_data_to_cities(df_clustered, cities_dataset_file_path)
    remove_city_data(df_clustered)


if __name__ == "__main__":
    preprocess_datasets("./combined_city_and_big_dataset.csv",
                        "./city_dataset_labels.csv")

