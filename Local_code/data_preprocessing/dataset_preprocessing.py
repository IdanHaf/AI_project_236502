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
    img_files = os.listdir(dir_path)

    df = pd.DataFrame(img_files, columns=['id'])

    print(f"df length: {len(df)}")
    output_file_path = './big_dataset_id.csv'
    df.to_csv(output_file_path, index=False)
    print("file saved")


def expand_original_csv(csv_file):
    df = pd.read_csv(csv_file)
    num_cols = 4
    expanded_rows = []

    for _, row in df.iterrows():
        for i in range(num_cols):
            new_row = row.copy()
            new_row['id'] = f"{row['id']}{i}.jpeg"
            expanded_rows.append(new_row)

    expanded_df = pd.DataFrame(expanded_rows)

    output_file_path = './expanded_dataset.csv'
    expanded_df.to_csv(output_file_path, index=False)

    print("file saved")


def add_coords_data(csv_file, extracted_csv):
    df1 = pd.read_csv(extracted_csv)

    # file with id, lat, and lng columns
    df2 = pd.read_csv(csv_file)

    merged_df = pd.merge(df1, df2[['id', 'lat', 'lng']], on='id', how='left')

    output_file_path = './merged_dataset.csv'
    merged_df.to_csv(output_file_path, index=False)


def add_city_to_dataset(city_file, big_dataset_file):
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

    output_file_path = './images_dataset_with_clusters.csv'
    df.to_csv(output_file_path, index=False)
    print("file saved")

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


def add_cluster_data_to_cities(images_dataset_with_clusters, city_csv):
    df1 = pd.read_csv(images_dataset_with_clusters, dtype={'id': str})

    # file with id, lat, and lng columns
    df2 = pd.read_csv(city_csv, dtype={'place_id': str})

    df1_renamed = df1.rename(columns={'id': 'place_id', 'lng': 'lon'})
    merged_df = pd.merge(df2, df1_renamed[['place_id', 'lat', 'lon', 'cluster_label']], on=['place_id', 'lat', 'lon'],
                         how='left')

    output_file_path = './city_images_dataset.csv'
    merged_df.to_csv(output_file_path, index=False)


def remove_city_data(csv_file):
    df = pd.read_csv(csv_file, dtype={'id': str})

    filtered_df = df.iloc[:747932]

    output_file_path = './big_dataset_labeled.csv'
    filtered_df.to_csv(output_file_path, index=False)


if __name__ == "__main__":
    remove_city_data("./images_dataset_with_clusters.csv")

