import pandas as pd
import sklearn.cluster as cluster
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np
import seaborn as sns


def plot_cluster_dataset(num_clusters, df_coords):
    print(f"start creating plots for {num_clusters}")

    kmeans = cluster.KMeans(n_clusters=num_clusters, init="k-means++", n_init=10)
    kmeans = kmeans.fit(df_coords)

    # Get cluster labels
    cluster_labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_

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
    latitude = df_coords['lat'].values
    longitude = df_coords['lng'].values

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


if __name__ == "__main__":
    df = pd.read_csv('./combined_city_and_big_dataset.csv', dtype={'id': str})
    df_coords = pd.DataFrame(df[["lat", "lng"]])

    clusters_num = [120, 130]

    for c in clusters_num:
        plot_cluster_dataset(c, df_coords)

# df_labels = pd.DataFrame(cluster_labels)
# print(df_labels)
# df_labels.to_csv('coords_labels.csv', header=False, index=False)
# class_counts = df['class'].value_counts()
# print(class_counts)















