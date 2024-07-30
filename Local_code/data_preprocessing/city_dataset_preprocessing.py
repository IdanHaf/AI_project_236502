import pandas as pd
import numpy as np
import sklearn.cluster as cluster
import os


def combine_data(csv_dir_path):
    csv_city_files = os.listdir(csv_dir_path)

    combined_df = pd.DataFrame()

    # Print the CSV files
    print("CSV files in", csv_dir_path, ":")
    for csv_file in csv_city_files:
        print(csv_file)

        file_path = os.path.join(csv_dir_path, csv_file)
        df_city = pd.read_csv(file_path)

        df_city = df_city.sort_values(by=['place_id', 'year'], ascending=[True, False])
        df_city = df_city.drop_duplicates(subset='place_id', keep='first')

        combined_df = pd.concat([combined_df, df_city], ignore_index=True)

    combined_df = combined_df.sample(frac=1).reset_index(drop=True)

    #   Add classes column.
    coordinates_df = combined_df[['lat', 'lon']]

    kmeans = cluster.KMeans(n_clusters=23, init="k-means++", n_init=10)
    kmeans = kmeans.fit(coordinates_df)

    # Get cluster labels and cluster centers.
    cluster_labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_

    combined_df['cluster_label'] = cluster_labels

    # Count number of samples at each class.
    cls_hist = np.zeros(len(cluster_centers))

    for i in range(len(cluster_labels)):
        cls_hist[cluster_labels[i]] += 1

    print(cls_hist)

    print(f"combined_df length: {len(combined_df)}")
    output_file_path = './city_dataset_labels.csv'
    combined_df.to_csv(output_file_path, index=False)
    print("combined file saved")


def samples_from_cities(csv_path):
    df = pd.read_csv(csv_path)

    total_samples = 0
    city_counts = df.groupby('city_id').size()
    for city, count in city_counts.items():
        print(f"{city}: {count} values")
        total_samples += count
    print(f"total:{total_samples}")


if __name__ == "__main__":
    combine_data('./Dataframes')
    samples_from_cities('./city_dataset_labels.csv')

