import pandas as pd
import sklearn.cluster as cluster


df_coords = pd.read_csv('../coords.csv', header=None)

kmeans = cluster.KMeans(n_clusters=72, init="k-means++", n_init=10)
kmeans = kmeans.fit(df_coords)

# Get cluster labels and cluster centers
cluster_labels = kmeans.labels_
df_labels = pd.DataFrame(cluster_labels)
print(df_labels)
df_labels.to_csv('coords_labels.csv', header=False, index=False)
