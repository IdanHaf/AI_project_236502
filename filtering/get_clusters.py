import pandas as pd

labeled_df = pd.read_csv('big_dataset_labeled.csv')
clusters = labeled_df[['cluster_label', 'cluster_center']].drop_duplicates(keep='first').sort_values(by='cluster_label', ascending=True)
clusters.to_csv("clusters.csv", index=False)