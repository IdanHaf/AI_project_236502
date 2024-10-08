import pandas as pd
import numpy as np

filename = 'train.csv'
sample_per_cluster = 100
num_cluster=120

if __name__ == '__main__':
    np.random.seed(7)
    df = pd.read_csv(filename)
    sample_df = pd.DataFrame(columns=df.columns)
    for i in range(num_cluster):
        rows = df[df['label'] == i].sample(n=sample_per_cluster, ignore_index=True)
        sample_df = pd.concat([sample_df, df], ignore_index=True)
    sample_df.to_csv('sample_df.csv', index=False)
