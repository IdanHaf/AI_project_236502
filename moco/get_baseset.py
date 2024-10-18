import pandas as pd
import numpy as np
import tqdm

filename = 'train.csv'
sample_per_cluster = 600
num_cluster = 120

if __name__ == '__main__':
    np.random.seed(7)
    df = pd.read_csv(filename)
    sample_df = pd.DataFrame(columns=df.columns)
    for i in tqdm.tqdm(range(num_cluster)):
        n = min(sample_per_cluster, df[df['label'] == i].shape[0])
        rows = df[df['label'] == i].sample(n=n, ignore_index=True, replace=False)
        sample_df = pd.concat([sample_df, rows], ignore_index=True)
    sample_df.to_csv('sample_df.csv', index=False)
