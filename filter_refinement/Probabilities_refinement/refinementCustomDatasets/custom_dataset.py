import torch
from torch.utils.data import Dataset
import pandas as pd
import ast
import numpy as np


class CustomRefinementDataset(Dataset):
    def __init__(self, csv_path):
        self.df_dataset = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.df_dataset)

    def __getitem__(self, idx):
        row = self.df_dataset.iloc[idx]
        region_prob = ast.literal_eval(row['prob_vector'])
        lang_prob = ast.literal_eval(row['lang_probs'])
        label = row['label']

        prob_input = region_prob + lang_prob

        prob_tensor = torch.tensor(prob_input, dtype=torch.float32)

        label_tensor = torch.tensor(label, dtype=torch.long)

        return prob_tensor, label_tensor


