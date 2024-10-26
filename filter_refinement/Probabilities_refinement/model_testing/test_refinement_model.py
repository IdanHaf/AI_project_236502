import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision
from torchvision import datasets
from refinementCustomDatasets.custom_dataset import CustomRefinementDataset
from model.RefinementModel import RefinementModel, validation_test_loop, plot_test_results
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import Counter


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Gpu is available: " + str(torch.cuda.is_available()))

    dataset = CustomRefinementDataset('./probs_dataset.csv')
    print(f"Number of samples in the dataset: {len(dataset)}")

    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    generator = torch.Generator()
    generator.manual_seed(387642706252)

    _, _, test_dataset = random_split(dataset, [train_size, val_size, test_size], generator=generator)
    # Plot histogram of the val datasets.
    # plot_val(val_dataset, "val")

    print(f"Using {torch.cuda.device_count()} GPUs")

    batch_size = 64

    test_loader = DataLoader(test_dataset, batch_size=32,
                              shuffle=True, num_workers=2, pin_memory=True)

    model_dict_path = './refine.pth'

    model = RefinementModel()
    model.load_state_dict(torch.load(model_dict_path))
    model.to(device)
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    # test_loss, test_accuracy, recall = validation_test_loop(test_loader, model, device, loss_func)
    # print(f"test Loss: {test_loss}, test Accuracy: {test_accuracy}")
    # print(f"Recall: {recall}")
    # print(f"Recall std: {np.std(recall)}")

    df_clusters = pd.read_csv("./clusters.csv")

    df_clusters['cluster_center'] = df_clusters['cluster_center'].apply(
        lambda x: np.fromstring(x.strip('[]'), sep=' '))

    plot_test_results(test_loader, model, device, df_clusters)



