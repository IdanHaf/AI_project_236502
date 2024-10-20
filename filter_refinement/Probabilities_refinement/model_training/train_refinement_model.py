import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision
from torchvision import datasets
from refinementCustomDatasets.custom_dataset import CustomRefinementDataset
from model.RefinementModel import RefinementModel, train_model, validation_loop
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter


def plot_val(dataset_to_hist, name):
    class_labels = []

    for _, label in dataset_to_hist:
        class_labels.append(label.item())

    label_counts = Counter(class_labels)
    classes, counts = zip(*sorted(label_counts.items()))  # Sort by class labels

    plt.figure(figsize=(10, 6))
    plt.bar(classes, counts, color='skyblue')
    plt.xlabel('Class')
    plt.ylabel('Number of Samples')
    plt.title('Class Distribution in Validation Dataset')
    plt.xticks(classes, rotation=90)
    plt.tight_layout()
    plt.savefig(f"{name}.png")
    plt.close()


def plot_loss_and_accuracy(train_losses, val_losses, val_accuracies, lr, batch_size):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f"Loss vs. Epochs for lr: {lr}, batch size: {batch_size}")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accuracies, label='Val Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title(f"Accuracy vs. Epochs for lr: {lr}, batch size: {batch_size}")
    plt.legend()

    plt.tight_layout()

    plt.savefig(f"training_{lr}_{batch_size}_validation_metrics.png")

    plt.close()


def plot_hyperparameter_tuning_results(all_train_losses, all_val_accuracies):
    plt.figure(figsize=(10, 6))
    for losses, lr, b_size in all_train_losses:
        plt.plot(losses, label=f"lr: {lr}, batch size: {b_size}")

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Hyperparameter training Losses')
    plt.legend()
    plt.grid(True)
    plt.savefig('hyperparameter_train_losses.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    for accuracies, lr, b_size in all_val_accuracies:
        plt.plot(accuracies, label=f"lr: {lr}, batch size: {b_size}")

    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Hyperparameter validation Accuracies')
    plt.legend()
    plt.grid(True)
    plt.savefig('hyperparameter_val_accuracies.png')
    plt.close()


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

    train_dataset, val_dataset, test_d = random_split(dataset, [train_size, val_size, test_size], generator=generator)
    # Plot histogram of the train and val datasets.
    plot_val(val_dataset, "val")
    plot_val(train_dataset, "train")
    plot_val(test_d, "test")

    all_models_training_loss = []
    all_models_val_accuracy = []

    batch_sizes = [64, 128, 32]
    learning_rates = [0.0001, 0.0005, 0.00075, 0.001]

    # Hyperparameter tuning (grid-search).
    for batch_size in batch_sizes:
        for learning_rate in learning_rates:
            print(f"Training batch size: {batch_size}, learning rate: {learning_rate}.")
            print(f"Using {torch.cuda.device_count()} GPUs")

            train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                      shuffle=True, num_workers=2, pin_memory=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size,
                                    shuffle=True)

            dataloaders = (train_loader, val_loader)
            model = RefinementModel()
            model.to(device)
            loss_func = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)

            train_losses, val_loss, val_accuracy = train_model(dataloaders,
                                                               model,
                                                               device,
                                                               loss_func,
                                                               optimizer,
                                                               50,
                                                               f"{batch_size}_{learning_rate}_refinement")

            plot_loss_and_accuracy(train_losses, val_loss, val_accuracy, learning_rate, batch_size)
            all_models_training_loss.append((train_losses, learning_rate, batch_size))
            all_models_val_accuracy.append((val_accuracy, learning_rate, batch_size))

    plot_hyperparameter_tuning_results(all_models_training_loss, all_models_val_accuracy)
