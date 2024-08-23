import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from customDatasets.custom_dataset import CustomImageDataset
import copy

# loading bar
from tqdm import tqdm


def plot_loss_and_accuracy(num_epochs, train_losses, val_losses, val_accuracies, lr, batch_size):
    epochs = range(1, num_epochs + 1)

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

    num_epochs = 50

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    city_dataset_path = './Images'
    city_csv_file_path = './city_images_dataset.csv'
    big_dataset_path = './results'
    big_csv_file_path = './big_dataset_labeled.csv'
    dataset = CustomImageDataset(city_csv_file_path, city_dataset_path, big_csv_file_path, big_dataset_path, transform)  # Idan Dataset loading

    print(f"Number of samples in the dataset: {len(dataset)}")

    train_size = int(0.6 * len(dataset))
    val_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    generator = torch.Generator()
    generator.manual_seed(387642706252)

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size], generator=generator)

    all_models_training_loss = []
    all_models_val_accuracy = []

    batch_sizes = [64, 128]
    learning_rates = [0.02, 0.03, 0.04]

    # Hyperparameter tuning (grid-search).
    for batch_size in batch_sizes:
        for learning_rate in learning_rates:
            print(f"Training batch size: {batch_size}, learning rate: {learning_rate}.")
            print(f"Using {torch.cuda.device_count()} GPUs")

            train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                      shuffle=True, num_workers=2, pin_memory=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size,
                                    shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size,
                                     shuffle=False)

            # Using the pre-trained ResNet-50 model
            model = torchvision.models.resnet50(weights='IMAGENET1K_V1')
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, 120)

            model = nn.DataParallel(model)
            model = model.to(device)

            # TODO:: add scheduler.
            loss_func = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)

            train_losses = []
            val_losses = []
            val_accuracies = []

            patience = 5
            best_epoch = 0
            best_val_loss = float('inf')
            best_model_weights = None
            stopped_epoch = num_epochs

            # Training loop.
            for epoch in range(num_epochs):
                train_bar = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs}")

                running_loss = 0.0
                model.train()
                for inputs, labels in train_loader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()

                    # Forward propogation
                    outputs = model(inputs)
                    loss = loss_func(outputs, labels)

                    # Backward propogation
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    train_bar.set_postfix(loss=running_loss / len(train_bar))

                avg_train_loss = running_loss / len(train_loader)
                train_losses.append(avg_train_loss)

                # Validation on val_loader.
                model.eval()
                running_val_loss = 0.0
                correct_predictions = 0
                total_predictions = 0

                with torch.no_grad():
                    for inputs, labels in val_loader:
                        inputs = inputs.to(device)
                        labels = labels.to(device)

                        val_outputs = model(inputs)
                        val_loss = loss_func(val_outputs, labels)
                        running_val_loss += val_loss.item()

                        probabilities = nn.functional.softmax(val_outputs, dim=1)
                        _, predicted_labels = torch.max(probabilities, 1)

                        correct_predictions += (predicted_labels == labels).sum().item()
                        total_predictions += labels.size(0)

                avg_val_loss = running_val_loss / len(val_loader)
                val_losses.append(avg_val_loss)

                val_accuracy = correct_predictions / total_predictions
                val_accuracies.append(val_accuracy)

                print(f'Epoch {epoch + 1}/{num_epochs}, avg training loss: {avg_train_loss:.4f}, '
                      f'avg validation loss: {avg_val_loss:.4f}, '
                      f'validation accuracy: {val_accuracy:.4f}')

                # Early stopping according to val loss.
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_model_weights = copy.deepcopy(model.state_dict())
                    best_epoch = epoch
                    patience = 5
                else:
                    patience -= 1
                    if patience == 0:
                        stopped_epoch = epoch
                        break

            torch.save(best_model_weights, f"classification_best_lr{learning_rate}_batch{batch_size}.pth")
            print(f"Module saved, best epoch: {best_epoch}")

            model.eval()
            plot_loss_and_accuracy(stopped_epoch, train_losses, val_losses, val_accuracies, learning_rate, batch_size)
            all_models_training_loss.append((train_losses, learning_rate, batch_size))
            all_models_val_accuracy.append((val_accuracies, learning_rate, batch_size))

    plot_hyperparameter_tuning_results(all_models_training_loss, all_models_val_accuracy)
