import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision
from torchvision import datasets
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


def plot_loss_and_accuracy(train_losses, val_losses, val_accuracies, lr, scheduler):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f"Loss vs. Epochs for lr: {lr}, scheduler: {scheduler}")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accuracies, label='Val Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title(f"Accuracy vs. Epochs for lr: {lr}, scheduler: {scheduler}")
    plt.legend()

    plt.tight_layout()

    plt.savefig(f"b3_language_training_{lr}_{scheduler}_validation_metrics.png")

    plt.close()


def plot_hyperparameter_tuning_results(all_train_losses, all_val_accuracies):
    plt.figure(figsize=(10, 6))
    for losses, lr, sched in all_train_losses:
        plt.plot(losses, label=f"lr: {lr}, batch size: {sched}")

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Hyperparameter training Losses')
    plt.legend()
    plt.grid(True)
    plt.savefig('hyperparameter_train_losses.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    for accuracies, lr, sched in all_val_accuracies:
        plt.plot(accuracies, label=f"lr: {lr}, batch size: {sched}")

    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Hyperparameter validation Accuracies')
    plt.legend()
    plt.grid(True)
    plt.savefig('language_hyperparameter_val_accuracies_b3.png')
    plt.close()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Gpu is available: " + str(torch.cuda.is_available()))

    num_epochs = 35

    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ColorJitter(brightness=0.1, contrast=0.2, saturation=0.1, hue=0.2),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = datasets.ImageFolder(root='language_dataset')
    print(f"labels: {dataset.classes}")

    print(f"Number of samples in the dataset: {len(dataset)} for example: {dataset[0]}")

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    generator = torch.Generator()
    generator.manual_seed(387642706252)

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform

    all_models_training_loss = []
    all_models_val_accuracy = []

    batch_sizes = [32]
    learning_rates = [0.005, 0.003, 0.007]
    best_val_loss_model = 1

    # Hyperparameter tuning (grid-search).
    for batch_size in batch_sizes:
        for learning_rate in learning_rates:
            print(f"Training batch size: {batch_size}, learning rate: {learning_rate}.")
            print(f"Using {torch.cuda.device_count()} GPUs")

            train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                      shuffle=True, num_workers=2, pin_memory=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size,
                                    shuffle=True)

            model = torchvision.models.efficientnet_b3(weights='IMAGENET1K_V1')
            num_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(num_features, 9)

            model = nn.DataParallel(model)
            model = model.to(device)

            loss_func = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            schedulers = [optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=0.0001),
                          torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.0001, max_lr=0.1,
                                                            step_size_up=5, mode="triangular"),
                          torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.0001, max_lr=0.1,
                                                            step_size_up=5, mode="triangular2"),
                          torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=1,
                                                                               eta_min=0.0001, last_epoch=-1)]

            schedulers_name = ['CosineAnnealingLR', 'CyclicLR_triangular', 'CyclicLR_triangular2',
                               'CosineAnnealingWarmRestarts']

            train_losses = []
            val_losses = []
            val_accuracies = []

            patience = 7
            best_epoch = 0
            best_val_loss = float('inf')
            best_model_weights = None
            best_scheduler = 'StepLR'

            for scheduler, scheduler_name in zip(schedulers, schedulers_name):
                print(f"Scheduler: {scheduler_name}.")

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

                    scheduler.step()

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
                        best_epoch = epoch + 1
                        best_scheduler = scheduler_name
                        patience = 7
                    else:
                        patience -= 1
                        if patience == 0:
                            break

                if best_val_loss < best_val_loss_model:
                    best_val_loss_model = best_val_loss
                    torch.save(best_model_weights, f"b3_language_best_lr{learning_rate}_{best_scheduler}.pth")
                    print(f"Module saved, best epoch: {best_epoch}")

                model.eval()
                plot_loss_and_accuracy(train_losses, val_losses, val_accuracies, learning_rate, best_scheduler)
                all_models_training_loss.append((train_losses, learning_rate, best_scheduler))
                all_models_val_accuracy.append((val_accuracies, learning_rate, best_scheduler))

    plot_hyperparameter_tuning_results(all_models_training_loss, all_models_val_accuracy)
