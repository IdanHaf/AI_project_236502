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
from Local_code.customDatasets.city_custom_dataset import CustomImageDataset

# loading bar
from tqdm import tqdm


def train_with_model(model, model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Gpu is available: " + str(torch.cuda.is_available()))

    num_epochs = 20
    batch_size = 64
    learning_rate = 0.05

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset_path = './Images'
    csv_file_path = './city_dataset_labels.csv'
    dataset = CustomImageDataset(csv_file_path, dataset_path, transform)  # Idan Dataset loading

    print(f"Number of samples in the dataset: {len(dataset)}")

    train_size = int(0.6 * len(dataset))
    val_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    generator = torch.Generator()
    generator.manual_seed(387642706252)

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size],
                                                            generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False)

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 23)

    model = nn.DataParallel(model)
    model = model.to(device)

    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    val_losses = []
    val_accuracies = []

    # Training loop.
    for epoch in tqdm(range(num_epochs)):

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

    torch.save(model.state_dict(), f'{model_name}_city_dataset_cls.pth')
    print("Module saved")

    # Test on test_loader.
    model.eval()

    # Plot accuracies
    epochs = range(1, num_epochs + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'{model_name} Loss vs. Epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_accuracies, label='Val Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title(f'{model_name} Accuracy vs. Epochs')
    plt.legend()

    plt.tight_layout()

    plt.savefig(f'{model_name} training_validation_metrics.png')
    plt.close()

    running_test_loss = 0.0
    predicted_labels_list = []
    true_labels_list = []

    print("Calculate accuracy on test.")

    with torch.no_grad():
        images, labels = next(iter(test_loader))
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        probabilities = nn.functional.softmax(outputs, dim=1)
        _, predicted_labels = torch.max(probabilities, 1)

        predicted_labels_list.extend(predicted_labels.cpu().numpy())
        true_labels_list.extend(labels.cpu().numpy())

        loss = loss_func(outputs, labels)
        running_test_loss += loss.item()

    test_loss = running_test_loss / len(test_loader)

    predicted_labels_array = np.array(predicted_labels_list)
    true_labels_array = np.array(true_labels_list)

    accuracy = np.mean(predicted_labels_array == true_labels_array)

    print(predicted_labels_array)

    print(f'Finished test, Loss: {test_loss:.4f}')
    print(f'Accuracy: {accuracy}.')

    print(true_labels_array)
    torch.cuda.empty_cache()

