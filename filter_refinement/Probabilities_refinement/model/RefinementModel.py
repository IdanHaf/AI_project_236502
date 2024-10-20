import torch
import torch.nn as nn
import copy
from tqdm import tqdm


class RefinementModel(nn.Module):
    def __init__(self):
        super().__init__()
        # input size = 120 (1st model) + 9 (language model)
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(129, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 120),
        )

    def forward(self, x):
        x = self.linear_relu_stack(x)
        return x


def validation_loop(val_loader, model, device, criterion):
    """
    Train model.
    :param val_loader: validation dataloaders.
    :param model: model to train.
    :param device: device to use.
    :param criterion: loss function to use.

    :return: validation loss and accuracy.
    """
    model.eval()
    running_val_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            val_outputs = model(inputs)
            val_loss = criterion(val_outputs, labels)
            running_val_loss += val_loss.item()

            probabilities = nn.functional.softmax(val_outputs, dim=1)
            _, predicted_labels = torch.max(probabilities, 1)

            correct_predictions += (predicted_labels == labels).sum().item()
            total_predictions += labels.size(0)

    avg_val_loss = running_val_loss / len(val_loader)
    val_accuracy = correct_predictions / total_predictions

    print(f'avg validation loss: {avg_val_loss:.4f}, validation accuracy: {val_accuracy:.4f}')
    return avg_val_loss, val_accuracy


def train_model(dataloaders, model, device, criterion, optimizer, num_epochs, model_name, patience=10):
    """
    Train model.
    :param dataloaders: tuple of train and validation dataloaders.
    :param model: model to train.
    :param device: device to use.
    :param criterion: loss function to use.
    :param optimizer: optimizer to use.
    :param num_epochs: number of epochs.
    :param model_name: name of the saved model.
    :param patience: how many epochs to wait without improvement in validation loss.

    :return: training loss over epochs, validation loss, accuracy over epochs
    """
    train_loader, val_loader = dataloaders
    train_losses = []
    validation_loss_results = []
    validation_accuracy_results = []
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = patience
    best_model_weights = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        running_loss = 0.0
        model.train()

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        print(f'Epoch {epoch + 1}/{num_epochs}, avg training loss: {avg_train_loss:.4f} ')
        epoch_val_loss, epoch_val_accuracy = validation_loop(val_loader, model, device, criterion)
        validation_loss_results.append(epoch_val_loss)
        validation_accuracy_results.append(epoch_val_accuracy)

        # Early stopping according to val loss.
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            patience_counter = patience
        else:
            patience_counter -= 1
            if patience_counter == 0:
                break

    torch.save(best_model_weights, f"{model_name}.pth")
    print(f"Module saved, best epoch: {best_epoch}")

    return train_losses, validation_loss_results, validation_accuracy_results

