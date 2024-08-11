import numpy as np
import torch
import torchvision
from torch import nn
from torch.utils.data import random_split, DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import v2 as T
import torchvision.transforms as transforms

from acc_util import evaluate_bbox_accuracy
import sign_dataloader
import json


# inspired by https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

def collate_fn(batch):
    images = []
    targets = []

    for image, target in batch:
        images.append(image)
        targets.append(target)

    # `images` is a list of tensors of shape [3, H, W]
    # `targets` is a list of dictionaries, each containing boxes and labels
    return images, targets

def get_transform(train):
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)


def validate(val_model, val_dataloader, device):
    val_model.eval()  # Set model to evaluation mode
    val_loss = 0
    with torch.no_grad():  # Disable gradient calculation
        for val_images, val_targets in val_dataloader:
            val_images = list(image.to(device) for image in val_images)
            val_targets = [{k: v.to(device) for k, v in t.items()} for t in val_targets]

            # Forward pass
            val_loss_dict = val_model(val_images, val_targets)
            val_losses = sum(loss for loss in val_loss_dict.values())
            val_loss += val_losses.item()

    avg_val_loss = val_loss / len(val_dataloader)
    print(f"Validation Loss: {avg_val_loss}")
    return avg_val_loss


num_epochs = 30
batch_size = 16
learning_rate = 0.05

transform = transforms.Compose([
        transforms.Resize((800, 800)),
        transforms.ToTensor(),
    ])

generator = torch.Generator()
generator.manual_seed(387642706252)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Gpu is available: " + str(torch.cuda.is_available()))

# load a model pre-trained on COCO
model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights="DEFAULT")

num_classes = 1
with open('labels.json', 'r') as f:
    labels = json.load(f)
    num_classes += len(labels.keys())

in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# load the dataset and dataloader
dataset = sign_dataloader.SignImageDataset(transform=transform)

# define the splits
train_size = int(0.6 * len(dataset))
val_size = int(0.2 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size], generator=generator)

# loader splits
val_loader = DataLoader(val_dataset, batch_size=batch_size,
                        shuffle=True, collate_fn=collate_fn)
train_loader = DataLoader(train_dataset, batch_size=batch_size,
                          shuffle=True, num_workers=2, pin_memory=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size,
                         shuffle=False, collate_fn=collate_fn)

# set to parallel and store the model
model = model.to(device)


params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(
    params,
    lr=learning_rate,
    momentum=0.9,
    weight_decay=0.0005
)

# and a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=3,
    gamma=0.1
)

losses = {}
# train the model
for epoch in range(num_epochs):
    model.train()
    for images, targets in train_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Forward pass
        loss_dict = model(images, targets)
        del images
        del targets
        torch.cuda.empty_cache()

        # Compute total loss
        losses = sum(loss for loss in loss_dict.values())

        # Backward pass and optimization
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

    # Update the learning rate
    lr_scheduler.step()

    # validate
    avg_val_loss = validate(model, val_loader, device)

    print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {losses.item()}, Validation Loss: {avg_val_loss}")

torch.save(model.state_dict(), "sign_detector.pth")

# test
model.eval()

print("Calculate accuracy on test.")
label_acc = []
bbox_acc = []
for images, targets in test_loader:
    images = list(image.to(device) for image in images)
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

    # Forward pass
    predictions = model(images)
    del images
    del targets
    for prediction in predictions:
        predicted_labels_list = []
        true_labels_list = []

        predicted_bboxes = []
        true_bboxes = []

        predicted_boxes = prediction['boxes'].cpu().numpy()
        predicted_labels = prediction['labels'].cpu().numpy()
        predicted_scores = prediction['scores'].cpu().numpy()

        score_threshold = 0.5
        filtered_boxes = predicted_boxes[predicted_scores > score_threshold]
        filtered_labels = predicted_labels[predicted_scores > score_threshold]
        filtered_scores = predicted_scores[predicted_scores > score_threshold]

        predicted_labels_list.extend(filtered_labels)
        true_labels_list.extend(targets['labels'])

        predicted_bboxes.extend(filtered_boxes)
        true_bboxes.extend(targets['boxes'])
        acc = evaluate_bbox_accuracy(predicted_labels_list, true_labels_list, predicted_bboxes, true_bboxes)
        label_acc.append(acc[0])
        bbox_acc.append(acc[1])

label_acc = np.array(label_acc)
bbox_acc = np.array(bbox_acc)
print(f'The label accuracy {np.mean(label_acc)}, The bbox fitting accuracy {np.mean(bbox_acc)}')
# find a way to define accuracy
torch.cuda.empty_cache()
