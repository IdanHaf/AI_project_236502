import numpy as np
import torch
import torchvision
from torch import nn
from torch.utils.data import random_split, DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import v2 as T
import torchvision.transforms as transforms
from torchvision.ops import nms

from acc_util import evaluate_bbox_accuracy
from torchvision_references.engine import train_one_epoch, evaluate
import sign_dataloader
import json

debug = True


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
    global debug

    val_model.eval()  # Set model to evaluation mode
    val_loss = 0
    # Loss functions
    box_loss_fn = nn.SmoothL1Loss()  # For bounding box regression
    class_loss_fn = nn.CrossEntropyLoss()  # For classification
    with torch.no_grad():  # Disable gradient calculation
        for val_images, val_targets in val_dataloader:
            val_images = list(image.to(device) for image in val_images)
            val_targets = [{k: v.to(device) for k, v in t.items()} for t in val_targets]
            # Forward pass
            detections = val_model(val_images)

            for detection, target in zip(detections, val_targets):
                pred_boxes = detection['boxes']
                pred_scores = detection['scores']
                keep_indices = nms(pred_boxes, pred_scores, 0.5)
                pred_boxes = pred_boxes[keep_indices]

                true_boxes = target['boxes']
                true_labels = target['labels']
                if debug:
                    print(f'pred scores are{pred_scores}')
                    print(f'true labels are{true_labels}')
                    debug = False

                # Handle size mismatches
                k = true_boxes.shape[0]
                pred_boxes = pred_boxes[:k]
                if pred_boxes.shape[0] < true_boxes.shape[0]:
                    # Padding with zeros
                    padding = torch.zeros((true_boxes.shape[0] - pred_boxes.shape[0], 4), device=pred_boxes.device)
                    pred_boxes = torch.cat([pred_boxes, padding], dim=0)

                # Box Regression Loss
                # To align with SmoothL1Loss, make sure pred_boxes and true_boxes have the same shape
                if pred_boxes.shape[0] > 0 and true_boxes.shape[0] > 0:
                    box_loss = box_loss_fn(pred_boxes, true_boxes)
                else:
                    box_loss = torch.tensor(0.0, device=device)

                # Classification Loss
                # # Convert predictions and targets to tensors
                # if pred_scores.shape[0] > 0 and true_labels.shape[0] > 0:
                #     print(f'true_labels shape {true_labels}')
                #     print(f'pred_labels shape {pred_scores}')
                #     class_loss = class_loss_fn(pred_scores, true_labels)
                # else:
                #     class_loss = torch.tensor(0.0, device=device)

                # Sum losses
                total_loss = box_loss
                val_loss += total_loss.item()

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
                        shuffle=False, collate_fn=collate_fn)
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
    print(f"Epoch {epoch + 1}/{num_epochs}")

    model.train()
    train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=100)
    # update the learning rate
    lr_scheduler.step()

    # validate
    avg_val_loss = validate(model, val_loader, device)

    torch.cuda.empty_cache()

torch.save(model.state_dict(), "sign_detector.pth")
