import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import json

# load a model pre-trained on COCO
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

num_classes = 1
with open('labels.json', 'r') as f:
    labels = json.load(f)
    num_classes += len(labels.keys())

in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# load the dataset and dataloader
# set to parallel and store the model
# train the model
# test