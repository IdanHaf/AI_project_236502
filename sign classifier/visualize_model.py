import os

import torch
import torchvision
from PIL import Image, ImageDraw

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms as transforms

import json


images = [file for file in os.listdir("images")]


image_list = []
for file in images:
    image_path = os.path.join("images", file)

    image = Image.open(image_path)
    image_list.append(image)

images = image_list

sizes = [image.size for image in images]
print(sizes[0])
transform = transforms.Compose([
    transforms.Resize((800, 800)),
    transforms.ToTensor(),
])

images = [transform(image) for image in images]
num_classes = 0

model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights="DEFAULT")
with open('labels.json', 'r') as f:
    labels = json.load(f)
    num_classes += len(labels.keys())

in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

model.load_state_dict(torch.load('sign_detector.pth', map_location=torch.device('cpu')))
model.eval()

with torch.no_grad():  # No need to calculate gradients for inference
    predictions = model(images)

image = transforms.ToPILImage()(images[1])
draw = ImageDraw.Draw(image)

# Draw rectangles and scores
pred = predictions[1]
boxes = pred['boxes'].tolist()
scores = pred['scores'].tolist()

print(boxes)

rects = Image.new('RGBA', image.size)
rects_draw = ImageDraw.Draw(rects)

for box, score in zip(boxes, scores):
    print(box)
    width_ratio, height_ratio = 1, 1
    x1, y1, y2, x2 = box # Convert coordinates to integers
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    rects_draw.rectangle((x1 + 1, y1 + 1, x2 - 1, y2 - 1))
    print(f'x1 {x1} y1 {y1} x2 {x2} y2 {y2}')

    draw.line(((x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)), fill='black', width=1)
    draw.text((x1, y1), f"{score:.2f}", fill="red")
image.show()
