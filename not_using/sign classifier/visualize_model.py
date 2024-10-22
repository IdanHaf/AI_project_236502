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

transform = transforms.Compose([
    transforms.Resize((800, 800)),
    transforms.ToTensor(),
])

images = [transform(image) for image in images]

num_classes = 2
model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(pretrained=True)


in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

model.load_state_dict(torch.load('sign_detector.pth', map_location=torch.device('cpu')))
model.eval()

with torch.no_grad():  # No need to calculate gradients for inference
    predictions = model(images)
print(predictions)

sign_dictionary = {}
for i in range(len(images)):
    image = transforms.ToPILImage()(images[i])
    draw = ImageDraw.Draw(image)
    signs = []
    # Draw rectangles and scores
    pred = predictions[i]
    boxes = pred['boxes'].tolist()
    scores = pred['scores'].tolist()

    rects = Image.new('RGBA', image.size)
    rects_draw = ImageDraw.Draw(rects)

    for box, score in zip(boxes, scores):
        if score < 0.5:
            continue

        width_ratio = 1
        height_ratio = 1
        x1, y1, x2, y2 = box # Convert coordinates to integers
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        x1 *= width_ratio
        x2 *= width_ratio
        y1 *= height_ratio
        y2 *= height_ratio

        sign_image = images[i][:, int(y1):int(y2), int(x1): int(x2)]
        signs.append(sign_image.clone())

        rects_draw.rectangle((x1 + 1, y1 + 1, x2 - 1, y2 - 1))

        draw.line(((x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)), fill='green', width=1)
        draw.text((x1, y1), f"{score:.2f}", fill="red")
    sign_dictionary[i] = signs
    image.show()
print(sign_dictionary)
torch.save(sign_dictionary, 'sign_dictionary.pth')
