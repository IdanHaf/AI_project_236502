import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


class SignDetector:
    def __init__(self, transform=None):
        num_classes = 401
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(pretrained=True)

        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        model.load_state_dict(torch.load('sign_detector_full_labels.pth'))
        model.eval()
        self.model = model
        self.transform = transform

    def detect(self, images):
        self.model.eval()
        if self.transform is not None:
            images = self.transform(images)
        with torch.no_grad():  # No need to calculate gradients for inference
            predictions = self.model(images)
        return predictions

    def export_sign(self, dataloader):
        sign_dictionary = {}
        i = 0
        for images, targets in dataloader:
            predictions = self.detect(images)

            for idx in range(len(images)):
                pred = predictions[idx]
                boxes = pred['boxes'].tolist()
                scores = pred['scores'].tolist()

                signs = []
                for box, score in zip(boxes, scores):
                    # Save the tensors of each image
                    if score < 0.5:
                        continue

                    x1, y1, x2, y2 = box  # Convert coordinates to integers
                    x1, x2 = min(x1, x2), max(x1, x2)
                    y1, y2 = min(y1, y2), max(y1, y2)

                    sign_image = images[i][:, int(y1):int(y2), int(x1): int(x2)]
                    signs.append(sign_image.clone())
                sign_dictionary[i + idx] = signs

            i += len(images)

        torch.save(sign_dictionary, 'sign_dictionary.pth')

