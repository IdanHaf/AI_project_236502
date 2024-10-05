import torch
import torch.nn as nn
import torchvision
import numpy as np


class LanguageModel:
    def __init__(self, batch_size, transform, model):
        self.batch_size = batch_size
        self.transform = transform

        self.model = model

        self.loss_func = nn.CrossEntropyLoss()

    def foreword_model(self, test_loader, device, handle_func=None):
        """
            Apply the text detection model, and then using the script detection.

            :returns: the model language probability vector predictions
        """