import torchvision
import torchvision.models

from classification_city_data_with_model import train_with_model

models = [torchvision.models.resnet50(weights="IMAGENET1K_V1"), torchvision.models.resnet101(weights="IMAGENET1K_V1"),
          torchvision.models.wide_resnet50_2(weights="IMAGENET1K_V1"),
          torchvision.models.wide_resnet101_2(weights="IMAGENET1K_V1"),
            , torchvision.models.vgg11(weights='IMAGENET1K_V1')]
model_names = ['resnet50', 'resnet101', 'wide_resnet50_2', 'wide_resnet101_2', 'vgg11']

for model, model_name in zip(models, model_names):
    train_with_model(model, model_name)