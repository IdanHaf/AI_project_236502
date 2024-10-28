import torch
import torchvision
from lightly.transforms import MoCoV2Transform
from torch import nn
from torchvision.transforms import transforms
import pandas as pd
from embedder import Embedder
from test_custom_dataset import CustomImageDataset

filename = 'sample_df.csv'
model_filename = 'moco.pth'

city_dataset_path = './Images'
city_csv_file_path = './city_images_dataset.csv'
big_dataset_path = './big_dataset'
big_csv_file_path = './big_dataset_labeled.csv'

transform = torchvision.transforms.Compose([transforms.Resize((224, 224)),
                                            MoCoV2Transform(input_size=224, cj_prob=0.2, cj_bright=0.1, cj_contrast=0.1,
                                                            cj_hue=0.1, cj_sat=0.1, min_scale=0.5,
                                                            random_gray_scale=0.0),
                                            ])
device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = CustomImageDataset(city_csv_file_path, city_dataset_path, big_csv_file_path, big_dataset_path, transform)

if __name__ == '__main__':
    baseset = pd.read_csv(filename)
    resnet = torchvision.models.resnet18()
    net = nn.Sequential(*list(resnet.children())[:-1])
    embedder = Embedder(net)
    embedder.load_csv(model_filename)
    embedder.to(device)
    result = []
    for _, row in baseset.iterrows():
        idx = row['id']
        img, _ = dataset[idx][0]
        img = img.unsqueeze(0)
        img = img.to(device)
        with torch.no_grad():
            q = embedder(img).cpu().detach()
        result.append([idx, row['lat'], row['lng'], row['label'], q])

    result_df = pd.DataFrame(result, columns=['id', 'lat', 'lng', 'label', 'query'])
    result_df.to_csv('baseset.csv', index=False)
