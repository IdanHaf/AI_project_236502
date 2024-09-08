import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from Sign_detector import SignDetector
from Local_code.customDatasets.custom_dataset import CustomImageDataset


transform = transforms.Compose([
    transforms.Resize((800, 800)),
    transforms.ToTensor(),
])
batch_size = 4

city_dataset_path = './Images'
city_csv_file_path = './city_images_dataset.csv'
big_dataset_path = './results'
big_csv_file_path = './big_dataset_labeled.csv'
dataset = CustomImageDataset(city_csv_file_path, city_dataset_path, big_csv_file_path, big_dataset_path, transform)
generator = torch.Generator()
generator.manual_seed(387642706252)

loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

detector = SignDetector()
detector.export_sign(loader)