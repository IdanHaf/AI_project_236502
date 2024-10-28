import copy

import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision
from torch import nn

from lightly.models.modules import MoCoProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.transforms.moco_transform import MoCoV2Transform
from lightly.utils.scheduler import cosine_schedule
import lightly.data as data
from torch.utils.data import random_split, DataLoader
from torchvision import transforms

from sklearn.metrics.pairwise import haversine_distances
from moco.test_custom_dataset import CustomImageDataset


# Inspired by https://docs.lightly.ai/self-supervised-learning/tutorials/package/tutorial_moco_memory_bank.html
# The code assumes that it runs on multiple GPU and that model is dataParallel object
class Embedder(nn.Module):
    def __init__(self, backbone):
        super().__init__()

        self.backbone = backbone
        self.projection_head = MoCoProjectionHead(512, 512, 128)

        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)

        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)
        self.K = 4096

        self.queue = nn.functional.normalize(torch.randn(128, self.K), dim=0)
        self.queue_cords = torch.full((self.K, 2), 1000)  # max distance is 500 so it will be negative examples

    def forward(self, x, coords=None):
        """
        The forward pass of the model
        :param x: The input images
        :param coords: The coordinate of the images
        :return: The output query
        """
        _query = self.backbone(x).flatten(start_dim=1)
        _query = self.projection_head(_query)

        if coords is None:
            return _query

        # update queue
        with torch.no_grad():
            k = self.forward_momentum(x)
            k = nn.functional.normalize(k, dim=1)

            self.queue = self.queue.to(k.device)
            self.queue = torch.cat((k.T, self.queue), dim=1)[:, :self.K]
            self.queue_cords = self.queue_cords.to(k.device)
            self.queue_cords = torch.cat((coords, self.queue_cords))[:self.K]

        return _query

    def forward_momentum(self, x):
        """
        Extract the key of x
        :param x: The image
        :return: The output key
        """
        _key = self.backbone_momentum(x).flatten(start_dim=1)
        _key = self.projection_head_momentum(_key).detach()
        return _key

    def load_csv(self, filename):
        """
        Load weights from pre-existing file
        :param filename: The filename of the weights
        """

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        state_dict = torch.load(filename, map_location=device)

        new_state_dict = {}
        for key in state_dict:
            new_key = key.replace('module.', '')  # Remove 'module.' prefix
            new_state_dict[new_key] = state_dict[key]
        self.load_state_dict(new_state_dict)


def weighted_loss(q, queue, coord, queue_coords, t=0.07, max_dist=500):
    """
    Calculate the loss on the current image
    :param q: The query image
    :param queue: The current queue of the model
    :param coord: The coordinate of the image
    :param queue_coords: The coordinate of the queue images.
    :param t: the temperature
    :param max_dist: Max distance for calculation
    :return: The loss values
    """
    queue = queue.to(q.device)
    queue_coords = queue_coords.to(q.device)
    logits = torch.matmul(q, queue.clone().detach()) / t

    if torch.isnan(coords).any():
        print("NaN detected in coords tensor:", coords)

    if torch.isnan(queue_coords).any():
        print("NaN detected in queue_coords tensor:", queue_coords)

    weights = torch.tensor(haversine_distances(coord.cpu().numpy(), queue_coords.cpu().numpy()), device=logits.device)
    weights = torch.max(1 - (weights / max_dist), torch.zeros_like(weights))

    return -torch.log((torch.exp(logits) * weights) / (torch.exp(logits) * (1 - weights)).sum(1, keepdim=True)).mean()


def validate(validation_loader, moco_model, criterion):
    """
    Validation run on the model
    :param validation_loader: The validation loader
    :param moco_model: The moco model
    :param criterion: The loss function
    :return: The average loss of the model
    """
    moco_model.eval()
    total_l = 0
    for batch in train_loader:
        images = batch[0]
        labels, lat, lng = batch[1]
        with torch.no_grad():
            crds = torch.stack((lat, lng), dim=1)
            x_q, _ = images

            x_q = x_q.to(device)
            q = moco_model(x_q, coords)
            v_loss = criterion(q, moco_model.module.queue, crds, moco_model.module.queue_cords)
            total_l += v_loss.detach()
    moco_model.train()
    return total_l / len(validation_loader)


if __name__ == '__main__':
    print("train")
    city_dataset_path = './Images'
    city_csv_file_path = './city_images_dataset.csv'
    big_dataset_path = './big_dataset'
    big_csv_file_path = './big_dataset_labeled.csv'

    transform = torchvision.transforms.Compose([transforms.Resize((224, 224)),
                                                MoCoV2Transform(input_size=224, cj_prob=0.2, cj_bright=0.1,
                                                                cj_contrast=0.1,
                                                                cj_hue=0.1, cj_sat=0.1, min_scale=0.5,
                                                                random_gray_scale=0.0),
                                                ])

    dataset = CustomImageDataset(city_csv_file_path, city_dataset_path, big_csv_file_path, big_dataset_path, transform)

    train_size = int(0.6 * len(dataset))
    val_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    generator = torch.Generator()
    generator.manual_seed(387642706252)
    dataset = data.dataset.LightlyDataset.from_torch_dataset(dataset)

    # Splitting the data.
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size],
                                                            generator=generator)
    batch_size = 128
    workers = 2
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False)
    flag = True
    for lr in [0.001, 0.0001]:
        resnet = torchvision.models.resnet18()
        net = nn.Sequential(*list(resnet.children())[:-1])
        model = Embedder(net)

        if torch.cuda.device_count() > 1:
            print(f'Using {torch.cuda.device_count()} GPUs!')
            model = nn.DataParallel(model)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)

        epochs = 30
        min_loss = float('inf')
        train_losses = []
        val_losses = []
        print("Starting Training")
        for epoch in range(epochs):
            total_loss = 0
            momentum_val = cosine_schedule(epoch, epochs, 0.996, 1)
            for batch in train_loader:
                images = batch[0]
                labels, lat, lng = batch[1]
                coords = torch.stack((lat, lng), dim=1)
                x_query, _ = images

                update_momentum(model.module.backbone, model.module.backbone_momentum, m=momentum_val)
                update_momentum(
                    model.module.projection_head, model.module.projection_head_momentum, m=momentum_val
                )
                x_query = x_query.to(device)
                query = model(x_query, coords)
                loss = weighted_loss(query, model.module.queue, coords, model.module.queue_cords)
                if flag:
                    print(f'coords are {coords}')
                    flag = False
                total_loss += loss.detach()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            avg_loss = total_loss / len(train_loader)
            train_losses.append(avg_loss)
            val_loss = validate(val_loader, model, weighted_loss)
            if val_loss <= min_loss:
                torch.save(model.state_dict(), f'model_{epoch}_loss_{avg_loss}_{lr}.pth')

            print(f"epoch: {epoch:>02}, loss: {avg_loss:.5f} lr: {lr}")
            val_losses.append(val_loss)

        epochs_a = np.arange(epochs)
        plt.plot(epochs_a, train_losses, label=f'train loss {lr}')
        plt.plot(epochs_a, val_losses, label=f'val loss {lr}')

        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title(f"train moco model with lr {lr}")
        plt.grid(True)
        plt.legend()
        plt.savefig(f'moco_train_{lr}.png')
        plt.show()

        print(f'test loss: {validate(test_loader, model, weighted_loss)} lr {lr}')
