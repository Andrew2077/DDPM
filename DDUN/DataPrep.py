from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import numpy as np


def Fashion_Mnist_loader(data_path, batch_size, shuff, drop_last):
    input_transformation = transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda x: (x * 2) - 1)]
    )
    data = datasets.FashionMNIST(
        root=data_path,
        train=True,
        download=True,
        transform=input_transformation,
    )

    loader = DataLoader(
        data,
        batch_size=batch_size,
        shuffle=shuff,
        drop_last=drop_last,
    )
    return loader



def prep_imshow(img):
    transformations = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        # transforms.Lambda(lambda t: t.permute(1, 2, 0)),  # CHW to HWC
        transforms.Lambda(lambda t: t * 255.0),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])
    return transformations(img)