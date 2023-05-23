import torch
import os
from torchvision.utils import make_grid
from torchvision import transforms, datasets
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader


def plot_image(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(
        torch.cat(
            [torch.cat([i for i in images.cpu()], dim=1)],
            dim=2)
        ).permute(1, 2, 0).cpu()
    
    plt.show()
    

def save_images(images, path, **kwargs):
    grid = make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    img = Image.fromarray(ndarr)
    img.save(path)
    
def get_data(args):
    transforms.Compose([
        #transforms.Resize(80),
        #transforms.RandomResizedCrop(args.images_size, scale=(0.8, 1.0), ratio=(1.0, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(*(0.5,)*args.in_ch, *(0.5,)*args.in_ch),
    ])
    #dataset = datasets.ImageFolder(args.dataset_path, transform=transforms)
    dataset = datasets.FashionMNIST(args.dataset_path, transform=transforms, download=True)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    return dataloader

def load_data(args):
    transform_data = transforms.Compose(
    [transforms.ToTensor(), transforms.Lambda(lambda x: (x - 0.5) * 2.0)])
    loader = DataLoader(
        datasets.FashionMNIST("data", train=True, download=True, transform=transform_data),
        batch_size=args.batch_size,
        shuffle=True,
    )
    return loader

def setup_logging(run_name):
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    os.makedirs(os.path.join('models', run_name), exist_ok=True)
    os.makedirs(os.path.join('results', run_name), exist_ok=True)
    
    