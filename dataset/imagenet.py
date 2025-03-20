import os
import shutil
import torch
from torchvision.datasets import ImageFolder
import torchvision.transforms as T


def _download_dataset():

    with open('tiny-imagenet/tiny-imagenet-200/val/val_annotations.txt') as f:
        for line in f:
            fn, cls, *_ = line.split('\t')
            os.makedirs(f'tiny-imagenet/tiny-imagenet-200/val/{cls}', exist_ok=True)

            shutil.copyfile(f'tiny-imagenet/tiny-imagenet-200/val/images/{fn}', f'tiny-imagenet/tiny-imagenet-200/val/{cls}/{fn}')

    shutil.rmtree('tiny-imagenet/tiny-imagenet-200/val/images')


def _get_imagenet_datasets():

    transform = T.Compose([
        T.Resize((224, 224)),  # Resize to fit the input dimensions of the network
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    tiny_imagenet_dataset_train = ImageFolder(root='tiny-imagenet/tiny-imagenet-200/train', transform=transform)
    tiny_imagenet_dataset_val = ImageFolder(root='tiny-imagenet/tiny-imagenet-200/val', transform=transform)
    return tiny_imagenet_dataset_train, tiny_imagenet_dataset_val


def get_imagenet_dataloaders():

    tiny_imagenet_dataset_train, tiny_imagenet_dataset_val = _get_imagenet_datasets()
    train_loader = torch.utils.data.DataLoader(tiny_imagenet_dataset_train, batch_size=64, shuffle=True)
    # shuffle set to True for train while we keep natural order for validation and test
    val_loader = torch.utils.data.DataLoader(tiny_imagenet_dataset_val, batch_size=64, shuffle=False)
    return train_loader, val_loader
