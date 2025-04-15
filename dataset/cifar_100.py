from typing import Dict, List, Tuple

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
from collections import defaultdict


def get_cifar_100_transform():
    transform = transforms.Compose(
        [
            transforms.Resize(
                224
            ),  # resize to 224 x 224 (required by ViT) # TODO clarify
            transforms.ToTensor(),
            transforms.Normalize([0.5071, 0.4865, 0.4409], [0.2673, 0.2564, 0.2762]),
        ]
    )
    return (
        transform  # TODO are we allowed to use mean/std for cifar_100 normalization??
    )


def get_cifar_100_datasets():
    transform = get_cifar_100_transform()
    trainset = torchvision.datasets.CIFAR100(
        root="./data", train=True, download=True, transform=transform
    )
    testset = torchvision.datasets.CIFAR100(
        root="./data", train=False, download=True, transform=transform
    )
    return trainset, testset


def get_cifar_100_train_valset_datasets(dataset: Dataset) -> Tuple[Dataset, Dataset]:
    train_size = int(0.8 * len(dataset))  # type: ignore # 80pc train, 20pc validation
    val_size = len(dataset) - train_size  # type: ignore
    trainset, valset = random_split(dataset, [train_size, val_size])
    return trainset, valset


def iid_sharding(dataset: Dataset, num_clients: int) -> Dict[int, List[int]]:
    # Split the dataset into K equal parts, each with samples from all classes
    indices = np.random.permutation(len(dataset))  # type: ignore
    client_data = defaultdict(list)

    for i in range(len(dataset)):  # type: ignore
        client_id = i % num_clients
        client_data[client_id].append(indices[i])

    return client_data


def non_iid_sharding(
    dataset: Dataset, num_clients: int, num_classes: int
) -> Dict[int, List[int]]:
    # Split the dataset into K parts with non-i.i.d. distribution
    client_data = defaultdict(list)
    class_indices = defaultdict(list)

    # Group data points by class
    for idx, (_, label) in enumerate(dataset):  # type: ignore
        class_indices[label].append(idx)

    # Distribute classes to clients
    classes = list(class_indices.keys())
    np.random.shuffle(classes)

    class_per_client = num_classes  # Number of classes per client

    for i in range(num_clients):
        selected_classes = classes[i * class_per_client : (i + 1) * class_per_client]
        for c in selected_classes:
            client_data[i].extend(class_indices[c])

    return client_data


def get_cifar_dataloader(
    dataset: Dataset,
    indices: List[int] = None,
    batch_size: int = 32,
    shuffle: bool = True,
) -> DataLoader:
    """
    Return a DataLoader for the CIFAR dataset, optionally using a subset of the data.

    Args:
        dataset (Dataset): The CIFAR dataset to be loaded.
        indices (List[int], optional): A list of indices to create a subset of the dataset. If None, the entire dataset is used. Defaults to None.
        batch_size (int, optional): The number of samples per batch. Defaults to 32.
        shuffle (bool, optional): Whether to shuffle the data at the beginning of each epoch. Defaults to True.
            NB Should be set to False for test dataset and to True for train/val datasets.

    Returns:
        DataLoader: A DataLoader for the CIFAR dataset, potentially using a subset of the data.
    """
    if indices is not None:
        dataset = torch.utils.data.Subset(dataset, indices=indices)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader


if __name__ == "__main__":

    K = 10  # Number of clients
    num_classes = 2  # Number of classes per client (non iid)

    train_set, _ = get_cifar_100_datasets()
    trainset, valset = get_cifar_100_train_valset_split(train_set)

    iid_client_data = iid_sharding(trainset, K)
    non_iid_client_data = non_iid_sharding(trainset, K, num_classes)

    client_0_iid_loader = get_cifar_dataloader(
        trainset, iid_client_data[0], shuffle=True
    )
    client_0_non_iid_data = get_cifar_dataloader(
        trainset, non_iid_client_data[0], shuffle=True
    )
    val_loader = DataLoader(valset, batch_size=32, shuffle=False)
