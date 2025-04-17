from collections import defaultdict
from typing import Optional, Dict, List
from pathlib import Path
import os
import numpy as np
import torch
import torch.utils.data
from torch import nn
from torch.utils.data import Subset, DataLoader, Dataset

from utils.numpy_utils import numpy_random_seed


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def check_forward_pass(
    model: nn.Module, dataloader: torch.utils.data.DataLoader, num_classes: int = 200
):
    device = get_device()
    sample_batch, _ = next(iter(dataloader))  # Get one batch
    sample_batch = sample_batch.to(device)

    output = model(sample_batch)
    assert output.shape == (dataloader.batch_size, num_classes), "Forward Pass Failed"
    print("Forward Pass works!")


def get_subset_loader(
    dataloader: torch.utils.data.DataLoader, subset_size: int = 1000
) -> torch.utils.data.DataLoader:
    dataset_size = len(dataloader.dataset)  # type: ignore
    subset_indices = np.random.choice(dataset_size, subset_size, replace=False)
    subset_dataset = Subset(dataloader.dataset, subset_indices)
    subset_loader = DataLoader(subset_dataset, batch_size=32, shuffle=True)
    return subset_loader


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    filename: Optional[str] = None,
):

    # Check if running in Colab
    is_colab = os.path.exists("/content/drive")

    # If not running on Colab, use the local directory or default
    checkpoint_dir = (
        Path("/content/drive/MyDrive/checkpoints/")
        if is_colab
        else Path("./checkpoints/")
    )

    checkpoint_dir = Path(checkpoint_dir) or Path("/content/drive/MyDrive/checkpoints/")
    filename = filename or "model_checkpoint.pth"

    if is_colab:
        from google.colab import drive

        drive.mount("/content/drive")

    assert checkpoint_dir.exists()

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }
    torch.save(checkpoint, checkpoint_dir / filename)


def load_checkpoint(
    model: nn.Module, optimizer: torch.optim.Optimizer, checkpoint_path: str
):
    assert Path(checkpoint_path).exists()
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    return model, optimizer, epoch, loss


def iid_sharding(
    dataset: Dataset, num_clients: int, seed: Optional[int] = 42
) -> Dict[int, List[int]]:
    # Split the dataset into num_clients equal parts, each with samples from all classes
    data_len = len(dataset)  # type: ignore
    with numpy_random_seed(seed):
        indices = np.random.permutation(data_len)
    client_data = defaultdict(list)

    for i in range(data_len):
        client_id = i % num_clients
        client_data[client_id].append(indices[i])

    return client_data


def non_iid_sharding(
    dataset: Dataset,
    num_clients: int,
    num_classes: int,
    seed: Optional[int] = 42,
) -> Dict[int, List[int]]:
    # Split the dataset into K parts with non-i.i.d. distribution
    client_data = defaultdict(list)
    class_indices = defaultdict(list)

    # Group data points by class
    for idx, (_, label) in enumerate(dataset):  # type: ignore
        class_indices[label].append(idx)

    # Distribute classes to clients
    classes = list(class_indices.keys())
    with numpy_random_seed(seed):
        np.random.shuffle(classes)

    class_per_client = num_classes  # Number of classes per client

    for i in range(num_clients):
        selected_classes = classes[i * class_per_client : (i + 1) * class_per_client]
        for c in selected_classes:
            client_data[i].extend(class_indices[c])

    return client_data
