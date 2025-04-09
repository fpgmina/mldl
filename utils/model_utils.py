from typing import Optional
from pathlib import Path
import os
import numpy as np
import torch
import torch.utils.data
from torch import nn
from torch.utils.data import Subset, DataLoader


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
    dataset_size = len(dataloader.dataset)
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
