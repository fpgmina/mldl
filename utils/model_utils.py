import numpy as np
import torch
import torch.utils.data
from torch import nn
from torch.utils.data import Subset, DataLoader


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def check_forward_pass(model, dataloader, num_classes=200):
    device = get_device()
    sample_batch, _ = next(iter(dataloader))  # Get one batch
    sample_batch = sample_batch.to(device)

    output = model(sample_batch)
    assert output.shape == (dataloader.batch_size, num_classes), "Forward Pass Failed"
    print("Forward Pass works!")


def get_subset_loader(dataloader, subset_size=1000) -> torch.utils.data.DataLoader:
    dataset_size = len(dataloader.dataset)
    subset_indices = np.random.choice(dataset_size, subset_size, replace=False)
    subset_dataset = Subset(dataloader.dataset, subset_indices)
    subset_loader = DataLoader(subset_dataset, batch_size=32, shuffle=True)
    return subset_loader


def get_model_from_checkpoint(model: nn.Module, checkpoint_path: str) -> nn.Module:
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint)
    return model
