import numpy as np
import torch
import torch.utils.data
from torch import nn
from torch.utils.data import Subset, DataLoader

from training.train import train_model
from training.train_params import TrainingParams


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def check_forward_pass(model, train_loader, num_classes=200):
    device = get_device()
    sample_batch, _ = next(iter(train_loader))  # Get one batch
    sample_batch = sample_batch.to(device)  # If you're using a GPU

    output = model(sample_batch)
    assert output.shape == (train_loader.batch_size, num_classes), "Forward Pass Failed"
    print("Forward Pass works!")


def get_subset_loader(dataloader, subset_size=1000) -> torch.utils.data.DataLoader:
    dataset_size = len(dataloader.dataset)
    subset_indices = np.random.choice(dataset_size, subset_size, replace=False)
    subset_dataset = Subset(dataloader.dataset, subset_indices)
    subset_loader = DataLoader(subset_dataset, batch_size=32, shuffle=True)
    return subset_loader


def train_on_subset(training_params, train_loader, val_loader=None, epochs=2, **kwargs):
    assert isinstance(training_params, TrainingParams)
    training_params_subset = TrainingParams(
        **{
            **training_params.__dict__,
            **{
                "training_name": f"{training_params.training_name}_SUBSET",
                "epochs": epochs,
            },
        }
    )
    train_loader_subset = get_subset_loader(train_loader)
    train_model(
        training_params=training_params_subset,
        train_loader=train_loader_subset,
        val_loader=val_loader,
        **kwargs,
    )
    print("Finished training on subset.")


def get_model_from_checkpoint(model: nn.Module, checkpoint_path: str) -> nn.Module:
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint)
    return model
