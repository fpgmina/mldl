import attr
import torch
from torch import nn
from typing import Optional, Dict, Any

@attr.s(frozen=True, kw_only=True)
class TrainingParams():
    """
    A class to store the parameters required for training a model.
    
    Attributes:
        training_name (str): A name for the training experiment.
        epochs (int): The number of epochs for training.
        learning_rate (float): The learning rate for the optimizer.
        architecture (nn.Module): The model architecture to be trained.
        optimizer (torch.optim.Optimizer): The optimizer to be used for training.
        loss_function (nn.Module): The loss function to be used.
        optimizer_params (Optional[Dict[str, Any]]): A dictionary of additional optimizer parameters (optional).
    """
    training_name: str = attr.ib(validator=attr.validators.instance_of(str))
    epochs: int = attr.ib(validator=attr.validators.instance_of(int))
    learning_rate: float = attr.ib(validator=attr.validators.ge(0.))
    architecture: nn.Module = attr.ib(validator=attr.validators.instance_of(nn.Module))
    optimizer: torch.optim.Optimizer = attr.ib(validator=attr.validators.instance_of(torch.optim.Optimizer))
    loss_function: nn.Module = attr.ib(validator=attr.validators.instance_of(nn.Module))
    optimizer_params: Optional[Dict[str, Any]] = attr.ib(default=None)