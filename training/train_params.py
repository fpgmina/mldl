import attr
import torch
from torch import nn
from typing import Optional, Dict, Any
from torch.optim import Optimizer


def is_nn_module(instance, attribute, value):
    """Validator to check if the value is an instance of nn.Module or its subclass."""
    if not isinstance(value, nn.Module):
        raise TypeError(
            f"{attribute.name} must be an instance of nn.Module or its subclass."
        )
    return value


def is_optimizer_class(instance, attribute, value):
    """Validator to check if the value is a subclass of torch.optim.Optimizer."""
    if not issubclass(value, Optimizer):
        raise TypeError(
            f"{attribute.name} must be a subclass of torch.optim.Optimizer."
        )
    return value


@attr.s(frozen=True, kw_only=True)
class TrainingParams:
    """
    A class to store the parameters required for training a model.

    Attributes:
        training_name (str): A name for the training experiment.
        epochs (int): The number of epochs for training.
        learning_rate (float): The learning rate for the optimizer.
        model (nn.Module): The model to be trained.
        optimizer_class (torch.optim.Optimizer): The class of the optimizer to be used for training.
        loss_function (nn.Module): The loss function to be used.
        optimizer_params (Optional[Dict[str, Any]]): A dictionary of additional optimizer parameters (optional).
    """

    training_name: str = attr.ib(validator=attr.validators.instance_of(str))
    epochs: int = attr.ib(validator=attr.validators.instance_of(int))
    learning_rate: float = attr.ib(validator=attr.validators.ge(0.0))
    model: nn.Module = attr.ib(
        validator=is_nn_module
    )  # Custom validation to pass instance check (instance_of checks for exact type and not for superclasses)
    loss_function: nn.Module = attr.ib(validator=is_nn_module)  # Custom validation
    optimizer_class: torch.optim.Optimizer = attr.ib(validator=is_optimizer_class)
    optimizer_params: Optional[Dict[str, Any]] = attr.ib(default=None)

    @property
    def optimizer(self):
        return self.optimizer_class(
            self.model.parameters(), lr=self.learning_rate, **self.optimizer_params
        )
