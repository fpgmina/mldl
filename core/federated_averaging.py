import enum
import logging

import attr
import torch
from typing import List, Dict, Optional, Tuple
from torch.utils.data import Dataset
import torch.nn as nn
import random
import copy

from tqdm import tqdm

from core.train import train_model
from core.train_params import TrainingParams
from dataset.cifar_100 import get_dataloader
from utils.model_utils import iid_sharding, non_iid_sharding
from utils.numpy_utils import numpy_random_seed


class ShardingType(enum.Enum):
    IID = enum.auto()
    NON_IID = enum.auto()


def federated_averaging(
    global_model: torch.nn.Module, client_models: List[torch.nn.Module]
) -> torch.nn.Module:
    """
    Perform Federated Averaging on the given list of client models to update the global model.

    The function averages the weights of the provided client models and updates the global model.

    Args:
    - global_model (torch.nn.Module): The global model that will be updated.
    - client_models (List[torch.nn.Module]): A list of client models whose weights will be averaged.

    Returns:
    - torch.nn.Module: The updated global model with averaged weights.

    Example:
    >>> updated_global_model = federated_averaging(global_model, client_models)
    """
    global_dict = global_model.state_dict()

    for key in global_dict:
        # Compute the average of the corresponding parameter across all client models
        global_dict[key] = torch.mean(
            torch.stack(
                [
                    client_model.state_dict()[key].float()
                    for client_model in client_models
                ],
                dim=0,
            ),
            dim=0,
        )

    # Load the averaged weights into the global model
    global_model.load_state_dict(global_dict)

    return global_model


@attr.s
class FederatedAveraging:
    global_model: torch.nn.Module = attr.ib()
    trainset: Dataset = attr.ib()
    valset: Dataset = attr.ib()
    sharding_type: ShardingType = attr.ib()
    _num_clients: int = attr.ib(default=100)  # number of clients
    _client_fraction: float = attr.ib(
        default=0.1
    )  # percentage of clients to select for training
    _rounds: int = attr.ib(
        default=10
    )  # the number of times the central server communicates with the client devices
    _num_classes: Optional[int] = attr.ib(
        default=None
    )  # num classes for non iid sharing
    _seed: int = attr.ib(default=42)
    _training_session_name: str = attr.ib(default="fl")
    _lr: float = attr.ib(default=1e-3)
    _momentum: float = attr.ib(default=0.9)
    _weight_decay: float = attr.ib(default=5e-4)

    def __attrs_post_init__(self):
        if self.sharding_type == ShardingType.NON_IID:
            assert (
                self._num_classes is not None
            ), f"num_classes cannot be None for {self.sharding_type.name}"

    def get_client_data(
        self, seed: int
    ) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
        if self.sharding_type == ShardingType.IID:
            train_client_data = iid_sharding(
                self.trainset, self._num_clients, seed=seed
            )
            val_client_data = iid_sharding(self.valset, self._num_clients, seed=seed)
        elif self.sharding_type == ShardingType.NON_IID:
            train_client_data = non_iid_sharding(
                self.trainset, self._num_clients, self._num_classes, seed=seed
            )
            val_client_data = non_iid_sharding(
                self.valset, self._num_clients, self._num_classes, seed=seed
            )
        else:
            raise NotImplementedError()
        return train_client_data, val_client_data

    def get_training_params(
        self, model: nn.Module, round: int, client_id: int
    ) -> TrainingParams:
        params = TrainingParams(
            training_name=f"fl_round_{round}_client_{client_id}",
            model=model,
            loss_function=nn.CrossEntropyLoss(),
            learning_rate=self._lr,
            optimizer_class=torch.optim.SGD,  # type: ignore
            scheduler_class=torch.optim.lr_scheduler.CosineAnnealingLR,  # type: ignore
            epochs=5,
            optimizer_params={
                "momentum": self._momentum,
                "weight_decay": self._weight_decay,
            },
            scheduler_params={"T_max": 10},
        )
        return params

    def train(self) -> None:
        """
        Run the Federated Averaging training process for the specified number of rounds.
        """
        for i in tqdm(range(self._rounds)):
            logging.info(f"Training round {i} of {self._rounds}:")
            # Select a subset of clients (client_fraction * num_clients)
            seed = self._seed + i  # update seed, for reproducibility in the loop
            with numpy_random_seed(seed=seed):
                selected_clients = random.sample(
                    range(self._num_clients),
                    int(self._client_fraction * self._num_clients),
                )
            logging.info(
                f"Selected {len(selected_clients)} clients: {selected_clients}"
            )
            client_models = []

            train_client_data_idx, val_client_data_idx = self.get_client_data(seed=seed)
            # Train the selected clients locally (simulated by local training process)
            for client_id in tqdm(selected_clients):
                local_model = copy.deepcopy(self.global_model)
                train_loader = get_dataloader(
                    self.trainset, train_client_data_idx[client_id], shuffle=True
                )
                val_loader = get_dataloader(
                    self.valset, val_client_data_idx[client_id], shuffle=False
                )
                training_params = self.get_training_params(
                    model=local_model, round=i, client_id=client_id
                )
                res_dict = train_model(
                    training_params=training_params,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    project_name=self._training_session_name
                    + f"_mom_{self._momentum:.2f}_decay_{self._weight_decay:.2f}_lr_{self._lr:.2f}",
                )
                logging.info(f"Best accuracy: {res_dict['best_accuracy']}")
                client_models.append(res_dict["model"])

            # Perform Federated Averaging to update the global model
            self.global_model = federated_averaging(self.global_model, client_models)

        # Evaluate the global model (add evaluation logic here if needed)

        logging.info("FL Training completed.")
