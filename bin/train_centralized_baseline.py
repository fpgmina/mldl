import torch
import torch.nn as nn

from core.train_params import TrainingParams
from dataset.cifar_100 import (
    get_cifar_100_datasets,
    get_cifar_100_train_valset_datasets,
    get_cifar_dataloader,
)
from models.dino_backbone import get_dino_backbone_model

if __name__ == "__main__":
    trainset, _ = get_cifar_100_datasets()
    trainset, valset = get_cifar_100_train_valset_datasets(trainset)
    train_dataloader, val_dataloader = get_cifar_dataloader(
        trainset
    ), get_cifar_dataloader(valset)
    model = get_dino_backbone_model()

    params = TrainingParams(
        training_name="centralized_baseline",
        model=model,
        loss_function=nn.CrossEntropyLoss(),
        learning_rate=1e-3,
        optimizer_class=torch.optim.Adam,
        epochs=10,
    )
