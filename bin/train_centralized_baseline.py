import optuna
import torch
import torch.nn as nn

from core.train import train_model
from core.train_params import TrainingParams
from dataset.cifar_100 import (
    get_cifar_100_datasets,
    get_cifar_100_train_valset_datasets,
    get_cifar_dataloader,
)
from models.dino_backbone import get_dino_backbone_model


def objective(trial: optuna.trial.Trial):
    trainset, _ = get_cifar_100_datasets()
    trainset, valset = get_cifar_100_train_valset_datasets(trainset)
    train_dataloader, val_dataloader = get_cifar_dataloader(
        trainset
    ), get_cifar_dataloader(valset)
    model = get_dino_backbone_model()

    momentum = trial.suggest_float("momentum", 0.8, 0.99)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)

    params = TrainingParams(
        training_name=f"centralized_baseline_momentum_{momentum}_wdecay_{weight_decay}_lr_{learning_rate}_cosineLR",
        model=model,
        loss_function=nn.CrossEntropyLoss(),
        learning_rate=learning_rate,
        optimizer_class=torch.optim.SGD,  # type: ignore
        scheduler_class=torch.optim.lr_scheduler.CosineAnnealingLR,  # type: ignore
        epochs=3,
        optimizer_params={"momentum": momentum, "weight_decay": weight_decay},
    )
    best_acc = train_model(
        training_params=params,
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        project_name="fl_centralized_baseline",
    )
    return best_acc


def optimize():
    # create a study object (maximize accuracy)
    study = optuna.create_study(direction="maximize")
    # run the optimization for 10 trials
    study.optimize(objective, n_trials=10)

    print("Best hyperparameters:", study.best_params)
    print("Best accuracy:", study.best_value)


if __name__ == "__main__":
    optimize()
