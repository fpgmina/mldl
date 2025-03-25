import torch
from torch import nn
from dataset.imagenet import get_imagenet_dataloaders
from models.alexnet import AlexNet
from training.train import train_model
from training.train_params import TrainingParams


if __name__ == "__main__":
    # learning_rates = [0.1, 0.001, 0.0001]
    # batch_sizes = [16, 32, 64]

    # # Loop over hyperparameters
    # for lr in learning_rates:
    #     for batch_size in batch_sizes:
    #         # Initialize the network
    #         training_params = TrainingParams(
    #             training_name=f"alexnet_train_lr_{lr}_batchsize_{batch_size}",
    #             epochs=10,
    #             learning_rate=lr,
    #             model=AlexNet(),
    #             optimizer_class=torch.optim.SGD,
    #             loss_function=nn.CrossEntropyLoss(),
    #             optimizer_params={"weight_decay": 5e-4, "momentum": 0.9},
    #         )

    #         train_loader, val_loader = get_imagenet_dataloaders(batch_size=batch_size)
    #         train_model(
    #             train_loader=train_loader,
    #             val_loader=val_loader,
    #             training_params=training_params,
    #         )

    training_params = TrainingParams(
        training_name="alexnet-training_adam",
        epochs=10,
        learning_rate=0.001,
        model=AlexNet(),
        optimizer_class=torch.optim.Adam,
        loss_function=nn.CrossEntropyLoss(),
    )

    train_loader, val_loader = get_imagenet_dataloaders(batch_size=32)
    train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        training_params=training_params,
    )
