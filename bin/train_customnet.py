import torch
from torch import nn
from dataset.imagenet import get_imagenet_dataloaders
from models.cnn import CustomNet
from training.train import train_model
from training.train_params import TrainingParams


if __name__ == "__main__":

    training_params = TrainingParams(
        training_name="cnn-training_refactor_data",
        epochs=10,
        learning_rate=0.001,
        model=CustomNet(),
        optimizer_class=torch.optim.Adam,
        loss_function=nn.CrossEntropyLoss(),
    )

    train_loader, val_loader = get_imagenet_dataloaders(batch_size=32)
    train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        training_params=training_params,
    )
