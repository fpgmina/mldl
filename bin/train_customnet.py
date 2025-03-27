import torch
from torch import nn
from dataset.imagenet import get_imagenet_dataloaders
from models.cnn import CustomNet
from training.train import train_model
from training.train_params import TrainingParams
from utils.model_utils import check_forward_pass, train_on_subset

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

    check_forward_pass(training_params.model.cuda(), train_loader)

    train_on_subset(training_params, train_loader, val_loader, epochs=2)

    train_model(
        training_params=training_params,
        train_loader=train_loader,
        val_loader=val_loader,
    )
