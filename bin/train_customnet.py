import torch
from torch import nn
import sys
import os

# Add the root project directory to the system path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from dataset.imagenet import get_imagenet_dataloaders
from models.cnn import CustomNet
from training.train import train_model
from training.train_params import TrainingParams


if __name__ == "__main__":

    training_params = TrainingParams(
        training_name="cnn-training_refactor", 
        epochs=10, 
        learning_rate=0.001, 
        architecture=CustomNet, 
        optimizer=torch.optim.Adam,
        loss_function=nn.CrossEntropyLoss,
        )

    train_loader, val_loader = get_imagenet_dataloaders(batch_size=32)
    train_model(train_loader=train_loader, val_loader=val_loader, training_params=training_params)
    







