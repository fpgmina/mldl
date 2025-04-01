# train pizza steak sushi
from models.transfer_learning import get_model
from dataset.pizza_steak_sushi import get_dataloaders, get_pretrained_model

if __name__ == "__main__":
    train_dataloader, test_dataloader, class_names = get_dataloaders()
    pretrained_model = get_pretrained_model()
    model = get_model(model=pretrained_model, class_names=class_names)
    params = TrainingParams(
        training_name="transfer_learning",
        model=model,
        loss_function=nn.CrossEntropyLoss(),
        learning_rate=1e-3,
        optimizer_class=torch.optim.Adam,
        epochs=10,
    )
    train_model(
        training_params=params,
        train_loader=train_dataloader,
        val_loader=test_dataloader,
        project_name="mldl_lab5",
    )
