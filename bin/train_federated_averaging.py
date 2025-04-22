from core.federated_averaging import FederatedAveraging, ShardingType
from dataset.cifar_100 import (
    get_cifar_dataloaders,
    get_cifar_100_train_valset_datasets,
    get_cifar_100_datasets,
)
from models.dino_backbone import get_dino_backbone_model

if __name__ == "__main__":
    train_dataloader, val_dataloader = get_cifar_dataloaders(batch_size=32)
    model = get_dino_backbone_model()
    trainset, _ = get_cifar_100_datasets()
    trainset, valset = get_cifar_100_train_valset_datasets(trainset)
    fedav = FederatedAveraging(
        global_model=model,
        trainset=trainset,
        valset=valset,
        sharding_type=ShardingType.IID,
    )
    fedav.train()
