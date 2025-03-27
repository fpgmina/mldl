import numpy as np
import torch.utils.data
from torch.utils.data import Subset, DataLoader


def check_forward_pass(model, train_loader, num_classes=200):
    sample_batch, _ = next(iter(train_loader))  # Get one batch
    sample_batch = sample_batch.cuda()  # If you're using a GPU

    output = model(sample_batch)
    assert output.shape == (train_loader.batch_size, num_classes), "Forward Pass Failed"
    print("Forward Pass works!")


def get_subset_loader(dataloader, subset_size=1000) -> torch.utils.data.DataLoader:
    dataset_size = len(dataloader.dataset)
    subset_indices = np.random.choice(dataset_size, subset_size, replace=False)
    subset_dataset = Subset(dataloader.dataset, subset_indices)
    subset_loader = DataLoader(subset_dataset, batch_size=32, shuffle=True)
    return subset_loader
