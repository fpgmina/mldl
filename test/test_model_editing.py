import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from core.model_editing import compute_fisher_diagonal


class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=3)  # Output: (batch, 4, 6, 6)
        self.fc = nn.Linear(4 * 6 * 6, 10)  # Correct size: 144 → 10 classes

    def forward(self, x):
        x = torch.relu(self.conv(x))
        x = x.view(x.size(0), -1)
        return self.fc(x)


def test_compute_fisher_diagonal():
    # Create dummy dataset (batch of 8, 1x8x8 images, 10 classes)
    X = torch.randn(8, 1, 8, 8)
    y = torch.randint(0, 10, (8,))
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=4)

    model = TinyCNN()
    loss_fn = nn.CrossEntropyLoss()

    fisher_diag = compute_fisher_diagonal(model, dataloader, loss_fn, num_batches=2)

    total_params = sum(p.numel() for p in model.parameters())
    assert isinstance(fisher_diag, torch.Tensor)
    assert fisher_diag.ndim == 1
    assert fisher_diag.numel() == total_params
    assert (fisher_diag >= 0).all(), "Fisher scores must be non-negative"
