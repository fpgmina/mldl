import torch
from torch import nn
from torch.utils.data import DataLoader
from typing import Optional

from utils.model_utils import get_device


# Loss function (averaged over N samples):
#
#     L(θ) = (1 / N) * ∑_{i=1}^{N} L^{(i)}(θ) where L^{(i)} = L(y^i, x^i) i.e. the loss on the i-th data point
#
# Gradient of the loss with respect to model parameters θ:
#
#     ∇L(θ) = (1 / N) * ∑_{i=1}^{N} ∇L^{(i)}(θ)
#
# Where:
#     - L^{(i)}(θ) is the loss on the i-th data point (x_i, y_i)
#     - ∇L^{(i)}(θ) = ∂L^{(i)} / ∂θ is the gradient of the loss for sample i
#     - N is the number of samples in the dataset or minibatch
#


def compute_fisher_diagonal(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    num_batches: Optional[int] = None,
) -> torch.Tensor:
    """
    Compute the diagonal of the Fisher Information Matrix using squared gradients.

    Returns:
        fisher_diag (Tensor): Flattened tensor of Fisher Information scores (1D).
    """
    model.eval()
    fisher_diag = None
    total_samples = 0
    device = get_device()

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        if num_batches is not None and batch_idx >= num_batches:
            break

        inputs, targets = inputs.to(device), targets.to(device)
        model.zero_grad()

        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()

        grads = []
        for p in model.parameters():
            if p.grad is not None:
                # g^(i) = ∂L^(i) / ∂θ the gradient of the mini batch (a p dimensional vector
                # where p is the number of parameters)
                grads.append(p.grad.detach().clone().flatten())
            else:
                grads.append(torch.zeros_like(p.data.flatten()))

        grad_vector = torch.cat(grads)

        if fisher_diag is None:
            fisher_diag = grad_vector.pow(2)
        else:
            fisher_diag += grad_vector.pow(2)

        total_samples += 1

    fisher_diag /= total_samples
    return fisher_diag
