import torch
from torch import nn
from torch.utils.data import DataLoader
from typing import Optional, List

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

    The Fisher Information is given by the expected value of the squared gradient of the loss function:

        Fisher(θ) = E_{(x, y) ~ D} [ (∂L(f_θ(x), y) / ∂θ)^2 ]

    This function approximates and returns the diagonal of the Fisher Information Matrix:

        Fisher(θ) ≈ (1 / N) * ∑_{i=1}^{N} (∇_θ L^{(i)}(θ))²

    Where:
        - L^{(i)}(θ) is the loss on the i-th data point
        - ∇_θ L^{(i)}(θ) is the gradient of the loss with respect to parameters θ
        - N is the number of samples (or mini-batches)
        - The square is element-wise and gives the diagonal approximation

    Args:
        model (nn.Module): The model whose parameters are being analyzed.
        dataloader (DataLoader): DataLoader providing input–target pairs.
        loss_fn (nn.Module): Loss function used to compute gradients.
        num_batches (Optional[int]): If set, only the first `num_batches` are used
            to estimate the Fisher information. Useful for faster computation.

    Returns:
        fisher_diag (torch.Tensor): A flattened tensor containing the Fisher diagonal estimate,
        one element per parameter.
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


def create_fisher_mask(
    fisher_diag: torch.Tensor, model: nn.Module, keep_ratio: float = 0.2
) -> List[torch.Tensor]:
    """
    Generate a binary mask for gradients, keeping only the top-k Fisher scores.

    Args:
        fisher_diag: Flattened Fisher information scores.
        model: Model whose parameters the mask will align with.
        keep_ratio: Fraction of weights to keep (e.g., 0.2 = top 20%).

    Returns:
        List of binary masks shaped like each model parameter.
    """
    k = int(len(fisher_diag) * keep_ratio)
    threshold = torch.topk(fisher_diag, k, largest=True).values[-1]
    flat_mask = (fisher_diag >= threshold).float()

    param_shapes = [p.shape for p in model.parameters()]
    param_sizes = [p.numel() for p in model.parameters()]
    split_masks = torch.split(flat_mask, param_sizes)
    shaped_masks = [m.view(shape) for m, shape in zip(split_masks, param_shapes)]

    return shaped_masks
