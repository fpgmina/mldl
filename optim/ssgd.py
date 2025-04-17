import torch
from torch.optim import SGD
from typing import Dict


class SparseSGDM(SGD):
    def __init__(
        self,
        params: torch.nn.Parameter,  # Model parameters to optimize
        grad_mask: Dict[
            str, torch.Tensor
        ],  # Dictionary with parameter names as keys and gradient masks as values
        lr: float = 0.01,
        momentum: float = 0.9,
        dampening: float = 0.0,  # Dampening factor (0 for no dampening)
        weight_decay: float = 0.0,  # L2 penalty (weight decay)
        nesterov: bool = False,  # Whether to use Nesterov momentum
    ):
        """
        SparseSGDM extends SGD to apply a gradient mask.
        grad_mask: A dictionary where the key is the parameter name, and the value is the corresponding mask.
        """
        super(SparseSGDM, self).__init__(
            params,
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )
        self.grad_mask = grad_mask  # Store the gradient mask for later use

    def step(self, closure: torch.nn.Module = None) -> torch.Tensor:
        """
        Performs a single optimization step, applying the gradient mask.
        """
        if closure is not None:
            closure()

        # Loop through all parameter groups in the optimizer
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue  # Skip if there are no gradients for this parameter

                # Apply gradient mask to zero-out gradients for parameters that are masked out
                param.grad.data *= self.grad_mask[
                    param.name
                ]  # Multiply gradients by the mask

        # Perform the standard SGD step to update parameters
        return super(SparseSGDM, self).step(closure)
