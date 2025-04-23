import torch
from torch.optim import SGD
from typing import Dict


class SparseSGDM(SGD):
    """
    SparseSGDM is an extension of PyTorch's SGD optimizer that applies
    a gradient mask during the optimization step. This allows you to freeze
    certain model parameters by zeroing out their gradients before updating.

    It is typically used in sparse training or fine-tuning workflows where only
    a subset of parameters (e.g. those selected by Fisher Information) are allowed to update.

    Args:
        named_params (Dict[str, torch.nn.Parameter]): Dictionary mapping parameter names
            to their corresponding tensors. Used to match parameters to their gradient masks.
        grad_mask (Dict[str, torch.Tensor]): Dictionary mapping parameter names to binary masks
            of the same shape. A value of 1 allows gradient updates; 0 freezes the parameter.
        lr (float): Learning rate.
        momentum (float): Momentum factor.
        dampening (float): Dampening for momentum.
        weight_decay (float): Weight decay (L2 penalty).
        nesterov (bool): Enables Nesterov momentum.

    Example:
        >>> optimizer = SparseSGDM(named_params, grad_mask, lr=0.01)
        >>> loss.backward()
        >>> optimizer.step()

    Note:
        - The optimizer assumes that `named_params` includes all parameters passed to `SGD`.
        - Parameters not included in `grad_mask` will not be masked.
    """

    def __init__(
        self,
        named_params: Dict[str, torch.nn.Parameter],  # name -> param
        grad_mask: Dict[str, torch.Tensor],
        lr: float = 0.01,
        momentum: float = 0.9,
        dampening: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
    ):
        super(SparseSGDM, self).__init__(
            named_params.values(),
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )
        self.named_params = named_params
        self.grad_mask = grad_mask

    def step(self, closure=None):
        if closure is not None:
            closure()

        for group in self.param_groups:
            for param in group["params"]:
                # Find the parameter name
                name = next(
                    key for key, val in self.named_params.items() if val is param
                )
                if param.grad is not None and name in self.grad_mask:
                    param.grad.data *= self.grad_mask[name]

        return super().step(closure)
