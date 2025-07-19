import os
import random
from typing import IO, Any, BinaryIO
from collections.abc import Callable, Iterable
from typing import Optional
import math

from jaxtyping import Float, Int
import torch
from torch import Tensor
import numpy.typing as npt


def run_cross_entropy(inputs: Float[Tensor, " batch_size vocab_size"],
                      targets: Int[Tensor, " batch_size"]) -> Float[Tensor, ""]:
    maxes, inds = torch.max(inputs, dim=-1, keepdim=True)
    exps = torch.exp(inputs - maxes)
    targets = torch.gather(inputs, 1, targets.unsqueeze(dim=1)) - maxes
    return torch.mean(-(targets - torch.log(torch.sum(exps, dim=-1, keepdim=False))))


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr: float, weight_decay: float, betas: tuple[float, float], eps: float):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.

            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p] # Get state associated with p.
                t = state.get("t", 0) # Get iteration number from the state, or initial value.
                grad = p.grad.data # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad # Update weight tensor in-place.
                state["t"] = t + 1 # Increment iteration number.

        return loss


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr: float, weight_decay: float, betas: tuple[float, float], eps: float):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, "weight_decay": weight_decay, "betas": betas, "eps": eps}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]
            b1, b2 = group["betas"]
            eps = group["eps"]
            decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 1)
                m = state.get("m", torch.zeros_like(p))
                v = state.get("v", torch.zeros_like(p))
                m = b1 * m + (1 - b1) * p.grad.data
                v = b2 * v + (1 - b2) * p.grad.data ** 2
                state['m'] = m
                state['v'] = v
                lr_t = lr * ((1 - b2**t)**0.5 / (1 - b1**t))

                p.data -= lr_t * (m / (v ** 0.5 + eps)) # Update weight tensor in-place.
                p.data -= lr * decay * p.data
                state["t"] = t + 1 # Increment iteration number.

        return loss


def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    if it < warmup_iters:
        return it / warmup_iters * max_learning_rate
    elif it < cosine_cycle_iters:
        return min_learning_rate + 0.5 * (1 + math.cos((it - warmup_iters) / (cosine_cycle_iters - warmup_iters) * math.pi)) \
               * (max_learning_rate - min_learning_rate)
    else:
        return min_learning_rate


def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    grads = [param.grad for param in parameters if param.grad is not None]
    norm = torch.sqrt(sum(grad.norm() ** 2 for grad in grads))
    eps = 10e-6
    if norm > max_l2_norm:
        scale = max_l2_norm / (norm + eps)
        for param in parameters:
            if param.grad is not None:
                    with torch.no_grad():
                        param.grad.data *= scale



def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    max_i = len(dataset) - context_length - 1
    ind_data = []
    ind_targets = []

    for _ in range(batch_size):
        i = random.randint(0, max_i)
        ind_data.append(list(range(i, i+context_length)))
        ind_targets.append(list(range(i+1, i+1+context_length)))

    return torch.tensor(dataset[ind_data], device=device), \
           torch.tensor(dataset[ind_targets], device=device)


def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
) -> int:
    state = {'model': model.state_dict(),
             'optim': optimizer.state_dict(),
             'iteration': iteration}
    torch.save(state, out)


def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
):
    state = torch.load(src)
    model.load_state_dict(state['model'])
    optimizer.load_state_dict(state['optim'])
    return state['iteration']


