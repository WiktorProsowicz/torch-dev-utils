"""Contains miscelaneous utilities."""

from typing import Any
from typing import Dict
import abc

import torch


class IOptimizerWrapper(abc.ABC):
    """Interface for custom classes wrapping torch.Optimizer objects."""

    @abc.abstractmethod
    def step(self):
        """Performs a single optimization step."""

    @abc.abstractmethod
    def zero_grad(self):
        """Zeroes the gradients of the optimizer."""

    @abc.abstractmethod
    def state_dict(self) -> Dict[str, Any]:
        """Returns the state of the optimizer as a dictionary."""

    @abc.abstractmethod
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Loads the optimizer state from the specified state dictionary."""


class _BuiltinOptimizerWrapper(IOptimizerWrapper):
    """Wraps a built-in PyTorch optimizer."""

    def __init__(self, optimizer: torch.optim.Optimizer):
        self._optimizer = optimizer

    def step(self):
        self._optimizer.step()

    def zero_grad(self):
        self._optimizer.zero_grad()

    def state_dict(self) -> Dict[str, Any]:
        return self._optimizer.state_dict()

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self._optimizer.load_state_dict(state_dict)


def wrap_torch_optimizer(optimizer: torch.optim.Optimizer) -> IOptimizerWrapper:
    """Wraps a built-in PyTorch optimizer to provide a consistent interface."""

    return _BuiltinOptimizerWrapper(optimizer)
