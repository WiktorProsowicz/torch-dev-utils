# -*- coding: utf-8 -*-
"""Contains utilities for model definition."""
import abc
from typing import Dict
from typing import Iterator
from typing import Optional
from typing import TypeAlias

import torch


NamedModelComps: TypeAlias = Dict[str, Optional[torch.nn.Module]]


class BaseModelComponents(abc.ABC):
    """Base class for the model components.

    The classes inheriting from this class are rather expected to be dataclasses, declaring the
        internal components, instead of actually using them.
    """

    def parameters(self) -> Iterator[torch.nn.Parameter]:
        """Returns the parameters of the model."""

        for component in self.get_components().values():
            if component is not None:
                yield from component.parameters()

    def eval(self):
        """Sets the model to evaluation mode."""

        for component in self.get_components().values():
            if component is not None:
                component.eval()

    def train(self):
        """Sets the model to training mode."""

        for component in self.get_components().values():
            if component is not None:
                component.train()

    @abc.abstractmethod
    def get_components(self) -> NamedModelComps:
        """Returns all named components possessed by the concrete class' instance."""
