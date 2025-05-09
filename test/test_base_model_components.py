# -*- coding: utf-8 -*-
import dataclasses
import itertools
from typing import Optional

import pytest
import torch

from torch_dev_utils import model


@dataclasses.dataclass
class TestModelComponents(model.BaseModelComponents):
    """Simple test model components."""

    comp_dense_1: torch.nn.Linear
    comp_dense_2: torch.nn.Linear
    comp_none: Optional[torch.nn.Module]

    def get_components(self) -> model.NamedModelComps:
        return {
            'comp_dense_1': self.comp_dense_1,
            'comp_dense_2': self.comp_dense_2,
            'comp_none': self.comp_none
        }


@pytest.fixture
def sample_model_components():
    """Fixture for creating a sample model components instance."""

    return TestModelComponents(
        comp_dense_1=torch.nn.Linear(10, 5),
        comp_dense_2=torch.nn.Linear(5, 2),
        comp_none=None
    )


# ----------------------------------------------------------------------------
# UNIT TESTS SECTION
# ----------------------------------------------------------------------------


def test_yields_correct_params(sample_model_components: TestModelComponents):

    def expected_params():
        return itertools.chain(
            sample_model_components.comp_dense_1.parameters(),
            sample_model_components.comp_dense_2.parameters())

    assert len(list(sample_model_components.parameters())) == len(list(expected_params()))

    for param, expected_param in zip(sample_model_components.parameters(), expected_params()):
        assert param is expected_param


def test_sets_eval_mode(sample_model_components: TestModelComponents):

    sample_model_components.eval()

    for comp in sample_model_components.get_components().values():
        if comp is not None:
            assert comp.training is False


def test_sets_train_mode(sample_model_components: TestModelComponents):

    sample_model_components.train()

    for comp in sample_model_components.get_components().values():
        if comp is not None:
            assert comp.training is True
