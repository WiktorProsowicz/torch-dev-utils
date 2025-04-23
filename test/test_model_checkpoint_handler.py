
from typing import Optional
import dataclasses
import os
import shutil

import pytest
import torch

from torch_dev_utils import serialization
from torch_dev_utils import model
from torch_dev_utils import misc


# ----------------------------------------------------------------------------
# DATA STRUCTS & FUNCTIONS SECTION
# ----------------------------------------------------------------------------


checkpoints_path = os.path.join(
    os.environ['TEST_RESOURCES'], 'model_checkpoint_handler')


@dataclasses.dataclass
class TestModelComps(model.BaseModelComponents):

    layer_1: torch.nn.Linear
    layer_2: Optional[torch.nn.Linear]

    def get_components(self) -> model.NamedModelComps:
        return {
            'layer_1': self.layer_1,
            'layer_2': self.layer_2
        }


# ----------------------------------------------------------------------------
# FIXTURES SECTION
# ----------------------------------------------------------------------------


@pytest.fixture
def sample_comps_complete():

    comps = TestModelComps(layer_1=torch.nn.Linear(10, 5),
                           layer_2=torch.nn.Linear(10, 5))
    opt = torch.optim.SGD(comps.parameters(), lr=0.1, momentum=0.9)

    return comps, misc.wrap_torch_optimizer(opt)


@pytest.fixture
def sample_comps_incomplete():

    comps = TestModelComps(layer_1=torch.nn.Linear(10, 5), layer_2=None)
    opt = torch.optim.SGD(comps.parameters(), lr=0.1, momentum=0.9)

    return comps, opt


# ----------------------------------------------------------------------------
# UNIT TESTS SECTION
# ----------------------------------------------------------------------------


@pytest.mark.xfail()
def test_fails_if_not_found_expected_comps(sample_comps_complete):

    comps, optimizer = sample_comps_complete

    handler = serialization.ModelCheckpointHandler(checkpoint_dir=checkpoints_path,
                                                   device=torch.device('cpu'),
                                                   missing_modules_strict=True)

    comps, optimizer, _ = handler.get_newest_checkpoint(comps, optimizer)


def test_correctly_loads_preexisting_state(sample_comps_incomplete):

    comps, optimizer = sample_comps_incomplete

    assert not optimizer.state_dict()['state']

    handler = serialization.ModelCheckpointHandler(checkpoint_dir=checkpoints_path,
                                                   device=torch.device('cpu'),
                                                   missing_modules_strict=False)

    assert handler.num_checkpoints() == 1

    comps, optimizer, metadata = handler.get_newest_checkpoint(
        comps, optimizer)

    assert optimizer.state_dict()['state']
    assert metadata['n_training_steps'] == 3000


def test_correctly_saves_state(sample_comps_incomplete):

    new_checkpoints_path = os.path.join(
        os.environ['TEST_RESULTS'], 'model_checkpoint_handler')
    shutil.copytree(checkpoints_path, new_checkpoints_path)

    handler = serialization.ModelCheckpointHandler(checkpoint_dir=new_checkpoints_path,
                                                   device=torch.device('cpu'),
                                                   missing_modules_strict=False)
    comps, optimizer = sample_comps_incomplete

    assert handler.num_checkpoints() == 1

    torch.nn.MSELoss()(comps.layer_1(torch.randn(10)), torch.randn(5)).backward()
    optimizer.step()
    handler.save_checkpoint(comps, optimizer, {'n_training_steps': 3001})

    assert handler.num_checkpoints() == 2

    comps, optimizer, metadata = handler.get_newest_checkpoint(
        comps, optimizer)

    assert metadata['n_training_steps'] == 3001
