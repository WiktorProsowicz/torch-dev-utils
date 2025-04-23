
import pytest
import torch

from torch_dev_utils import misc


# ----------------------------------------------------------------------------
# DATA STRUCTS & FUNCTIONS SECTION
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# FIXTURES SECTION
# ----------------------------------------------------------------------------


@pytest.fixture
def sgd_wrapper():
    """Creates wrapper for simple SGD optimizer."""

    model = torch.nn.Linear(10, 1)
    optim = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    return model, misc.wrap_torch_optimizer(optim)


@pytest.fixture
def sgd_wrapper_2():
    """Creates wrapper for simple SGD optimizer."""

    model = torch.nn.Linear(10, 1)
    optim = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.99)

    return model, misc.wrap_torch_optimizer(optim)


# ----------------------------------------------------------------------------
# UNIT TESTS SECTION
# ----------------------------------------------------------------------------


def test_returns_correct_state_dict(sgd_wrapper):

    _, optim = sgd_wrapper

    state_dict = optim.state_dict()

    assert 'state' in state_dict
    assert 'param_groups' in state_dict
    assert len(state_dict['param_groups']) == 1
    assert state_dict['param_groups'][0]['lr'] == 0.1
    assert state_dict['param_groups'][0]['momentum'] == 0.9


def test_performs_optimization_step(sgd_wrapper):

    model, optim = sgd_wrapper

    torch.nn.MSELoss()(model(torch.randn(10)), torch.randn(1)).backward()

    optim.step()

    state_dict = optim.state_dict()

    assert state_dict['state']


def test_loads_state_dict(sgd_wrapper, sgd_wrapper_2):

    _, optim = sgd_wrapper
    _, optim_2 = sgd_wrapper_2

    state_dict = optim.state_dict()

    optim_2.load_state_dict(state_dict)
    state_dict_after = optim_2.state_dict()

    assert (state_dict_after['param_groups'][0]['lr']
            == state_dict['param_groups'][0]['lr'])
    
    assert (state_dict_after['param_groups'][0]['momentum']
            == state_dict['param_groups'][0]['momentum'])
