# -*- coding: utf-8 -*-
import dataclasses
import os
import shutil
from typing import Dict
from typing import Tuple

import pytest
import torch
import torch.utils.tensorboard as tb
from tensorboard.backend.event_processing import event_accumulator  # type: ignore

from torch_dev_utils import data_loading
from torch_dev_utils import misc
from torch_dev_utils import model
from torch_dev_utils import serialization
from torch_dev_utils import training


# ----------------------------------------------------------------------------
# DATA STRUCTS & FUNCTIONS SECTION
# ----------------------------------------------------------------------------

resources_path = os.path.join(
    os.environ['TEST_RESOURCES'], 'base_trainer')

results_path = os.path.join(
    os.environ['TEST_RESULTS'], 'base_trainer')


dataset_path = os.path.join(resources_path, 'dataset')


@dataclasses.dataclass
class SampleModelComps(model.BaseModelComponents):

    layer: torch.nn.Linear

    def get_components(self) -> model.NamedModelComps:
        return {
            'layer': self.layer,
        }


class SampleTrainer(training.BaseTrainer):

    def __init__(self, params: training.BaseTrainerParams):
        super().__init__(params)

        self.steps_start_cnt = 0
        self.steps_end_cnt = 0
        self.loss_func = torch.nn.MSELoss()

    @property
    def model_comps(self) -> SampleModelComps:
        assert isinstance(self._model_comps, SampleModelComps)
        return self._model_comps

    def _compute_losses(self,
                        input_batch: Tuple[torch.Tensor, ...]
                        ) -> Dict[str, torch.Tensor]:

        inputs, labels = input_batch

        return {
            'total_loss': self.loss_func(self.model_comps.layer(inputs), labels)
        }

    def _on_step_end(self, step_idx):
        self.steps_start_cnt += 1

    def _on_step_start(self, step_idx):
        self.steps_end_cnt += 1


# ----------------------------------------------------------------------------
# FIXTURES SECTION
# ----------------------------------------------------------------------------


@pytest.fixture(scope='session')
def training_data():

    checkpoints_path = os.path.join(results_path, 'checkpoints')
    shutil.copytree(os.path.join(resources_path, 'checkpoints'), checkpoints_path)

    model_comps = SampleModelComps(layer=torch.nn.Linear(5, 1))
    optimizer = torch.optim.SGD(model_comps.parameters(), lr=0.1)

    logging_path = os.path.join(results_path, 'logs')
    tb_logger = tb.writer.SummaryWriter(logging_path)
    handler = serialization.ModelCheckpointHandler(
        checkpoints_path,
        torch.device('cpu'),
        missing_modules_strict=True)

    train_ds, test_ds, _ = data_loading.get_datasets(dataset_path, 0.8, 0, True)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=1)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=1)

    trainer_params = training.BaseTrainerParams(
        model_comps=model_comps,
        optimizer=misc.wrap_torch_optimizer(optimizer),
        checkpoints_handler=handler,
        train_data_loader=train_loader,
        val_data_loader=test_loader,
        tb_logger=tb_logger,
        device=torch.device('cpu'),
        validation_interval=5,
        checkpoints_interval=5,
        log_interval=1)

    trainer = SampleTrainer(trainer_params)

    trainer.run_training(num_steps=20, start_step=10, use_profiler=False)

    return trainer, handler, logging_path

# ----------------------------------------------------------------------------
# UNIT TESTS SECTION
# ----------------------------------------------------------------------------


def test_checkpoints_are_saved(training_data):

    _, handler, _ = training_data

    assert handler.num_checkpoints() == 3


def test_training_hooks_are_called(training_data):

    trainer, _, _ = training_data

    assert trainer.steps_start_cnt == 10
    assert trainer.steps_end_cnt == 10


@pytest.mark.parametrize('label, expected_length', [
    ('total_loss_training', 10),
    ('total_loss_validation', 2),
])
def test_losses_are_logged(label, expected_length, training_data):

    _, _, logging_path = training_data

    assert os.path.exists(os.path.join(logging_path, label))

    run = event_accumulator.EventAccumulator(os.path.join(logging_path, label))
    run.Reload()

    assert len(run.Scalars('total_loss')) == expected_length
