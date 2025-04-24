# -*- coding: utf-8 -*-
"""Contains utilities for training and evaluation of models."""
import abc
import logging
import sys
import time
from typing import Dict
from typing import Optional
from typing import Tuple

import torch.utils.tensorboard

from torch_dev_utils import misc
from torch_dev_utils import model
from torch_dev_utils import serialization


class BaseTrainer(abc.ABC):
    """Base class for the model training pipeline.

    The class defines the functions for:
    - Profiling code
    - Running validation and training loop
    - Saving and loading model checkpoints
    - Running backward propagation and optimization
    """

    def __init__(self,  # pylint: disable=too-many-positional-arguments
                 model_comps: model.BaseModelComponents,
                 optimizer: misc.IOptimizerWrapper,
                 checkpoints_handler: serialization.ModelCheckpointHandler,
                 train_data_loader: torch.utils.data.DataLoader,
                 val_data_loader: torch.utils.data.DataLoader,
                 tb_logger: torch.utils.tensorboard.writer.SummaryWriter,
                 device: torch.device,
                 validation_interval: int,
                 checkpoints_interval: int,
                 log_interval: int = 100):
        """Initializes the trainer.

        Args:
            model_comps: Components of the model to be trained.
            optimizer: The optimizer to use for training.
            checkpoints_handler: The handler for saving/loading model checkpoints.
            train_data_loader: The data loader for the training data.
            val_data_loader: The data loader for the validation data.
            tb_logger: The TensorBoard logger.
            device: The device to run the training on.
            validation_interval: The number of steps between validation runs.
            checkpoints_interval: The number of steps between saving checkpoints.
            log_interval: The number of steps between logging training progress.
        """

        if checkpoints_handler.num_checkpoints() > 0:
            model_comps, optimizer, _ = checkpoints_handler.get_newest_checkpoint(  # type: ignore
                model_comps,
                optimizer)

        self._model_comps = model_comps
        self._train_data_loader = train_data_loader
        self._val_data_loader = val_data_loader
        self._tb_logger = tb_logger
        self._device = device
        self._validation_interval = validation_interval
        self._checkpoints_handler = checkpoints_handler
        self._checkpoints_interval = checkpoints_interval
        self._optimizer = optimizer
        self._log_interval = log_interval

    def run_training(self, num_steps: int, start_step: int = 0, use_profiler: bool = False):
        """Runs the training pipeline.

        Args:
            num_steps: The total number of training steps to run. Once the step index reaches this,
                the training stops.
            start_step: The step to start training from.
            use_profiler: Whether to use the code profiling while training.
        """

        if use_profiler:

            with torch.profiler.profile(
                schedule=torch.profiler.schedule(wait=2, warmup=5, active=5, repeat=1),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    self._tb_logger.get_logdir()),
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
            ) as profiler:

                self._run_training_pipeline(num_steps, start_step, profiler)

        else:

            self._run_training_pipeline(num_steps, start_step)

    def _run_training_pipeline(self, num_steps: int,
                               start_step: int = 0,
                               profiler: Optional[torch.profiler.profile] = None):
        """Runs the training pipeline.

        Args:
            num_steps: The number of training steps to run.
            profiler: The profiler to use for profiling the code.
        """

        start_time = time.time()
        logging.debug('Training pipeline started.')

        data_loader_enum = enumerate(self._train_data_loader)

        for step_idx in range(start_step, num_steps):

            self._on_step_start(step_idx)

            try:
                _, batch = next(data_loader_enum)

            except StopIteration:
                data_loader_enum = enumerate(self._train_data_loader)
                _, batch = next(data_loader_enum)

            if profiler:
                profiler.step()

            self._run_training_step(step_idx, batch)

            if (step_idx + 1) % self._log_interval == 0:
                logging.debug('Performed %d training steps. (Avg time/step in sec: %.2f).',
                              step_idx + 1,
                              (time.time() - start_time) / (step_idx - start_step + 1))

            if (step_idx + 1) % self._validation_interval == 0:

                logging.debug('Running validation after %d steps...', step_idx + 1)
                self._run_validation(step_idx)

            if (step_idx + 1) % self._checkpoints_interval == 0:

                logging.debug('Saving checkpoint after %d steps...', step_idx + 1)
                meta_data = {
                    'n_training_steps': step_idx + 1,
                }
                self._checkpoints_handler.save_checkpoint(self._model_comps,
                                                          self._optimizer,
                                                          meta_data)

            self._on_step_end(step_idx)

        logging.info('Training pipeline finished.')
        logging.debug('Training took %.2f minutes.', (time.time() - start_time) / 60)
        logging.debug('Average time per step: %.2f seconds.',
                      (time.time() - start_time) / (num_steps - start_step))

    def _run_training_step(self, step_idx: int, batch):
        """Runs a single training step.

        Args:
            step_idx: The index of the current step.
        """

        self._model_comps.train()

        batch = tuple(tensor.to(self._device) for tensor in batch)

        self._optimizer.zero_grad()

        losses_and_metrics = self._compute_losses(batch)

        for name, value in losses_and_metrics.items():
            self._tb_logger.add_scalars(name, {'training': value.item()}, step_idx)

        if 'total_loss' not in losses_and_metrics:
            logging.critical('Returned losses dictionary should contain "total_loss" key!')
            sys.exit(1)

        total_loss = losses_and_metrics['total_loss']

        total_loss.backward()
        self._optimizer.step()

    def _run_validation(self, step_idx: int):
        """Runs validation on the validation data.

        Args:
            step_idx: The index of the current training step.
        """

        self._model_comps.eval()

        with torch.no_grad():

            avg_losses_and_metrics = {}

            for batch in self._val_data_loader:

                batch = tuple(tensor.to(self._device) for tensor in batch)

                losses_and_metrics = self._compute_losses(batch)

                for name, value in losses_and_metrics.items():

                    if name not in avg_losses_and_metrics:
                        avg_losses_and_metrics[name] = torch.tensor(0.0, device=self._device)

                    avg_losses_and_metrics[name] += value

            for name in avg_losses_and_metrics:
                avg_losses_and_metrics[name] /= len(self._val_data_loader)

                self._tb_logger.add_scalars(name,
                                            {'validation': avg_losses_and_metrics[name].item()},
                                            step_idx)

    @abc.abstractmethod
    def _compute_losses(self,
                        input_batch: Tuple[torch.Tensor, ...]
                        ) -> Dict[str, torch.Tensor]:
        """Computes the losses for the given inputs.

        Args:
            input_batch: Inputs to the model. The input tensors are expected to be moved to the
                chosen device before calling this function.

        Returns:
            A dictionary containing the computed losses and metrics. The dictionary should
            contain at least a tensor named 'total_loss', since the backward propagation is
            computed with respect to this value.
        """

    @abc.abstractmethod
    def _on_step_end(self, step_idx: int):
        """Runs after each training step.

        Args:
            step_idx: The index of the current step.
        """

    @abc.abstractmethod
    def _on_step_start(self, step_idx: int):
        """Runs after each training step.

        Args:
            step_idx: The index of the current step.
        """
