"""Contains utilities for serializing and deserializing PyTorch-based models."""

from typing import Dict
from typing import Any
from typing import Tuple
import sys
import logging
import os
import json
import time

import torch

from model import NamedModelComps
import model
import misc


def _try_load_state_dict(module: torch.nn.Module,
                         saved_module_path: str,
                         device: torch.device,
                         be_strict: bool = False):
    """Attempts to load the state dict of the module from the specified path."""

    if not os.path.exists(saved_module_path):

        if be_strict:
            logging.critical("Module state dict not found at '%s'.", saved_module_path)
            sys.exit(1)

        else:
            logging.warning("Module state dict not found at '%s'.", saved_module_path)
            return

    module.load_state_dict(torch.load(saved_module_path, weights_only=True, map_location=device))


def _load_from_path(modules: NamedModelComps,
                    path: str,
                    device: torch.device,
                    be_strict: bool = False):
    """Loads the given modules from the specified directory."""

    if not os.path.exists(path):
        logging.critical("Model components not found at '%s'.", path)
        sys.exit(1)

    for component_name, component in modules.items():
        if component is not None:
            _try_load_state_dict(component,
                                 os.path.join(path, f'{component_name}.pth'),
                                 device,
                                 be_strict)


def _save_to_path(self, modules: NamedModelComps, path: str):
    """Saves the model components to the specified directory."""

    os.makedirs(path, exist_ok=True)

    for component_name, component in modules.items():
        if component is not None:
            torch.save(component.state_dict(), os.path.join(path, f'{component_name}.pth'))


class ModelCheckpointHandler:
    """Handles the saving and loading of model checkpoints.

    The class is responsible for managing files inside the checkpoints directory. It shall
    store both the metadata of the saved checkpoints as well as the model components.
    """

    def __init__(self, checkpoint_dir: str,
                 device: torch.device,
                 missing_modules_strict: bool = False
                 ):
        """Initializes the ModelCheckpointHandler.

        Args:
            checkpoint_dir: The directory to store the model checkpoints.
            checkpoint_basename: The base name of the checkpoints.
            device: The device to load the model components to.
            missing_modules_strict: If True, if the state dict of a component is not found while
                loading the model's state, the program will terminate with critical error. 
        """

        self._checkpoint_dir = checkpoint_dir
        self._metadata_path = os.path.join(checkpoint_dir, 'metadata.json')
        self._device = device
        self._missing_modules_strict = missing_modules_strict

    def num_checkpoints(self) -> int:
        """Returns the number of saved checkpoints."""

        return len(self._get_metadata()['checkpoints'])

    def get_newest_checkpoint(self,
                              model_components: model.BaseModelComponents,
                              optimizer: misc.IOptimizerWrapper
                              ) -> Tuple[model.BaseModelComponents,
                                         misc.IOptimizerWrapper,
                                         Dict[str, Any]]:
        """Loads the newest checkpoint from the checkpoint directory.

        Args:
            model_components: The components of the model to load the checkpoint into.

        Returns:
            A tuple containing the model components, optimizer and the metadata of the
            newest checkpoint.
        """

        metadata = self._get_metadata()

        if not metadata['checkpoints']:
            logging.critical('No checkpoints found.')
            sys.exit(1)

        newest_checkpoint = metadata['checkpoints'][-1]

        checkpoint_path = os.path.join(self._checkpoint_dir, newest_checkpoint['directory_name'])
        optim_state_path = os.path.join(checkpoint_path, 'optimizer_state.pth')

        _load_from_path(model_components.get_components(),
                        checkpoint_path,
                        self._device,
                        self._missing_modules_strict)
        optimizer.load_state_dict(torch.load(optim_state_path, weights_only=True))

        return model_components, optimizer, newest_checkpoint['metadata']

    def save_checkpoint(self,
                        model_components: model.BaseModelComponents,
                        optimizer: misc.IOptimizerWrapper,
                        checkpoint_metadata: Dict[str, Any]):
        """Saves the model components as a checkpoint.

        Args:
            model_components: The components of the model to save.
            optimizer: The optimizer at a current training stage to be saved.
            checkpoint_metadata: The metadata of the checkpoint.
        """

        metadata = self._get_metadata()

        if not metadata['checkpoints']:
            checkpoint_dir = os.path.join(self._checkpoint_dir,
                                          'ckpt_0')

        else:
            last_ckpt_dir = metadata['checkpoints'][-1]['directory_name']
            last_ckpt_num = int(last_ckpt_dir.split('_')[1])

            checkpoint_dir = os.path.join(self._checkpoint_dir,
                                          f'ckpt_{last_ckpt_num + 1}')

        _save_to_path(model_components.get_components(), checkpoint_dir)
        torch.save(optimizer.state_dict(), os.path.join(checkpoint_dir, 'optimizer_state.pth'))

        metadata['checkpoints'].append({
            'directory_name': os.path.basename(checkpoint_dir),
            'metadata': checkpoint_metadata,
            'timestamp': int(time.time())
        })

        self._save_metadata(metadata)

    def _get_metadata(self) -> Dict[str, Any]:
        """Returns the metadata of the saved checkpoints."""

        if not os.path.exists(self._metadata_path):
            return {
                'checkpoints': []
            }

        with open(self._metadata_path, 'r', encoding='utf-8') as file:
            return json.load(file)

    def _save_metadata(self, metadata: Dict[str, Any]):
        """Saves the metadata of the saved checkpoints."""

        with open(self._metadata_path, 'w', encoding='utf-8') as file:
            json.dump(metadata, file)
