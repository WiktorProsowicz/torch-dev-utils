"""Contains tools for loading serialized dataset."""

import os
from typing import List
from typing import Tuple

import numpy as np
import torch
from torch.utils import data as torch_data


class _ProcessedDataset(torch_data.Dataset):
    """Contains processed dataset samples."""

    def __init__(self, dataset_path: str, file_ids: List[str]):
        """Initializes dataset.

        Args:
            dataset_path: Path to the folder containing the dataset samples.
            file_ids: List of file IDs to load.
        """

        super().__init__()

        self._dataset_path = dataset_path
        self._file_ids = file_ids

    def __len__(self) -> int:
        """Returns the number of samples in the dataset."""
        return len(self._file_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns the dataset sample with the given index.

        Args:
            idx: Index of the sample to return.

        Returns:
            Tuple containing the elements of the requested data sample.
        """

        file_id = self._file_ids[idx]
        sample_path = os.path.join(self._dataset_path, f'{file_id}.pt')
        return torch.load(sample_path, weights_only=True)


def get_datasets(processed_dataset_path: str,
                 train_split_ratio: float,
                 n_test_files: int,
                 split_deterministic: bool = True
                 ) -> Tuple[torch_data.Dataset, torch_data.Dataset, torch_data.Dataset]:
    """Returns train/validation/test sets for processed dataset.

    A processed dataset is a dataset that has already been pre-processed and saved in a directory 
    as a set of .pt files. The files are expected to contain PyTorch tensors and built-in structures
    si that it could be loaded using torch.load. The files' names shall be considered as the IDs of
    the samples.  

    Args:
        dataset_path: Path to the processed dataset.
        train_split_ratio: Ratio of the dataset to use for training.
        n_test_files: Number of files to use for testing.
        split_deterministic: If True, the dataset will be split using a fixed seed.

    Returns:
        Datasets for training, validation, and testing. The testing dataset the
        training/validation sets contain data samples according to the given split
        minus the number of test files.
    """

    seed = 2137 if split_deterministic else None

    rng = np.random.RandomState(seed)  # pylint: disable=no-member
    file_ids = []

    for file_name in os.listdir(processed_dataset_path):
        if file_name.endswith('.pt'):
            file_ids.append(file_name.split('.')[0])

    test_file_ids = rng.choice(file_ids, n_test_files, replace=False)

    file_ids = [file_id for file_id in file_ids if file_id not in test_file_ids]

    n_val_files = round(len(file_ids) * (1.0 - train_split_ratio))
    val_file_ids = rng.choice(file_ids, n_val_files, replace=False)
    train_file_ids = [file_id for file_id in file_ids if file_id not in val_file_ids]

    return (_ProcessedDataset(processed_dataset_path, train_file_ids),
            _ProcessedDataset(processed_dataset_path, val_file_ids),
            _ProcessedDataset(processed_dataset_path, test_file_ids))
