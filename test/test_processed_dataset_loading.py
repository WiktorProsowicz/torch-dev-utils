# -*- coding: utf-8 -*-
import os

import pytest

from torch_dev_utils import data_loading


# ----------------------------------------------------------------------------
# DATA STRUCTS & FUNCTIONS SECTION
# ----------------------------------------------------------------------------


ds_path = os.path.join(
    os.environ['TEST_RESOURCES'], 'processed_dataset_loading')


# ----------------------------------------------------------------------------
# FIXTURES SECTION
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# UNIT TESTS SECTION
# ----------------------------------------------------------------------------


@pytest.mark.parametrize('train_split_ratio, n_test_files, expected_lengths',
                         [(0.75, 1, (3, 1, 1)),
                          (0.8, 0, (4, 1, 0)),
                          (1., 3, (2, 0, 3)),
                          (0., 0, (0, 5, 0))])
def test_lengths_of_datasets_are_correct(train_split_ratio, n_test_files, expected_lengths):

    train_set, val_set, test_set = data_loading.get_datasets(ds_path,
                                                             train_split_ratio,
                                                             n_test_files,
                                                             True)

    assert len(train_set) == expected_lengths[0]
    assert len(val_set) == expected_lengths[1]
    assert len(test_set) == expected_lengths[2]


def test_dataset_is_loaded_deterministically():

    train_set_1, val_set_1, test_set_1 = data_loading.get_datasets(ds_path,
                                                                   0.75,
                                                                   1,
                                                                   True)

    train_set_2, val_set_2, test_set_2 = data_loading.get_datasets(ds_path,
                                                                   0.75,
                                                                   1,
                                                                   True)

    assert len(train_set_1) == len(train_set_2)
    assert len(val_set_1) == len(val_set_2)
    assert len(test_set_1) == len(test_set_2)

    for sample_1, sample_2 in zip(train_set_1, train_set_2):
        assert sample_1 == sample_2

    for sample_1, sample_2 in zip(val_set_1, val_set_2):
        assert sample_1 == sample_2

    for sample_1, sample_2 in zip(test_set_1, test_set_2):
        assert sample_1 == sample_2
