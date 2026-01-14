# change: CV helpers for grouped cross-validation
"""
Small module providing grouped cross-validation utilities.
Keep main_offline minimal by delegating CV split creation here.
"""

from typing import List, Tuple

import numpy as np
from sklearn.model_selection import GroupKFold


def grouped_kfold_indices(
    groups: np.ndarray, n_splits: int = 5
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Return a list of (train_idx, val_idx) index arrays produced by GroupKFold,
    ensuring that no group is split between train and validation.

    Parameters:
    - groups: array-like of group labels (length = n_samples)
    - n_splits: number of folds

    Returns:
    - List of (train_idx, val_idx) tuples
    """
    groups = np.asarray(groups)
    gkf = GroupKFold(n_splits=n_splits)
    splits = []
    # GroupKFold yields indices for arrays (0..n_samples-1)
    for train_idx, val_idx in gkf.split(
        np.zeros(len(groups)), np.zeros(len(groups)), groups
    ):
        splits.append((train_idx, val_idx))
    return splits


# change: convenience function to get fold generator
def grouped_kfold_generator(groups: np.ndarray, n_splits: int = 5):
    for train_idx, val_idx in grouped_kfold_indices(groups, n_splits=n_splits):
        yield train_idx, val_idx


# change: helper to extract windows inside CV folds to avoid leakage
