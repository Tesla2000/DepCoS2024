from typing import Literal

import torch
import torchaudio.transforms as T
import torchvision.transforms as transforms
from sklearn.model_selection import StratifiedKFold

from Config import Config
from Evaluation.utilities import (
    get_patients_id,
    get_files_path,
)
from Models.Augmentations import Augmentation


def get_transform(
    vowels: list[Literal['a', 'i', 'u', 'all']],
    augmentation: Augmentation,
    num_splits: int,
    random_state=42, ):
    patients_ids, file_paths = set(), set()
    for vowel in vowels:
        file_path = (
            Config.lists_path
            / f'Vowels_{vowel}_{Config.disease}.txt'
        )
        # Load patient IDs and file paths from a file
        patients_ids += get_patients_id(file_path)
        file_paths += get_files_path(file_path)

    # Define augmentations
    transform_resize = transforms.Compose(
        [transforms.Resize((224, 224), antialias=None)]
    )

    def add_trailing_zeros(x):
        zeros = torch.zeros((*x.shape[:-1], 500))
        zeros[:, :, : x.shape[-1]] = x
        return zeros

    transform_pad_zeros = transforms.Compose([add_trailing_zeros])

    transform_frequency_masking = transforms.Compose(
        [
            T.FrequencyMasking(freq_mask_param=50),
            transforms.Resize((224, 224), antialias=None),
        ]
    )

    transform_time_masking = transforms.Compose(
        [
            T.TimeMasking(time_mask_param=30),
            transforms.Resize((224, 224), antialias=None),
        ]
    )

    transform_combined_masking = transforms.Compose(
        [
            T.FrequencyMasking(freq_mask_param=50),
            T.TimeMasking(time_mask_param=30),
            transforms.Resize((224, 224), antialias=None),
        ]
    )

    # Choose the desired augmentation
    val_transform = transform_resize
    if augmentation == Augmentation.FREQUENCY_MASKING:
        transform = transform_frequency_masking
    elif augmentation == Augmentation.TIME_MASKING:
        transform = transform_time_masking
    elif augmentation == Augmentation.COMBINED_MASKING:
        transform = transform_combined_masking
    elif augmentation == Augmentation.PAD_ZEROS:
        transform = transform_pad_zeros
        val_transform = transform_pad_zeros
    elif augmentation == Augmentation.RESIZE:
        transform = transform_resize
    else:
        transform = transforms.Compose([])
        val_transform = transform
    if num_splits == 1:
        skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=random_state)
        cv_iterable = skf.split(patients_ids, [label for _, label in patients_ids])
        next(cv_iterable)
    else:
        skf = StratifiedKFold(
            n_splits=num_splits, shuffle=True, random_state=random_state
        )
        cv_iterable = skf.split(patients_ids, [label for _, label in patients_ids])
    return cv_iterable, transform, val_transform, patients_ids, file_paths
