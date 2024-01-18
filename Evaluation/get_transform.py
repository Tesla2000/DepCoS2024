from typing import Literal

import torch
import torchaudio.transforms as T
import torchvision.transforms as transforms

from Models.Augmentations import Augmentation


def get_transform(
    augmentation: Augmentation,
):
    # Load patient IDs and file paths from a file

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
    return transform, val_transform
