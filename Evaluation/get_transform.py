from typing import Callable

import numpy as np
import torch
import torchaudio.transforms as T
import torchvision.transforms as transforms
from torchvision.transforms import Compose

from Config import Config
from Models.Augmentations import Augmentation


def get_transform(
    augmentation: Augmentation,
) -> tuple[tuple[Callable, Compose], tuple[Callable, Compose]]:
    audio_transform = audio_val_transform = lambda y: y
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

    val_spectrogram_transform = transform_resize
    if augmentation == Augmentation.FREQUENCY_MASKING:
        spectrogram_transform = transform_frequency_masking
    elif augmentation == Augmentation.TIME_MASKING:
        spectrogram_transform = transform_time_masking
    elif augmentation == Augmentation.COMBINED_MASKING:
        spectrogram_transform = transform_combined_masking
    elif augmentation in (Augmentation.PAD_ZEROS, Augmentation.ADD_NOISE_AND_PAD):
        spectrogram_transform = transform_pad_zeros
        val_spectrogram_transform = transform_pad_zeros
    elif augmentation in (Augmentation.RESIZE, Augmentation.ADD_NOISE):
        spectrogram_transform = transform_resize
    else:
        spectrogram_transform = transforms.Compose([])
        val_spectrogram_transform = spectrogram_transform
    if augmentation in (Augmentation.ADD_NOISE, Augmentation.ADD_NOISE_AND_PAD):
        audio_transform = lambda y: y + Config.sigma * np.mean(y) * np.random.normal(
            size=y.shape
        )
    return (audio_transform, spectrogram_transform), (
        audio_val_transform,
        val_spectrogram_transform,
    )
