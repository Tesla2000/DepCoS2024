"""File contains additional utilities"""
import operator
import re
from functools import reduce
from typing import Literal

import torch as tc

from Config import Config


def check_cuda_availability():
    """
    Checks if a CUDA-compatible GPU is available.

    Returns:
        tc.device: A PyTorch device object set to "cuda" if CUDA is available,
        or "cpu" if not.
    """
    return tc.device("cuda" if tc.cuda.is_available() else "cpu")


# Function to move data to a specified device
def to_device(data, device):
    """
    Moves PyTorch tensors or data structures to a specified device.

    Args:
        data: Input data to move to the device.
        device: The target device ("cuda" for GPU or "cpu" for CPU).

    Returns:
        data: Input data moved to the specified device.
    """
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


# Function to get file paths from a text file
def get_files_path(
    vowels: list[Literal["a", "i", "u", "all"]], health: bool = False
) -> set[str]:
    if health:
        return set(
            dict(
                (audio_file.name, str(audio_file))
                for vowel in vowels
                for audio_file in Config.healthy_patients_folder.iterdir()
                if re.findall(rf"_{vowel}", audio_file.name) or vowel == "all"
            ).values()
        )
    return set(
        dict(
            (audio_file.name, str(audio_file))
            for vowel in vowels
            for data_folder in Config.vowels_path.iterdir()
            for audio_file in data_folder.iterdir()
            if re.findall(rf"_{vowel}", audio_file.name) or vowel == "all"
        ).values()
    )


# Function to get a list of unique patient IDs from a text file
def get_patients_id(
    vowels: list[Literal["a", "i", "u", "all"]], health: bool = False
) -> set[str]:
    return set(
        reduce(
            operator.add,
            map(re.compile(r"\d+").findall, get_files_path(vowels, health)),
        )
    )
