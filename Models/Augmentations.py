from enum import Enum


class Augmentation(Enum):
    ADD_NOISE = "add_noise"
    ADD_NOISE_AND_PAD = "add_noise_and_pad"
    PAD_ZEROS = "pad_zeros"
    TIME_MASKING = "time_masking"
    FREQUENCY_MASKING = "frequency_masking"
    COMBINED_MASKING = "combined_masking"
    RESIZE = "resize"
