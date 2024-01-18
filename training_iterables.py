from itertools import chain, product

from Config import Config, WindowParameters
from Models.Augmentations import Augmentation

window_iterable = product(
    (
        WindowParameters(
            use_window=True,
            window_size=Config.window_size,
            window_stride=Config.window_stride,
        ),
    ),
    (
        Augmentation.PAD_ZEROS,
        Augmentation.ADD_NOISE_AND_PAD,
    ),
)

traditional_iterable = product(
    (WindowParameters(use_window=False),),
    (
        Augmentation.FREQUENCY_MASKING,
        Augmentation.TIME_MASKING,
        Augmentation.COMBINED_MASKING,
        Augmentation.RESIZE,
        Augmentation.ADD_NOISE,
    ),
)


spectrogram_training_iterable = product(
    Config.model_creators,
    (
        ["a"],
        ["u"],
        ["i"],
        ["all"],
        ["a", "u", "i"],
    ),
    chain.from_iterable(
        (
            traditional_iterable,
            window_iterable,
        )
    ),
)
