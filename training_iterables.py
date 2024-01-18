from itertools import chain, product

from Config import Config, WindowParameters
from Models.Augmentations import Augmentation

window_iterable = product(
    (
        WindowParameters(
            use_window=True, window_size=40, window_stride=10
        ),
    ),
    (Augmentation.PAD_ZEROS,),
)

traditional_iterable = product(
    (WindowParameters(use_window=False),),
    (
        Augmentation.FREQUENCY_MASKING,
        Augmentation.TIME_MASKING,
        Augmentation.COMBINED_MASKING,
        Augmentation.RESIZE,
    ),
)


spectrogram_training_iterable = product(
    Config.base_models,
    (["a"], ["u"], ["i"], ["a", "u", "i"], ["all"]),
    chain.from_iterable((window_iterable, traditional_iterable,)),
)
