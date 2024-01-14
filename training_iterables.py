from itertools import chain, product

from Config import Config, WindowParameters
from Models import SpectrogramDataset
from Models.WaveformDataset import WaveformDataset

waveform_training_iterable = (
    (1, WaveformDataset, False, Config.waveform_model_creator, (WindowParameters(use_window=False), None)),)

window_iterable = product(
    (
        WindowParameters(
            use_window=True, window_size=40, window_stride=10
        ),
    ),
    ("pad_zeros",),
)

traditional_iterable = product(
    (WindowParameters(use_window=False),),
    (
        "frequency_masking",
        "time_masking",
        "combined_masking",
        "resize",
    ),
)

spectrogram_training_iterable = product(
    (Config.batch_size,),
    (SpectrogramDataset,),
    (False, True),
    Config.base_models,
    chain.from_iterable((traditional_iterable, window_iterable,)),
)
