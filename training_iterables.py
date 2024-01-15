from itertools import chain, product, repeat

from Config import Config, WindowParameters
from Models import SpectrogramDataset
from Models.LeNet5 import LeNet5
from Models.WaveformDataset import WaveformDataset

waveform_training_iterable = (
    (1, WaveformDataset, False, (Config.waveform_model_creator, 1e-5), (WindowParameters(use_window=False), None)),)

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
    chain.from_iterable((
        ((LeNet5, 1e-4),),
        zip(Config.base_models, repeat(1e-5))
    )),
    chain.from_iterable((traditional_iterable, window_iterable,)),
)
