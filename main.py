from itertools import chain

from Config import Config
from Evaluation.training_validation import training_validation
from Evaluation.utilities import check_cuda_availability
from Models.model_adjustment import adjust
from training_iterables import waveform_training_iterable, spectrogram_training_iterable

if __name__ == "__main__":
    device = check_cuda_availability()

    for (
        batch_size,
        spectrogram_dataset,
        many_channels,
        (model_creation_function, learning_rate),
        (window_arguments, augmentation),
    ) in chain.from_iterable((waveform_training_iterable, spectrogram_training_iterable)):
        model_creation_function = adjust(
            model_creation_function, many_channels, *window_arguments
        )
        file_path = (
            Config.lists_path
            / f'Vowels_{"all" if many_channels else Config.vowel}_{Config.disease}.txt'
        )
        training_validation(
            device=device,
            file_path=file_path,
            batch_size=batch_size,
            num_splits=Config.num_splits,
            early_stopping_patience=Config.early_stopping_patience,
            criterion=Config.criterion,
            model_creator=model_creation_function,
            learning_rate=learning_rate,
            random_state=Config.random_state,
            augmentation=augmentation,
            dataset_type=spectrogram_dataset
        )
