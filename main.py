import torch

from Config import Config
from Evaluation.training_validation import training_validation
from Models.model_adjustment import adjust
from training_iterables import spectrogram_training_iterable

if __name__ == "__main__":
    # device = check_cuda_availability()
    device: torch.device = None

    for (
        model_creation_function,
        vowels,
        (window_arguments, augmentation),
    ) in spectrogram_training_iterable:
        many_channels = vowels[0] == 'all'
        model_creation_function = adjust(
            model_creation_function, many_channels, *window_arguments
        )
        training_validation(
            device=device,
            vowels=vowels,
            batch_size=Config.batch_size,
            num_splits=Config.num_splits,
            early_stopping_patience=Config.early_stopping_patience,
            criterion=Config.criterion,
            model_creator=model_creation_function,
            learning_rate=Config.learning_rate,
            random_state=Config.random_state,
            augmentation=augmentation,
        )
