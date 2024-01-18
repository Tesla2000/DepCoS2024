import os
from zipfile import ZipFile

import wandb

from Config import Config
from Evaluation.training_validation import training_validation
from Evaluation.utilities import check_cuda_availability
from Models.model_adjustment import adjust
from training_iterables import spectrogram_training_iterable

if __name__ == "__main__":
    if not Config.vowels_path.exists():
        print("Downloading data...")
        zip_path = Config.data_path / "Vowels.zip"
        os.system(f"gdown https://drive.google.com/uc?id=1G1lCMR6hhW3BT8FWHekvHEXhxTf8yiqw -O {zip_path}")
        with ZipFile(zip_path) as zip_ref:
            zip_ref.extractall(Config.data_path)
            os.remove(zip_path)
        print("Files downloaded")

    device = check_cuda_availability()
    # device: torch.device = None

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
            gamma=Config.gamma
        )
