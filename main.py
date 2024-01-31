import os
from datetime import datetime
from zipfile import ZipFile

from Config import Config
from Evaluation.training_validation import training_validation
from Evaluation.utilities import check_cuda_availability
from Models.model_adjustment import adjust
from training_iterables import spectrogram_training_iterable

if __name__ == "__main__":
    if not Config.vowels_path.exists():
        print("Downloading data...")
        zip_path = Config.data_path / "Vowels.zip"
        os.system(
            f"gdown https://drive.google.com/uc?id={Config.google_drive_file_id} -O {zip_path}"
        )
        with ZipFile(zip_path) as zip_ref:
            zip_ref.extractall(Config.data_path)
            os.remove(zip_path)
        print("Files downloaded")

    if Config.device is None:
        Config.device = check_cuda_availability()
    project_name = datetime.now().strftime('%Y%m%d%H%M')

    for (
        model_creation_function,
        vowels,
        (window_arguments, augmentation),
    ) in spectrogram_training_iterable:
        many_channels = vowels[0] == "all"
        model_creation_function = adjust(
            model_creation_function, many_channels, *window_arguments
        )
        training_validation(
            device=Config.device,
            vowels=vowels,
            batch_size=Config.batch_size,
            num_splits=Config.num_splits,
            early_stopping_patience=Config.early_stopping_patience,
            criterion=Config.criterion,
            model_creator=model_creation_function,
            learning_rate=Config.learning_rate,
            learning_rate_scheduler_creator=Config.learning_rate_scheduler_creator,
            random_state=Config.random_state,
            augmentation=augmentation,
            project_name=project_name,
        )
