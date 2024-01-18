from typing import Literal

from sklearn.model_selection import StratifiedKFold

from Config import Config
from Evaluation.utilities import (
    get_patients_id,
    get_files_path,
)


def get_paths(
    vowels: list[Literal["a", "i", "u", "all"]],
    num_splits: int,
    random_state: int = 42,
):
    patients_ids = tuple(get_patients_id(vowels))
    file_paths = tuple(get_files_path(vowels))
    healthy_patient_ids = set(get_patients_id(vowels, health=True))
    if num_splits == 1:
        skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=random_state)
        cv_iterable = skf.split(
            patients_ids,
            [int(patient_id not in healthy_patient_ids) for patient_id in patients_ids],
        )
        next(cv_iterable)
    else:
        skf = StratifiedKFold(
            n_splits=num_splits, shuffle=True, random_state=random_state
        )
        cv_iterable = skf.split(
            patients_ids,
            [
                int(patient_id not in Config.healthy_patients_folder.iterdir())
                for patient_id in patients_ids
            ],
        )
    return cv_iterable, patients_ids, file_paths
