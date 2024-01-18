import random
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
from torch import nn
from torch.optim import lr_scheduler
from torchvision import models


class Config:
    device = None
    sigma = 0.05
    gamma = 0.9
    window_stride = 10
    window_size = 40
    diseases = (
        "Rekurrensparese",
        "Dysphonie",
        "Funktionelle Dysphonie",
        "Hyperfunktionelle Dysphonie",
        "Laryngitis",
    )
    base_models = (
        partial(models.vgg19, num_classes=1),
        models.resnet101,
    )
    criterion = nn.BCELoss()
    num_splits = 1
    early_stopping_patience = 3
    batch_size = 16
    learning_rate_scheduler_creator = lambda optimizer: lr_scheduler.ExponentialLR(optimizer, gamma=.9)
    learning_rate = 1e-4
    root_path = Path(".")
    data_path = root_path / "Data"
    session_time = datetime.now().strftime("%Y%m%d%H%M")
    results_folder = data_path.joinpath("results")
    results_folder.mkdir(exist_ok=True, parents=True)
    summary_folder = data_path.joinpath(f"summaries/{session_time}")
    summary_folder.mkdir(exist_ok=True, parents=True)
    lists_path = data_path / "Lists"
    vowels_path = data_path / "Vowels"
    healthy_patients_folder = vowels_path / "Healthy"
    random_state = 42


torch.manual_seed(Config.random_state)
np.random.seed(Config.random_state)
random.seed(Config.random_state)
Config.data_path.mkdir(exist_ok=True, parents=True)
Config.lists_path.mkdir(exist_ok=True, parents=True)


class WindowParameters(NamedTuple):
    use_window: bool = True
    window_size: int = 40
    window_stride: int = 10
