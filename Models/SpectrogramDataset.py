import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import librosa
import librosa.display
import matplotlib
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import Compose

matplotlib.use("agg")


@dataclass
class Sample:
    label: int = 0
    audio_paths: list[Path] = field(default_factory=list)
    log_mel_spec_dbs: list[Tensor] = field(default_factory=list)


class SpectrogramDataset(Dataset):
    """
    A custom PyTorch dataset for handling audio spectrogram data.
    """
    spectrograms = dict()

    def __init__(
        self,
        paths_to_audio,
        transforms: tuple[Callable, Compose] = None,
        hop_length=512,
        n_fft=2048,
        n_mels=128,
        fmin=0,
        fmax=None,
        split_channels: bool = False,
    ):
        """
        Initializes the dataset.

        Args:
            paths_to_audio (list): List of file paths to audio files.
            transforms: Transformation to be applied first to audio, second to spectrogram.
            hop_length (int): Number of samples between successive frames.
            n_fft (int): Number of samples in each window.
            n_mels (int): Number of mel filterbanks.
            fmin (float): Minimum frequency.
            fmax (float): Maximum frequency.
        """
        self.paths_to_audio = paths_to_audio
        if transforms is None:
            transforms = lambda x: x, None
        self.audio_transform, self.spectrogram_transform = transforms
        self.samples = defaultdict(Sample)  # Dictionary to store sample information
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax

        # Extract and store sample information
        for audio_path in set(self.paths_to_audio):
            sample_id = re.findall(r"\d+_\w" if split_channels else r"\d+", audio_path)[
                0
            ]

            self.samples[sample_id].label = int("Healthy" not in audio_path)
            self.samples[sample_id].audio_paths.append(audio_path)

    def __len__(self):
        """
        Returns the number of samples in the dataset.

        Returns:
            int: The number of samples.
        """
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Gets an item (spectrogram and label) from the dataset.

        Args:
            idx (int): Index of the sample in the dataset.

        Returns:
            torch.Tensor: Spectrogram tensor.
            int: Sample label.
        """
        sample_id = list(self.samples.keys())[idx]
        audio_paths = self.samples[sample_id].audio_paths
        label = self.samples[sample_id].label

        self.samples[sample_id].log_mel_spec_dbs = []
        for audio_path in sorted(audio_paths):
            if self.spectrograms.get(audio_path):
                scaled_img = np.copy(self.spectrograms[audio_path])
            else:
                y, sr = librosa.load(audio_path, sr=None)
                y = self.audio_transform(y)
                mel_spec = librosa.feature.melspectrogram(
                    y=y,
                    sr=sr,
                    hop_length=self.hop_length,
                    n_fft=self.n_fft,
                    n_mels=self.n_mels,
                    fmin=self.fmin,
                    fmax=self.fmax,
                )
                log_mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
                min_val = log_mel_spec_db.min()
                max_val = log_mel_spec_db.max()
                scaled_img = (log_mel_spec_db - min_val) / (max_val - min_val)
                self.spectrograms[audio_path] = scaled_img.copy()

            if self.spectrogram_transform:
                pil_img = Image.fromarray(scaled_img)

                pil_tensor = transforms.ToTensor()(pil_img)

                pil_tensor = self.spectrogram_transform(pil_tensor)

                log_mel_spec_db = np.array(pil_tensor)

            self.samples[sample_id].log_mel_spec_dbs.append(Tensor(log_mel_spec_db))

        return torch.concat(self.samples[sample_id].log_mel_spec_dbs), label


if __name__ == "__main__":
    dataset = SpectrogramDataset(["Data/Vowels/Dysphonie/368_a.wav 1"])
    spectrogram, label = dataset[0]
    librosa.display.specshow(spectrogram.numpy(), cmap="plasma")
    # plt.savefig('spec.jpg')
    # plt.show()
