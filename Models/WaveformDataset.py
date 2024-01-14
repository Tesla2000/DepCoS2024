import os
from collections import defaultdict
from dataclasses import dataclass

import librosa
import librosa.display
import matplotlib
import torch
from torch import Tensor
from torch.utils.data import Dataset

matplotlib.use("agg")


@dataclass
class Sample:
    label: int
    waveform: Tensor


class WaveformDataset(Dataset):
    """
    A custom PyTorch dataset for handling audio spectrogram data.
    """

    def __init__(
        self,
        paths_to_audio,
        transform=None,
        hop_length=512,
        n_fft=2048,
        n_mels=128,
        fmin=0,
        fmax=None,
        sample_rate=32000,
    ):
        """
        Initializes the dataset.

        Args:
            paths_to_audio (list): List of file paths to audio files.
            transform (callable): A function/transform to apply to the spectrogram data.
            hop_length (int): Number of samples between successive frames.
            n_fft (int): Number of samples in each window.
            n_mels (int): Number of mel filterbanks.
            fmin (float): Minimum frequency.
            fmax (float): Maximum frequency.
        """
        self.paths_to_audio = paths_to_audio
        self.transform = transform
        self.samples = defaultdict(Sample)  # Dictionary to store sample information
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax

        # Extract and store sample information
        for path in set(self.paths_to_audio):
            audio_path, label = path.split(" ")
            sample_id = os.path.splitext(os.path.basename(audio_path))[0]
            sample_id = sample_id.split("_")[0]
            (waveform, _) = librosa.core.load(audio_path, sr=sample_rate, mono=True)

            self.samples[sample_id] = Sample(int(label), Tensor(waveform))

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
        return self.samples[sample_id].waveform, self.samples[sample_id].label


if __name__ == "__main__":
    dataset = WaveformDataset(["Data/Vowels/Dysphonie/368_a.wav 1"])
    waveform, label = dataset[0]
