import re
from dataclasses import dataclass


@dataclass
class Result:
    f1: float
    model: str
    multichannel: bool
    window: bool
    vowel: str
    augmentation: str

    @classmethod
    def from_file_name(cls, file_name: str) -> "Result":
        return Result(
            f1=float(re.findall(r'f1\_(0\.\d+)', file_name)[0]),
            multichannel='MultiChannel' in file_name,
            window='Window' in file_name,
            vowel=re.findall(r'\_([auil]+)_', file_name)[0],
            augmentation=re.findall(r'\.([A-Z\_]+)\.pth', file_name)[0].replace('_', ' '),
            model=re.findall(r'(Window|Traditional)([A-Za-z]+)\_', file_name)[0][1]
        )