import re
from functools import partial
from itertools import product

import numpy as np
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from PIL import Image
from torchvision import transforms, models
import torch
import torch.nn.functional as F

from Config import Config
from Evaluation.get_paths import get_paths
from Evaluation.utilities import get_patients_id, get_files_path
from Models import SpectrogramDataset
from Models.model_adjustment import adjust
import matplotlib.pyplot as plt

def main():
    batch_size = 1
    model = adjust(partial(models.vgg19, num_classes=1), multichannel=False, window=False)()
    # model.load_state_dict(torch.load())
    for vowels in [["a"], ["i"], ["u"], ["all"]]:
        file_paths = set(get_files_path(vowels))
        healthy_patient_ids = set(get_patients_id(vowels, health=True))
        healthy_paths = set(path for path in file_paths if re.findall("\d+", path)[0] in healthy_patient_ids)
        # diseased_paths = file_paths - healthy_paths
        transformations = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((224, 224), antialias=None),
            ]
        )
        spectrograms = torch.tensor(np.array(tuple(map(transformations, map(Image.fromarray, map(SpectrogramDataset([]).audio_file2spectrogram, healthy_paths))))))
        target_layers = [model.features[-1]]

        targets = [ClassifierOutputTarget(0)]
        for cam_method in (
                ScoreCAM,
                AblationCAM,
                GradCAM,
        ):
            file_path = Config.grad_cam_path.joinpath(cam_method.__name__ + "".join(vowels) + ".png")
            if file_path.exists():
                continue
            with cam_method(model=model, target_layers=target_layers) as cam:
                grayscale_cam = np.mean(np.concatenate(tuple(cam(input_tensor=spectrograms[i:i+batch_size], targets=targets) for i in range(0, len(spectrograms), batch_size))), axis=0) * 255
                array = grayscale_cam.astype(np.uint8)
                img = Image.fromarray(array)
                img.save(file_path)

if __name__ == '__main__':
    main()
