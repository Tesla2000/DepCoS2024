import re
from functools import partial

import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM, ScoreCAM, AblationCAM
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget
from torchvision import transforms, models
from tqdm import tqdm

from Config import Config
from Evaluation.utilities import get_patients_id, get_files_path, \
    check_cuda_availability
from Models import SpectrogramDataset
from Models.model_adjustment import adjust


def main():
    batch_size = 1
    model = adjust(partial(models.vgg19, num_classes=1), multichannel=False, window=False)()
    model_path = Config.results_folder.joinpath("f1_0.73_SingleChannelTraditionalVGG_aui_Augmentation.RESIZE.pth")
    model.load_state_dict(torch.load(model_path, map_location=check_cuda_availability()))
    vowels = re.findall(r"_([auil]+)_", model_path.name)[0]
    file_paths = set(get_files_path(vowels))
    healthy_patient_ids = set(get_patients_id(vowels, health=True))
    healthy_paths = set(path for path in file_paths if re.findall("\d+", path)[0] in healthy_patient_ids)
    diseased_paths = file_paths - healthy_paths
    transformations = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((224, 224), antialias=None),
        ]
    )
    target_layers = [model.features[-1]]

    for state, category, paths in (("health", -1, healthy_paths), ("diseased", 1, diseased_paths)):
        spectrograms = torch.tensor(np.array(tuple(map(transformations, map(Image.fromarray, map(SpectrogramDataset([]).audio_file2spectrogram, paths))))))
        targets = [BinaryClassifierOutputTarget(category),]
        for cam_method in (
                ScoreCAM,
                AblationCAM,
                GradCAM,
        ):
            file_path = Config.grad_cam_path.joinpath(f"{model.__name__}_{cam_method.__name__}_{state}_{''.join(vowels)}.png")
            if file_path.exists():
                continue
            with cam_method(model=model, target_layers=target_layers) as cam:
                grayscale_cam = np.mean(np.concatenate(tuple(cam(input_tensor=spectrograms[i:i+batch_size], targets=targets) for i in tqdm(range(0, len(spectrograms), batch_size)))), axis=0) * 255
                array = grayscale_cam.astype(np.uint8)
                img = Image.fromarray(array)
                img.save(file_path)

if __name__ == '__main__':
    main()
