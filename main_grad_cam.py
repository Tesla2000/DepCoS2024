import re

import cv2
import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM, ScoreCAM, AblationCAM
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget
from torchvision import transforms
from tqdm import tqdm

from Config import Config
from Evaluation.utilities import get_files_path, \
    check_cuda_availability
from Models import SpectrogramDataset
from Models.model_adjustment import adjust
import torch

def main():
    batch_size = 1
    model_paths = (
        Config.results_folder.joinpath("f1_0.73_SingleChannelTraditionalVGG_aui_Augmentation.RESIZE.pth"), # vgg
        # Config.results_folder.joinpath("f1_0.72_SingleChannelTraditionalResNet18_a_Augmentation.FREQUENCY_MASKING.pth"), # resnet18
        Config.results_folder.joinpath("f1_0.69_SingleChannelTraditionalResNet101_a_Augmentation.FREQUENCY_MASKING.pth"), # resnet101
        # Config.results_folder.joinpath("f1_0.73_SingleChannelTraditionalDenseNet_i_Augmentation.TIME_MASKING.pth"), # densenet121
        # Config.results_folder.joinpath(""), # efficientnet_b2
        # Config.results_folder.joinpath(""), # regnet_x_3_2gf
        # Config.results_folder.joinpath(""), # deit
    )
    target_layer_generators = [
        lambda model: model.features[-1],  # vgg
        # lambda model: model.layer4[-1],  # resnet18
        lambda model: model.layer4[-1],  # resnet101
        # lambda model: model.features[-1],  # densenet121
        # lambda model: model.features[-1],  # efficientnet_b2
        # lambda model: getattr(model.trunk_output.block4, "block4-1").f.b,  # regnet_x_3_2gf
        # lambda model: model.blocks[-1].norm1,  # deit
    ]
    n_samples = 50
    for model_creation_function, model_path, target_layer_generator in zip(Config.model_creators, model_paths, target_layer_generators):
        model = adjust(model_creation_function, multichannel=False, window=False)()
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cuda:1")))
        vowels = re.findall(r"_([auil]+)_", model_path.name)[0]
        file_paths = set(get_files_path(vowels))
        healthy_paths = set(path for path in file_paths if "Healthy" in path)
        diseased_paths = file_paths - healthy_paths
        transformations = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((224, 224), antialias=None),
            ]
        )

        for state, category, paths in (
                ("health", -1, healthy_paths),
                ("diseased", 1, diseased_paths),
        ):
            spectrograms = torch.tensor(np.array(tuple(map(transformations, map(Image.fromarray, map(SpectrogramDataset([]).audio_file2spectrogram, paths))))))
            targets = [BinaryClassifierOutputTarget(category),]
            for cam_method in (
                    GradCAM,
                    ScoreCAM,
                    AblationCAM,
            ):
                file_path = Config.grad_cam_path.joinpath(f"{model.__name__}_{cam_method.__name__}_{state}_{''.join(vowels)}.png")
                if file_path.exists():
                    continue
                with cam_method(model=model, target_layers=[target_layer_generator(model)]) as cam:
                    grayscale_cam = np.mean(np.concatenate(tuple(cam(input_tensor=spectrograms[i:i+batch_size], targets=targets, eigen_smooth=True) for i in tqdm(range(0, min(n_samples, len(spectrograms)), batch_size)))), axis=0)
                    heatmap = cv2.applyColorMap(np.uint8(255 * grayscale_cam), cv2.COLORMAP_JET)
                    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(heatmap)
                    img.save(file_path)

if __name__ == '__main__':
    main()
