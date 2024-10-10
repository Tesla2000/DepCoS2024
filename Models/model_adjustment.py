from functools import partial
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from timm.models import VisionTransformer
from torch import tensor, Tensor
from torchvision.models import VGG, ResNet, EfficientNet, DenseNet, RegNet
import torch.nn.functional as F


def _window_forward_wrapper(forward, window_size: int, window_stride: int):
    def inner(x):
        results = torch.empty(len(x))
        device = x.device
        for index, sample in enumerate(x):
            sample = sample.cpu()
            for i in range(1, 500):
                if torch.sum(sample[:, :, -i]) != 0:
                    sample = sample[:, :, :max(window_size, sample.shape[-1] - i)]
                    break
            windows = tensor(
                np.array(
                    tuple(
                        sample[:, :, i : i + window_size].numpy()
                        for i in range(
                            0, sample.shape[-1] - window_size + 1, window_stride
                        )
                    )
                )
            ).to(device)
            windows = F.interpolate(windows,
                                           size=(224, 224),
                                           mode='bilinear',
                                           align_corners=False)

            windows = forward(windows)
            mean_windows = torch.mean(windows)
            results[index] = torch.sigmoid(mean_windows)
        return results.unsqueeze(1).to(device)

    return inner


def _forward_wrapper(forward_function):
    def inner(x):
        return torch.sigmoid(forward_function(x))

    return inner


def adjust(
    model_creation_function: Callable[[], nn.Module],
    multichannel: bool,
    window: bool,
    window_size: int = None,
    window_stride: int = None,
) -> Callable[[], nn.Module]:
    def wrapper():
        model = model_creation_function()
        model.__name__ = "MultiChannel" if multichannel else "SingleChannel"
        if isinstance(model, ResNet) and (
            (multichannel and model.conv1.in_channels != 3)
            or (not multichannel and model.conv1.in_channels != 1)
        ):
            model.conv1 = nn.Conv2d(
                3 if multichannel else 1,
                out_channels=model.conv1.out_channels,
                kernel_size=model.conv1.kernel_size,
                stride=model.conv1.stride,
                padding=model.conv1.padding,
                bias=isinstance(model.conv1.bias, Tensor),
            )
        elif isinstance(model, VGG) and (
            (multichannel and model.features[0].in_channels != 3)
            or (not multichannel and model.features[0].in_channels != 1)
        ):
            model.features[0] = nn.Conv2d(
                1 + 2 * multichannel,
                out_channels=model.features[0].out_channels,
                kernel_size=model.features[0].kernel_size,
                stride=model.features[0].stride,
                padding=model.features[0].padding,
                bias=isinstance(model.features[0].bias, Tensor),
            )
        elif isinstance(model, EfficientNet):
            last = model.classifier[-1]
            model.classifier[-1] = nn.Linear(
                in_features=last.in_features,
                out_features=1,
                bias=last.bias is not None,
            )
            first_layer = model.features[0]
            first_layer[0] = nn.Conv2d(
                1 + 2 * multichannel,
                first_layer[0].out_channels,
                first_layer[0].kernel_size,
                first_layer[0].stride,
                first_layer[0].padding,
                first_layer[0].dilation,
                first_layer[0].groups,
                first_layer[0].bias is not None,
                first_layer[0].padding_mode,
            )
            model.features[0] = first_layer
        elif isinstance(model, DenseNet):
            model.classifier = nn.Linear(
                in_features=model.classifier.in_features,
                out_features=1,
                bias=model.classifier.bias is not None,
            )
            first_layer = model.features[0]
            model.features[0] = nn.Conv2d(
                1 + 2 * multichannel,
                first_layer.out_channels,
                first_layer.kernel_size,
                first_layer.stride,
                first_layer.padding,
                first_layer.dilation,
                first_layer.groups,
                first_layer.bias is not None,
                first_layer.padding_mode,
            )
        elif isinstance(model, RegNet):
            model.fc = nn.Linear(
                in_features=model.fc.in_features,
                out_features=1,
                bias=model.fc.bias is not None,
            )
            model.stem[0] = nn.Conv2d(
                1 + 2 * multichannel,
                model.stem[0].out_channels,
                model.stem[0].kernel_size,
                model.stem[0].stride,
                model.stem[0].padding,
                model.stem[0].dilation,
                model.stem[0].groups,
                model.stem[0].bias is not None,
                model.stem[0].padding_mode,
            )
        elif isinstance(model, DenseNet):
            model.classifier = nn.Linear(
                in_features=model.classifier.in_features,
                out_features=1,
                bias=model.classifier.bias is not None,
            )
            first_layer = model.features[0]
            model.features[0] = nn.Conv2d(
                1 + 2 * multichannel,
                first_layer.out_channels,
                first_layer.kernel_size,
                first_layer.stride,
                first_layer.padding,
                first_layer.dilation,
                first_layer.groups,
                first_layer.bias is not None,
                first_layer.padding_mode,
            )
        elif isinstance(model, VisionTransformer):
            model.patch_embed.proj = nn.Conv2d(
                1 + 2 * multichannel,
                model.patch_embed.proj.out_channels,
                model.patch_embed.proj.kernel_size,
                model.patch_embed.proj.stride,
                model.patch_embed.proj.padding,
                model.patch_embed.proj.dilation,
                model.patch_embed.proj.groups,
                model.patch_embed.proj.bias is not None,
                model.patch_embed.proj.padding_mode,
            )
        if isinstance(model, ResNet):
            model.fc = nn.Linear(model.fc.in_features, 1)
        model.forward = (
            partial(
                _window_forward_wrapper,
                window_size=window_size,
                window_stride=window_stride,
            )
            if window
            else _forward_wrapper
        )(model.forward)
        model.__name__ += "Window" if window else "Traditional"
        model.__name__ += type(model).__name__
        return model

    return wrapper
