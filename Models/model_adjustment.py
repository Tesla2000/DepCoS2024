from functools import partial
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch import tensor, Tensor
from torchvision.models import VGG, ResNet, EfficientNet


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
                first_layer[0].bias,
                first_layer[0].padding_mode,
            )
            model.features[0] = first_layer
        elif not isinstance(model, VGG):
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
