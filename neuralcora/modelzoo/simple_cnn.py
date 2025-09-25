"""Simple CNN architectures for NeuralCora."""
from __future__ import annotations

import torch
from torch import nn


class SimpleCNNForecaster(nn.Module):
    """Baseline fully-convolutional forecaster treating time steps as channels."""

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        hidden_channels: tuple[int, ...] | list[int] = (32, 64, 64),
        kernel_size: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if isinstance(hidden_channels, list):
            hidden_channels = tuple(hidden_channels)
        padding = kernel_size // 2
        layers: list[nn.Module] = []
        in_channels = input_channels
        for hidden in hidden_channels:
            layers.append(
                nn.Conv2d(
                    in_channels,
                    hidden,
                    kernel_size=kernel_size,
                    padding=padding,
                    bias=True,
                )
            )
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout2d(p=dropout))
            in_channels = hidden
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Conv2d(in_channels, output_channels, kernel_size=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 4:
            raise ValueError(
                f"Expected input tensor of shape (batch, channels, height, width), got {inputs.shape}"
            )
        features = self.backbone(inputs)
        return self.head(features)
