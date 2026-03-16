"""Mini VGG-style CNN backbone for OCR.

Author: David
"""

import torch.nn as nn


class MiniVggCNN(nn.Module):
    """Compact VGG-like CNN that maps grayscale images to sequence features."""

    def __init__(self, input_channel: int = 1, output_channel: int = 512) -> None:
        super().__init__()

        channels = [64, 128, 256, 256, 512, 512, output_channel]
        self.convnet = nn.Sequential(
            nn.Conv2d(input_channel, channels[0], 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(channels[0], channels[1], 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(channels[1], channels[2], 3, 1, 1),
            nn.BatchNorm2d(channels[2]),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels[2], channels[3], 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2), (2, 1), (0, 1)),
            nn.Conv2d(channels[3], channels[4], 3, 1, 1),
            nn.BatchNorm2d(channels[4]),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels[4], channels[5], 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2), (2, 1), (0, 1)),
            nn.Conv2d(channels[5], channels[6], 2, 1, 0),
            nn.BatchNorm2d(channels[6]),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.convnet(x)
