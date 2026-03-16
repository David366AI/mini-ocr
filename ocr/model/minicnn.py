"""Feature extraction module for mini OCR models.

Author: David
"""

import torch.nn as nn

from ocr.model.vgg16 import MiniVggCNN


class MiniCNN(nn.Module):
    """Wrapper over supported CNN backbones.

    Only `vgg` is supported in this project.
    """

    def __init__(
        self,
        backbone: str = "vgg",
        input_channel: int = 1,
        output_channel: int = 512,
    ) -> None:
        super().__init__()

        if backbone != "vgg":
            raise ValueError(f"Only 'vgg' backbone is supported, got: {backbone}")

        self.cnn = MiniVggCNN(
            input_channel=input_channel,
            output_channel=output_channel,
        )

    def forward(self, x):
        return self.cnn(x)
