"""Mini CRNN model for OCR.

Author: David
"""

import torch.nn as nn

from ocr.model.encoder import Encoder
from ocr.model.minicnn import MiniCNN


class MiniCRNN(nn.Module):
    """A compact CRNN model with VGG backbone and RNN encoder."""

    def __init__(
        self,
        charset_size: int,
        backbone: str,
        encoder_type: str,
        encoder_input_size: int,
        encoder_hidden_size: int,
        encoder_layers: int,
        encoder_bidirectional: bool,
        max_seq_len: int,
    ) -> None:
        super().__init__()

        if encoder_type not in {"gru", "lstm"}:
            raise ValueError(f"Unsupported encoder_type: {encoder_type}. Expected 'gru' or 'lstm'.")

        self.max_seq_len = max_seq_len
        self.encoder_type = encoder_type

        self.cnn = MiniCNN(
            backbone=backbone,
            output_channel=encoder_input_size,
        )
        self.encoder = Encoder(
            rnn_type=encoder_type,
            input_size=encoder_input_size,
            hidden_size=encoder_hidden_size,
            num_layers=encoder_layers,
            bidirectional=encoder_bidirectional,
        )

        decoder_input_size = encoder_hidden_size * 2 if encoder_bidirectional else encoder_hidden_size
        self.decoder = nn.Linear(decoder_input_size, charset_size)

    def forward(self, x):
        # CNN output: (B, C, H, W) -> sequence: (B, W, C)
        x = self.cnn(x)
        x = x.mean(dim=2).permute((0, 2, 1))

        output, _ = self.encoder(x)
        output = self.decoder(output)
        return output
