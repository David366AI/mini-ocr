"""Sequence encoder module for mini OCR.

Author: David
"""

import torch
import torch.nn as nn


class Encoder(nn.Module):
    """RNN encoder supporting GRU and LSTM."""

    def __init__(
        self,
        rnn_type: str,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        bidirectional: bool,
    ) -> None:
        super().__init__()

        if rnn_type == "gru":
            self.rnn = nn.GRU(
                batch_first=True,
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                bidirectional=bidirectional,
            )
        elif rnn_type == "lstm":
            self.rnn = nn.LSTM(
                batch_first=True,
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                bidirectional=bidirectional,
            )
        else:
            raise ValueError(f"Unsupported rnn_type: {rnn_type}")

        self._init_weights()

    def forward(self, x):
        output, hidden = self.rnn(x)
        return output, hidden

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.GRU, nn.LSTM, nn.RNN)):
                for name, param in module.named_parameters():
                    if "weight_ih" in name:
                        torch.nn.init.xavier_uniform_(param.data)
                    elif "weight_hh" in name:
                        torch.nn.init.orthogonal_(param.data)
                    elif "bias" in name:
                        param.data.fill_(0)
