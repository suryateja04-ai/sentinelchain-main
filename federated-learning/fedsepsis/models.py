"""PyTorch models that are trained with federated averaging.

The sizes are this module's defaults; set them to match the Phase-I models if they differ.
"""
from __future__ import annotations

import torch
from torch import nn

from .config import FLAT, N_FEATURES


class LogisticRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(FLAT, 1)

    def forward(self, x):                      # x: (batch, 16, 39)
        return self.linear(x.flatten(1)).squeeze(1)


class Recurrent(nn.Module):
    def __init__(self, cell: str, hidden: int = 64, layers: int = 1, dropout: float = 0.0):
        super().__init__()
        rnn = {"lstm": nn.LSTM, "gru": nn.GRU}[cell]
        self.rnn = rnn(N_FEATURES, hidden, layers, batch_first=True,
                       dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out[:, -1]).squeeze(1)


class CausalBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel, dilation):
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv = nn.Conv1d(c_in, c_out, kernel, dilation=dilation)
        self.skip = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()
        self.act = nn.ReLU()

    def forward(self, x):                      # x: (batch, channels, time)
        y = self.conv(nn.functional.pad(x, (self.pad, 0)))   # pad on the left only: causal
        return self.act(y + self.skip(x))


class TemporalCNN(nn.Module):
    def __init__(self, channels: int = 64, kernel: int = 3, dilations=(1, 2, 4)):
        super().__init__()
        blocks, c = [], N_FEATURES
        for d in dilations:
            blocks.append(CausalBlock(c, channels, kernel, d))
            c = channels
        self.net = nn.Sequential(*blocks)
        self.head = nn.Linear(channels, 1)

    def forward(self, x):
        h = self.net(x.transpose(1, 2))
        return self.head(h[:, :, -1]).squeeze(1)


NEURAL = {
    "logistic_regression": LogisticRegression,
    "lstm": lambda: Recurrent("lstm"),
    "gru": lambda: Recurrent("gru"),
    "temporal_cnn": TemporalCNN,
}


def build(name: str) -> nn.Module:
    torch.manual_seed(42)                      # every hospital starts from the same weights
    return NEURAL[name]()
