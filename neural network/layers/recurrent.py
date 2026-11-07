"""OMEGA NEURAL NETWORK ENGINE: Recurrent Layers (RNN, LSTM, GRU)."""

from typing import Optional, Tuple
import torch
import torch.nn as nn


class RNN(nn.Module):
    """Elman Recurrent Neural Network layer."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1, batch_first: bool = True, dropout: float = 0.0):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers=num_layers, batch_first=batch_first, dropout=dropout if num_layers > 1 else 0.0)

    def forward(self, x: torch.Tensor, h_0: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.rnn(x, h_0)


class LSTM(nn.Module):
    """Long Short-Term Memory layer."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1, batch_first: bool = True, dropout: float = 0.0, bidirectional: bool = False):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=batch_first,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

    def forward(self, x: torch.Tensor, state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        return self.lstm(x, state)


class GRU(nn.Module):
    """Gated Recurrent Unit layer."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1, batch_first: bool = True, dropout: float = 0.0):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=batch_first, dropout=dropout if num_layers > 1 else 0.0)

    def forward(self, x: torch.Tensor, h_0: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.gru(x, h_0)
