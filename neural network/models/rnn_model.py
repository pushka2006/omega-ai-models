"""OMEGA NEURAL NETWORK ENGINE: Sequence Recurrent Network Architecture."""

import torch
import torch.nn as nn


class SequenceClassifier(nn.Module):
    """Sequence Classifier powered by configurable LSTM or GRU recurrent backbone."""

    def __init__(
        self,
        input_size: int = 10,
        hidden_size: int = 64,
        num_layers: int = 2,
        num_classes: int = 2,
        rnn_type: str = "lstm",  # 'lstm' or 'gru'
        dropout: float = 0.2,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.rnn_type = rnn_type.lower()
        num_directions = 2 if bidirectional else 1

        if self.rnn_type == "lstm":
            self.rnn = nn.LSTM(
                input_size,
                hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
                bidirectional=bidirectional,
            )
        else:
            self.rnn = nn.GRU(
                input_size,
                hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
                bidirectional=bidirectional,
            )

        self.fc = nn.Linear(hidden_size * num_directions, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        # Take the final sequence step representation
        last_step = out[:, -1, :]
        return self.fc(last_step)
