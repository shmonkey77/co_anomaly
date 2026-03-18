"""
Feedforward neural network for change order anomaly detection.

Architecture: 3 fully connected layers with BatchNorm, ReLU, and Dropout.
Output: single sigmoid unit → anomaly probability (0–1).
"""

import torch
import torch.nn as nn
from model.dataset import INPUT_DIM


class AnomalyDetector(nn.Module):
    """
    3-layer feedforward network.

    Layer sizes: INPUT_DIM → 64 → 32 → 16 → 1

    Design choices:
    - BatchNorm before activation stabilizes training on mixed-scale features
    - Dropout(0.3) reduces overfitting on the relatively small dataset
    - Sigmoid output gives a probability score suitable for thresholding
    """

    def __init__(self, input_dim: int = INPUT_DIM, dropout: float = 0.3):
        super().__init__()

        self.net = nn.Sequential(
            # Layer 1
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Layer 2
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Layer 3
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),

            # Output
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = AnomalyDetector()
    print(model)
    print(f"\nTrainable parameters: {count_parameters(model):,}")

    # Sanity check — forward pass with a dummy batch
    dummy = torch.randn(8, INPUT_DIM)
    out = model(dummy)
    print(f"Output shape: {out.shape}")   # should be (8,)
    print(f"Output range: [{out.min():.3f}, {out.max():.3f}]")  # should be 0–1
