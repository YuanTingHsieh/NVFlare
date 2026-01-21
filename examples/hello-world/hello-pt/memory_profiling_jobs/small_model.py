"""Small model for quick profiling tests (~50MB)."""

import torch
import torch.nn as nn


class GigabyteModel(nn.Module):
    """Small test model (~50MB, ~13M parameters).

    Quick to run - full 3-job profiling takes ~1-2 minutes instead of 20.
    """

    def __init__(self):
        super().__init__()
        # Much smaller layers for fast testing
        self.fc1 = nn.Linear(1000, 2000)  # ~2M params
        self.fc2 = nn.Linear(2000, 3000)  # ~6M params
        self.fc3 = nn.Linear(3000, 1000)  # ~3M params
        self.fc4 = nn.Linear(1000, 500)  # ~0.5M params
        self.fc5 = nn.Linear(500, 100)  # ~0.05M params
        # Total: ~11.55M params * 4 bytes = ~46MB

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        return x


if __name__ == "__main__":
    # Test model size
    model = GigabyteModel()
    total_params = sum(p.numel() for p in model.parameters())
    size_mb = total_params * 4 / (1024**2)  # 4 bytes per float32
    print(f"SmallModel: {total_params:,} parameters ({size_mb:.1f} MB)")
