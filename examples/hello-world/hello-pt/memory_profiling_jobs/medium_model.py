# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
~500MB model for memory profiling with multiple clients.
1/5 size of the 2.43GB model for faster profiling with 3 clients.
"""

import torch.nn as nn


class GigabyteModel(nn.Module):
    """~500MB model for memory profiling with multiple clients.

    This creates a model with approximately 125 million parameters (~500MB).
    1/5 the size of customer's model for faster profiling with 3 clients.
    """

    def __init__(self):
        super(GigabyteModel, self).__init__()

        # Convolutional backbone
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

        # Fully connected layers to reach ~500MB
        # 500MB = ~125M float32 parameters
        # Using 5000x25000 = 125M params
        self.fc_layers = nn.Sequential(
            nn.Linear(512, 5000),  # 2.56M params
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(5000, 25000),  # 125M params (most of the model!)
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(25000, 10),  # 250K params
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


def get_model_size_mb(model):
    """Calculate model size in MB."""
    total_params = 0
    total_size = 0
    for param in model.parameters():
        param_count = param.nelement()
        param_size = param.nelement() * param.element_size()
        total_params += param_count
        total_size += param_size

    size_mb = total_size / 1024 / 1024
    return total_params, size_mb


if __name__ == "__main__":
    model = GigabyteModel()
    params, size_mb = get_model_size_mb(model)
    print(f"Model: {params:,} parameters, {size_mb:.2f} MB")
    print("\n1/5 size of customer's 2.43GB model for faster profiling")
    print("\nExpected memory with 3 clients:")
    print(f"  - Without in-place aggregation: ~{(1 + 3 * 2) * size_mb:.0f} MB")
    print(f"    (1 global + 3*(1 client + 1 temporary during aggregation))")
    print(f"  - With in-place aggregation: ~{(1 + 3) * size_mb:.0f} MB")
    print(f"    (1 global + 3 clients, no temporaries)")
    print(f"  - Expected savings: ~{3 * size_mb:.0f} MB")
