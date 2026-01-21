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
1GB model for memory profiling.
"""

import torch.nn as nn


class GigabyteModel(nn.Module):
    """~1GB model for memory profiling.

    This creates a model with approximately 262 million parameters (1GB).
    """

    def __init__(self):
        super(GigabyteModel, self).__init__()

        # Large convolutional backbone
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

        # Very large fully connected layers to reach 1GB
        # 1GB = 268,435,456 bytes = 67,108,864 float32 parameters
        # We need ~262 million parameters total
        self.fc_layers = nn.Sequential(
            nn.Linear(1024, 16384),  # 16.7M params
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16384, 16384),  # 268M params (most of the model!)
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16384, 10),  # 164K params
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
    print("\nExpected memory with 3 clients:")
    print(f"  - Standard FedAvg: ~{(3 + 2) * size_mb:.0f} MB (3 clients + aggregation buffer + result)")
    print(f"  - Memory-efficient FedAvg: ~{(1 + 1) * size_mb:.0f} MB (1 global + processing)")
    print(f"  - Expected savings: ~{3 * size_mb:.0f} MB")
