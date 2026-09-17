# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
"""Test-only real CUDA and deterministic slow-upload payload fixture."""

import hashlib

import torch
import torch.nn as nn
import torch.nn.functional as F

from nvflare.apis.dxo import from_shareable
from nvflare.apis.executor import Executor
from nvflare.apis.filter import Filter


class SimpleNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, value):
        value = self.pool(F.relu(self.conv1(value)))
        value = self.pool(F.relu(self.conv2(value)))
        value = torch.flatten(value, 1)
        value = F.relu(self.fc1(value))
        value = F.relu(self.fc2(value))
        return self.fc3(value)


def _payload(size):
    pattern = b"NVFLARE-G05-SLOW-UPLOAD\x00"
    return (pattern * ((size + len(pattern) - 1) // len(pattern)))[:size]


class SlowUploadExecutor(Executor):
    def __init__(self, payload_mib=64):
        super().__init__()
        self.payload_mib = payload_mib

    def execute(self, task_name, shareable, fl_ctx, abort_signal):
        if not torch.cuda.is_available():
            raise RuntimeError("T5 requires a real CUDA device")
        device = torch.device("cuda:0")
        left = torch.ones((1024, 1024), device=device)
        right = torch.full((1024, 1024), 2.0, device=device)
        marker = float(torch.mm(left, right)[0, 0].item())
        torch.cuda.synchronize(device)

        dxo = from_shareable(shareable)
        payload = _payload(self.payload_mib * 1024 * 1024)
        dxo.meta.update(
            {
                "NUM_STEPS_CURRENT_ROUND": 1,
                "architecture_a_t5_payload": payload,
                "architecture_a_t5_payload_bytes": len(payload),
                "architecture_a_t5_payload_sha256": hashlib.sha256(payload).hexdigest(),
                "architecture_a_t5_cuda_device": str(device),
                "architecture_a_t5_cuda_marker": marker,
            }
        )
        return dxo.to_shareable()


class SlowUploadVerificationFilter(Filter):
    def __init__(self, expected_mib=64):
        super().__init__()
        self.expected_bytes = expected_mib * 1024 * 1024

    def process(self, shareable, fl_ctx):
        dxo = from_shareable(shareable)
        payload = dxo.meta.pop("architecture_a_t5_payload", None)
        declared_bytes = dxo.meta.pop("architecture_a_t5_payload_bytes", None)
        declared_digest = dxo.meta.pop("architecture_a_t5_payload_sha256", None)
        device = dxo.meta.pop("architecture_a_t5_cuda_device", None)
        marker = dxo.meta.pop("architecture_a_t5_cuda_marker", None)
        if not isinstance(payload, bytes):
            raise ValueError("missing Architecture A T5 byte payload")
        digest = hashlib.sha256(payload).hexdigest()
        if len(payload) != self.expected_bytes or declared_bytes != len(payload) or declared_digest != digest:
            raise ValueError("Architecture A T5 byte payload failed size/digest verification")
        if device != "cuda:0" or marker != 2048.0:
            raise ValueError("Architecture A T5 CUDA evidence is invalid")
        shareable.set_header("architecture_a_t5_payload_verified", True)
        shareable.set_header("architecture_a_t5_payload_bytes", len(payload))
        shareable.set_header("architecture_a_t5_payload_sha256", digest)
        dxo.update_shareable(shareable)
        self.log_info(fl_ctx, f"verified Architecture A T5 payload: {len(payload)} bytes, sha256={digest}")
        return shareable
