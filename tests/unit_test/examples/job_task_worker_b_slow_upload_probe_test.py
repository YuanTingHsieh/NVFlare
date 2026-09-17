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

import importlib.util
from pathlib import Path

import pytest


def _load_probe():
    path = Path(__file__).parents[3] / "examples" / "advanced" / "job-task-worker-b" / "slow_upload_probe.py"
    spec = importlib.util.spec_from_file_location("slow_upload_probe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_execution_identity_uses_worker_pid_from_durable_result():
    probe = _load_probe()
    records = [
        {"event": "worker_launched", "attempt": "attempt-1", "scheduler_id": "676"},
        {"event": "allocation_settled", "attempt": "attempt-1", "settled": True},
        {"event": "durable_result_loaded", "attempt": "attempt-1", "worker_pid": 1234},
    ]

    assert probe._execution_identity(records, "attempt-1") == (1234, "676")


@pytest.mark.parametrize(
    "records, message",
    [
        (
            [
                {"event": "worker_launched", "attempt": "attempt-1", "scheduler_id": "676"},
                {"event": "durable_result_loaded", "attempt": "attempt-1"},
            ],
            "worker PID",
        ),
        (
            [
                {"event": "worker_launched", "attempt": "attempt-1"},
                {"event": "durable_result_loaded", "attempt": "attempt-1", "worker_pid": 1234},
            ],
            "Slurm allocation",
        ),
    ],
)
def test_execution_identity_rejects_missing_runtime_identity(records, message):
    probe = _load_probe()

    with pytest.raises(RuntimeError, match=message):
        probe._execution_identity(records, "attempt-1")


def test_audit_requires_the_gpu_process_list_to_be_empty(monkeypatch):
    probe = _load_probe()

    def process_is_gone(pid, signal):
        raise ProcessLookupError

    outputs = iter(["", "9876"])
    monkeypatch.setattr(probe.os, "kill", process_is_gone)
    monkeypatch.setattr(probe, "_run", lambda command: next(outputs))

    with pytest.raises(RuntimeError, match="gpu_pids=\\{9876\\}"):
        probe._audit(1234, "676", "squeue", "nvidia-smi")
