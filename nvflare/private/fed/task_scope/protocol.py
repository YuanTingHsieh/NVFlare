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
"""Attempt-scoped handoff, separate from the legacy job process RC file."""

import json
import os

PROBE_TOPIC = "experimental_task_scope"
TERMINAL_TOPIC = "experimental_task_scope_terminal"
STATUS = "status"
REASON = "reason"
TASK_NAME = "task_name"
TASK_TOKEN = "task_token"
READY = "READY"
WAIT = "WAIT"
DONE = "DONE"
ERROR = "ERROR"
TASK_COMPLETE = "TASK_COMPLETE"
IDLE = "IDLE"
END_RUN = "END_RUN"
INPUT_READY = "INPUT_READY"
RESULT_READY = "RESULT_READY"
PULL = "pull"
COMPUTE = "compute"
PUSH = "push"
PHASES = (PULL, COMPUTE, PUSH)
PHASE_OPTION = "__task_scope_phase"
ATTEMPT_OPTION = "__task_scope_attempt"
DIRECTORY_OPTION = "__task_scope_dir"
WORKER_MODULE_CONTEXT_KEY = "__task_scope_worker_module"
RECEIPT_FILE = "receipt.json"


def _validate_receipt(receipt, attempt):
    if not isinstance(receipt, dict) or receipt.get("attempt") != attempt:
        raise ValueError("missing or stale task-scope receipt")
    if receipt.get(STATUS) not in (TASK_COMPLETE, INPUT_READY, RESULT_READY, IDLE, END_RUN):
        raise ValueError("invalid task-scope receipt status")
    if receipt[STATUS] in (TASK_COMPLETE, INPUT_READY, RESULT_READY):
        if not isinstance(receipt.get("task_id"), str) or not receipt["task_id"]:
            raise ValueError("completed task receipt requires a task ID")
    if receipt[STATUS] in (INPUT_READY, RESULT_READY) and "phase" not in receipt:
        raise ValueError("local handoff receipt requires a phase")
    if "phase" in receipt:
        allowed = {PULL: (INPUT_READY, IDLE, END_RUN), COMPUTE: (RESULT_READY,), PUSH: (TASK_COMPLETE,)}
        if receipt.get("phase") not in allowed or receipt[STATUS] not in allowed[receipt["phase"]]:
            raise ValueError("receipt status does not match its phase")
    return receipt


def write_receipt(directory, attempt, outcome):
    """Write once, after worker shutdown. The parent reads only after allocation exit.

    INPUT_READY/RESULT_READY attest a local phase handoff. Only TASK_COMPLETE
    records the existing task-submit ACK, not a new durable server commit.
    A partial write is never sufficient for advancing the phase or task.
    """
    receipt = _validate_receipt(dict(outcome, attempt=attempt), attempt)
    path = os.path.join(directory, RECEIPT_FILE)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as stream:
        json.dump(receipt, stream)
        stream.flush()
        os.fsync(stream.fileno())


def read_receipt(directory, attempt):
    path = os.path.join(directory, RECEIPT_FILE)
    if os.path.islink(path):
        raise ValueError("task-scope receipt must not be a symlink")
    with open(path) as stream:
        content = stream.read(65537)
    if len(content) > 65536:
        raise ValueError("task-scope receipt too large")
    return _validate_receipt(json.loads(content), attempt)
