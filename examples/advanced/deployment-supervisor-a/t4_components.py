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
"""Bounded real-Slurm workspace continuity fixture for Architecture A."""

import json
import os
import time

from nvflare.apis.controller_spec import Task
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.impl.controller import Controller
from nvflare.apis.shareable import Shareable, make_reply


def _save_counter_checkpoint(path, state):
    temporary = path + ".tmp"
    with open(temporary, "w") as stream:
        json.dump(state, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


class WorkspaceCounterExecutor(Executor):
    def __init__(self, compute_seconds=2.0):
        super().__init__()
        self.compute_seconds = compute_seconds

    def execute(self, task_name, shareable, fl_ctx, abort_signal):
        round_number = shareable["round"]
        state_dir = fl_ctx.get_prop("deployment_task_state_dir")
        path = os.path.join(state_dir, "task_scope_counter.json")
        previous = {"round": -1, "value": 0}
        if os.path.exists(path):
            with open(path) as stream:
                previous = json.load(stream)
        if previous["round"] == round_number:
            return Shareable(previous)
        if previous["round"] != round_number - 1:
            raise RuntimeError(f"state continuity lost: {previous['round']} -> {round_number}")
        end = time.monotonic() + self.compute_seconds
        while time.monotonic() < end:
            if abort_signal.triggered:
                return make_reply(ReturnCode.TASK_ABORTED)
            time.sleep(0.05)
        result = {
            "round": round_number,
            "value": previous["value"] + 1,
            "pid": os.getpid(),
            "slurm_id": os.environ.get("SLURM_JOB_ID"),
            "finished_at": time.time(),
        }
        _save_counter_checkpoint(path, result)
        return Shareable(result)


class GapController(Controller):
    def __init__(self, rounds=3, gap_seconds=15.0, task_timeout=600):
        super().__init__()
        self.rounds = rounds
        self.gap_seconds = gap_seconds
        self.task_timeout = task_timeout
        self.results = {}

    def start_controller(self, fl_ctx):
        pass

    def stop_controller(self, fl_ctx):
        pass

    def process_result_of_unknown_task(self, client, task_name, task_id, result, fl_ctx):
        raise RuntimeError(f"unexpected result for {task_id}")

    def _receive(self, client_task, fl_ctx):
        result = client_task.result
        round_number = client_task.task.data["round"]
        if (
            result.get_return_code() != ReturnCode.OK
            or result.get("round") != round_number
            or result.get("value") != round_number + 1
            or not isinstance(result.get("pid"), int)
            or not result.get("slurm_id")
        ):
            raise RuntimeError("counter task failed or lost deployed checkpoint continuity")
        self.results.setdefault(str(round_number), {})[client_task.client.name] = {
            key: result[key] for key in ("round", "value", "pid", "slurm_id", "finished_at")
        }

    def control_flow(self, abort_signal, fl_ctx):
        clients = [client.name for client in fl_ctx.get_engine().get_clients()]
        for round_number in range(self.rounds):
            if abort_signal.triggered:
                return
            task = Task(
                name="count",
                data=Shareable({"round": round_number}),
                timeout=self.task_timeout,
                result_received_cb=self._receive,
            )
            self.broadcast_and_wait(
                task, fl_ctx, targets=clients, min_responses=len(clients), abort_signal=abort_signal
            )
            if len(self.results.get(str(round_number), {})) != len(clients):
                raise RuntimeError("did not receive all counter results before the task deadline")
            end = time.monotonic() + self.gap_seconds
            while time.monotonic() < end:
                if abort_signal.triggered:
                    return
                time.sleep(0.1)
