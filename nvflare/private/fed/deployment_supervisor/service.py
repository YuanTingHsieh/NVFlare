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
"""Server task-service contract and a Controller-facing prototype adapter."""

import copy
import threading
import uuid
from dataclasses import dataclass

from nvflare.apis.shareable import Shareable


@dataclass(frozen=True)
class TaskAssignment:
    job_id: str
    task_id: str
    task_name: str
    client_name: str
    attempt: str
    data: Shareable
    round_number: int = 0


class ControllerTaskServiceAdapter:
    """Translate ordinary Controller broadcast/send intent into task leases.

    This executable adapter models the server deployment task-service boundary.
    It intentionally does not replace WFCommServer in this prototype; production
    integration must have existing Controllers enqueue through this contract.
    """

    def __init__(self):
        self._pending = []
        self._active = {}
        self._results = {}
        self._failures = {}
        self._lock = threading.RLock()

    def broadcast(self, *, job_id, task_name, data, clients, round_number=0):
        return [self._schedule(job_id, task_name, data, client_name, round_number) for client_name in clients]

    def send(self, *, job_id, task_name, data, client, round_number=0):
        return self._schedule(job_id, task_name, data, client, round_number)

    def _schedule(self, job_id, task_name, data, client_name, round_number):
        if not isinstance(data, Shareable):
            raise TypeError("task data must be a Shareable")
        task_id = str(uuid.uuid4())
        assignment = TaskAssignment(
            job_id=job_id,
            task_id=task_id,
            task_name=task_name,
            client_name=client_name,
            attempt=str(uuid.uuid4()),
            data=copy.deepcopy(data),
            round_number=round_number,
        )
        with self._lock:
            self._pending.append(assignment)
        return task_id

    def acquire(self, client_name):
        with self._lock:
            for index, task in enumerate(self._pending):
                if task.client_name == client_name:
                    self._pending.pop(index)
                    self._active[task.task_id] = task
                    return task
        return None

    def publish(self, assignment, result):
        if not isinstance(result, Shareable):
            raise TypeError("task result must be a Shareable")
        with self._lock:
            active = self._active.get(assignment.task_id)
            if not self._same_attempt(active, assignment):
                raise ValueError("stale, duplicate, or unknown task attempt")
            self._active.pop(assignment.task_id)
            self._results[assignment.task_id] = result

    def fail(self, assignment, reason):
        with self._lock:
            active = self._active.get(assignment.task_id)
            if self._same_attempt(active, assignment):
                self._active.pop(assignment.task_id)
                self._failures[assignment.task_id] = str(reason)

    @staticmethod
    def _same_attempt(active, supplied):
        return (
            active is not None
            and active.task_id == supplied.task_id
            and active.attempt == supplied.attempt
            and active.client_name == supplied.client_name
            and active.job_id == supplied.job_id
        )

    def result(self, task_id):
        with self._lock:
            return self._results.get(task_id)

    def failure(self, task_id):
        with self._lock:
            return self._failures.get(task_id)

    def is_idle(self):
        with self._lock:
            return not self._pending and not self._active
