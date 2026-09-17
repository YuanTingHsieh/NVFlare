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
"""Small application components for the deployment-supervisor prototype."""

import json
import os

from nvflare.apis.executor import Executor
from nvflare.apis.filter import Filter
from nvflare.apis.shareable import Shareable


class IncrementExecutor(Executor):
    def execute(self, task_name, shareable, fl_ctx, abort_signal):
        result = Shareable()
        result["value"] = shareable["value"] + 1
        result["site"] = fl_ctx.get_identity_name()
        return result


class DeclaredStateExecutor(Executor):
    """Example of explicit application state rather than implicit object fields."""

    def execute(self, task_name, shareable, fl_ctx, abort_signal):
        state_dir = fl_ctx.get_prop("deployment_task_state_dir")
        state_file = os.path.join(state_dir, "counter.json")
        try:
            with open(state_file) as stream:
                count = json.load(stream)["count"]
        except FileNotFoundError:
            count = 0
        count += 1
        temporary = f"{state_file}.tmp"
        with open(temporary, "w") as stream:
            json.dump({"count": count}, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, state_file)
        result = Shareable()
        result["count"] = count
        return result


class CredentialProbeExecutor(Executor):
    def execute(self, task_name, shareable, fl_ctx, abort_signal):
        result = Shareable()
        result["credentials"] = sorted(name for name in os.environ if name.startswith("NVFLARE_JOB_"))
        return result


class AddFilter(Filter):
    def __init__(self, amount):
        super().__init__()
        self.amount = amount

    def process(self, shareable, fl_ctx):
        shareable["value"] += self.amount
        return shareable


class OffsetComponent:
    def __init__(self, amount):
        self.amount = amount


class ComponentStateExecutor(Executor):
    """Exercises one explicit task-local component plus durable domain state."""

    def execute(self, task_name, shareable, fl_ctx, abort_signal):
        component = fl_ctx.get_engine().get_component("offset")
        if component is None:
            raise RuntimeError("missing task-local offset component")
        state_file = os.path.join(fl_ctx.get_prop("deployment_task_state_dir"), "component-count.json")
        try:
            with open(state_file) as stream:
                count = json.load(stream)["count"]
        except FileNotFoundError:
            count = 0
        count += 1
        temporary = f"{state_file}.tmp"
        with open(temporary, "w") as stream:
            json.dump({"count": count}, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, state_file)
        return Shareable({"value": shareable["value"] + component.amount, "count": count})
