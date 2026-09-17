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
"""Fresh task worker: Executor or unchanged Client API script, without a Cell."""

import argparse
import importlib
import json
import os
import runpy
import shlex
import sys

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey, FLMetaKey, ReturnCode
from nvflare.apis.fl_context import FLContextManager
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.client.api_spec import CLIENT_API_KEY
from nvflare.client.config import ConfigKey
from nvflare.client.in_process.api import (
    TOPIC_ABORT,
    TOPIC_GLOBAL_RESULT,
    TOPIC_LOCAL_RESULT,
    TOPIC_LOG_DATA,
    TOPIC_STOP,
    InProcessClientAPI,
)
from nvflare.fuel.data_event.data_bus import DataBus
from nvflare.private.fed.deployment_supervisor.artifacts import read_artifact, write_artifact
from nvflare.private.fed.utils.fed_utils import nvflare_fobs_initialize


def _load_factory(path):
    module_name, separator, name = path.partition(":")
    if not separator:
        module_name, separator, name = path.rpartition(".")
    if not module_name or not name:
        raise ValueError("executor factory must be 'module:name' or 'module.name'")
    return getattr(importlib.import_module(module_name), name)


def _load_components(definitions):
    result = []
    for definition in definitions or []:
        if not isinstance(definition, dict):
            raise TypeError("worker component definition must be a dict")
        path = definition.get("path") or definition.get("class_path")
        if not isinstance(path, str) or not path:
            raise ValueError("worker component definition requires a path")
        result.append(_load_factory(path)(**(definition.get("args") or {})))
    return result


class _TaskComponentRegistry:
    """The deliberately small Engine surface available to a fresh task worker."""

    def __init__(self, definitions):
        self._components = {}
        for definition in definitions or []:
            if not isinstance(definition, dict):
                raise TypeError("task component definition must be a dict")
            component_id = definition.get("id")
            if not isinstance(component_id, str) or not component_id:
                raise ValueError("task component definition requires a nonempty id")
            if component_id in self._components:
                raise ValueError(f"duplicate task component id: {component_id}")
            path = definition.get("path") or definition.get("class_path")
            if not isinstance(path, str) or not path:
                raise ValueError(f"task component '{component_id}' requires a path")
            self._components[component_id] = _load_factory(path)(**(definition.get("args") or {}))

    def get_component(self, component_id):
        return self._components.get(component_id)


def _apply_filters(definitions, data, fl_ctx):
    value = data
    for task_filter in _load_components(definitions):
        value = task_filter.process(value, fl_ctx)
        if not isinstance(value, Shareable):
            raise TypeError("managed task filter must return a Shareable")
    return value


def _new_context(spec):
    manager = FLContextManager(
        engine=_TaskComponentRegistry(spec.get("task_components")),
        identity_name=spec["client_name"],
        job_id=spec["job_id"],
    )
    fl_ctx = manager.new_context()
    fl_ctx.set_prop(FLContextKey.TASK_NAME, spec["task_name"], private=True, sticky=False)
    fl_ctx.set_prop(FLContextKey.TASK_ID, spec["task_id"], private=True, sticky=False)
    fl_ctx.set_prop("deployment_task_state_dir", spec.get("state_dir"), private=True, sticky=False)
    return fl_ctx


def _run_executor(spec, data, fl_ctx):
    executor = _load_factory(spec["entrypoint"])(**(spec.get("factory_args") or {}))
    signal = Signal()
    executor.handle_event(EventType.START_RUN, fl_ctx)
    try:
        executor.handle_event(EventType.BEFORE_TASK_EXECUTION, fl_ctx)
        result = executor.execute(spec["task_name"], data, fl_ctx, signal)
        fl_ctx.set_prop(FLContextKey.TASK_RESULT, result, private=True, sticky=False)
        executor.handle_event(EventType.AFTER_TASK_EXECUTION, fl_ctx)
    finally:
        executor.handle_event(EventType.END_RUN, fl_ctx)
    if not isinstance(result, Shareable):
        raise TypeError("Executor result must be a Shareable")
    return result


def _run_script(spec, data):
    """Run an existing flare.receive()/send() script over a process-local adapter."""
    bus = DataBus()
    results = []
    logs = []

    def capture_result(topic, datum, data_bus):
        results.append(datum)
        data_bus.publish([TOPIC_STOP], "single task completed")

    def capture_log(topic, datum, data_bus):
        logs.append(datum)

    bus.subscribe([TOPIC_LOCAL_RESULT], capture_result)
    bus.subscribe([TOPIC_LOG_DATA], capture_log)
    metadata = {
        FLMetaKey.JOB_ID: spec["job_id"],
        FLMetaKey.SITE_NAME: spec["client_name"],
        ConfigKey.TASK_NAME: spec["task_name"],
        ConfigKey.TASK_EXCHANGE: spec.get("task_exchange", {}),
        ConfigKey.MEMORY_GC_ROUNDS: spec.get("memory_gc_rounds", 0),
        ConfigKey.CUDA_EMPTY_CACHE: spec.get("cuda_empty_cache", False),
    }
    api = InProcessClientAPI(task_metadata=metadata, result_check_interval=0.01)
    api.init()
    bus.put_data(CLIENT_API_KEY, api)
    data.set_header(FLMetaKey.JOB_ID, spec["job_id"])
    data.set_header(FLMetaKey.SITE_NAME, spec["client_name"])
    old_argv = sys.argv
    old_state = os.environ.get("NVFLARE_TASK_STATE_DIR")
    try:
        sys.argv = [spec["entrypoint"], *shlex.split(spec.get("arguments", ""))]
        os.environ["NVFLARE_TASK_STATE_DIR"] = spec["state_dir"]
        bus.publish([TOPIC_GLOBAL_RESULT], data)
        runpy.run_path(spec["entrypoint"], run_name="__main__")
    except BaseException:
        bus.publish([TOPIC_ABORT], "task script failed")
        raise
    finally:
        sys.argv = old_argv
        if old_state is None:
            os.environ.pop("NVFLARE_TASK_STATE_DIR", None)
        else:
            os.environ["NVFLARE_TASK_STATE_DIR"] = old_state
        api.close()
        bus.unsubscribe(TOPIC_LOCAL_RESULT, capture_result)
        bus.unsubscribe(TOPIC_LOG_DATA, capture_log)
        bus.put_data(CLIENT_API_KEY, None)
    if len(results) != 1 or not isinstance(results[0], Shareable):
        return make_reply(ReturnCode.EXECUTION_RESULT_ERROR)
    result = results[0]
    if logs:
        result.set_header("deployment_worker_log_count", len(logs))
    return result


def run(attempt_dir):
    nvflare_fobs_initialize()
    with open(os.path.join(attempt_dir, "worker.json")) as stream:
        spec = json.load(stream)
    for path in reversed(spec.get("module_search_paths") or []):
        if not isinstance(path, str) or not os.path.isabs(path) or not os.path.isdir(path):
            raise ValueError(f"invalid worker module search path: {path!r}")
        sys.path.insert(0, path)
    input_task_name, data = read_artifact(
        attempt_dir,
        attempt=spec["attempt"],
        job_id=spec["job_id"],
        task_id=spec["task_id"],
        kind="input",
    )
    if input_task_name != spec["task_name"]:
        raise ValueError("input artifact task name does not match the worker specification")
    fl_ctx = _new_context(spec)
    data = _apply_filters(spec.get("input_filters"), data, fl_ctx)
    if spec["kind"] == "executor":
        result = _run_executor(spec, data, fl_ctx)
    elif spec["kind"] == "script":
        result = _run_script(spec, data)
    else:
        raise ValueError(f"unsupported worker kind {spec['kind']!r}")
    result = _apply_filters(spec.get("result_filters"), result, fl_ctx)
    # Match ClientRunner._process_task: workflow cookies belong to the task
    # envelope and must accompany the result even when application code creates
    # a new Shareable. Controllers and widgets use them for round/workflow
    # correlation (for example IntimeModelSelector's contribution round).
    cookie_jar = data.get_cookie_jar()
    if cookie_jar:
        result.set_cookie_jar(cookie_jar)
    write_artifact(
        attempt_dir,
        attempt=spec["attempt"],
        job_id=spec["job_id"],
        task_id=spec["task_id"],
        task_name=spec["task_name"],
        kind="result",
        data=result,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("attempt_dir")
    args = parser.parse_args()
    run(args.attempt_dir)


if __name__ == "__main__":
    main()
