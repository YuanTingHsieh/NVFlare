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
"""Fresh application task worker with no federation Cell or publication authority."""

import argparse
import importlib
import json
import os
import resource
import runpy
import shlex
import sys
import time

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey, FLMetaKey, ReturnCode
from nvflare.apis.fl_context import FLContext, FLContextManager
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.apis.workspace import Workspace
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
from nvflare.private.event import fire_event
from nvflare.private.fed.job_task_worker.artifacts import read_artifact, write_artifact
from nvflare.private.fed.job_task_worker.protocol import STATE_DIR_CTX_PROP, STATE_DIR_ENV
from nvflare.private.fed.utils.fed_utils import nvflare_fobs_initialize


def _load_factory(path):
    if not isinstance(path, str):
        raise TypeError("application factory path must be a string")
    module_name, separator, name = path.partition(":")
    if not separator:
        module_name, separator, name = path.rpartition(".")
    if not module_name or not name:
        raise ValueError("application factory must be 'module:name' or 'module.name'")
    return getattr(importlib.import_module(module_name), name)


def _load_components(definitions):
    components = {}
    for definition in definitions or []:
        if not isinstance(definition, dict):
            raise TypeError("task-worker component definition must be a dict")
        component_id = definition.get("id")
        path = definition.get("factory_path")
        if not isinstance(component_id, str) or not component_id or component_id in components:
            raise ValueError("task-worker component IDs must be nonempty and unique")
        components[component_id] = _load_factory(path)(**(definition.get("args") or {}))
    return components


class TaskWorkerEngine:
    """Deliberately small task-local Engine surface for application components."""

    def __init__(self, workspace, components):
        self._workspace = workspace
        self._components = components
        self._handlers = []
        self._context_manager = None

    def get_component(self, component_id):
        return self._components.get(component_id)

    def get_all_components(self):
        return self._components

    def get_workspace(self):
        return self._workspace

    def new_context(self):
        if self._context_manager is None:
            raise RuntimeError("task-worker context manager is not initialized")
        return self._context_manager.new_context()

    def fire_event(self, event_type, fl_ctx):
        fire_event(event=event_type, handlers=self._handlers, ctx=fl_ctx)

    def get_cell(self):
        return None

    def register_aux_message_handler(self, *args, **kwargs):
        raise RuntimeError("fresh application task workers do not expose federation auxiliary messaging")

    def send_aux_request(self, *args, **kwargs):
        raise RuntimeError("fresh application task workers do not expose federation auxiliary messaging")


def _new_context(spec, components):
    workspace = Workspace(spec["workspace_root"], site_name=spec["client_name"])
    engine = TaskWorkerEngine(workspace, components)
    manager = FLContextManager(engine=engine, identity_name=spec["client_name"], job_id=spec["job_id"])
    engine._context_manager = manager
    fl_ctx = manager.new_context()
    fl_ctx.set_prop(FLContextKey.TASK_NAME, spec["task_name"], private=True, sticky=False)
    fl_ctx.set_prop(FLContextKey.TASK_ID, spec["task_id"], private=True, sticky=False)
    fl_ctx.set_prop(FLContextKey.WORKSPACE_OBJECT, workspace, private=True, sticky=True)
    fl_ctx.set_prop(FLContextKey.APP_ROOT, spec["app_root"], private=True, sticky=True)
    fl_ctx.set_prop(FLContextKey.PROCESS_TYPE, "client_task_worker", private=True, sticky=True)
    fl_ctx.set_prop(STATE_DIR_CTX_PROP, spec["state_dir"], private=True, sticky=True)
    return engine, fl_ctx


def _restore_peer_context(data, fl_ctx):
    peer_props = data.get_peer_props()
    if isinstance(peer_props, dict):
        peer_ctx = FLContext()
        peer_ctx.set_public_props(peer_props)
        fl_ctx.set_peer_context(peer_ctx)


def _run_executor(spec, data, engine, fl_ctx):
    executor = _load_factory(spec["application_path"])(**(spec.get("application_args") or {}))
    engine._handlers = [*engine.get_all_components().values(), executor]
    abort_signal = Signal()
    engine.fire_event(EventType.START_RUN, fl_ctx)
    try:
        fl_ctx.set_prop(FLContextKey.TASK_DATA, data, private=True, sticky=False)
        engine.fire_event(EventType.BEFORE_TASK_EXECUTION, fl_ctx)
        result = executor.execute(spec["task_name"], data, fl_ctx, abort_signal)
        if not isinstance(result, Shareable):
            raise TypeError("application Executor result must be a Shareable")
        fl_ctx.set_prop(FLContextKey.TASK_RESULT, result, private=True, sticky=False)
        engine.fire_event(EventType.AFTER_TASK_EXECUTION, fl_ctx)
        return result
    finally:
        engine.fire_event(EventType.END_RUN, fl_ctx)


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
    old_sys_path = sys.path.copy()
    old_state = os.environ.get(STATE_DIR_ENV)
    try:
        sys.argv = [spec["application_path"], *shlex.split(spec.get("arguments", ""))]
        # runpy does not give a file-backed script the normal ``sys.path[0]``
        # that ``python path/to/script.py`` provides.  Bootstrap the deployed
        # script directory explicitly so job-local sibling imports work without
        # depending on a launcher or ambient PYTHONPATH.
        sys.path.insert(0, os.path.dirname(os.path.abspath(spec["application_path"])))
        os.environ[STATE_DIR_ENV] = spec["state_dir"]
        bus.publish([TOPIC_GLOBAL_RESULT], data)
        runpy.run_path(spec["application_path"], run_name="__main__")
    except BaseException:
        bus.publish([TOPIC_ABORT], "task script failed")
        raise
    finally:
        sys.argv = old_argv
        sys.path[:] = old_sys_path
        if old_state is None:
            os.environ.pop(STATE_DIR_ENV, None)
        else:
            os.environ[STATE_DIR_ENV] = old_state
        api.close()
        bus.unsubscribe(TOPIC_LOCAL_RESULT, capture_result)
        bus.unsubscribe(TOPIC_LOG_DATA, capture_log)
        bus.put_data(CLIENT_API_KEY, None)
    if len(results) != 1 or not isinstance(results[0], Shareable):
        return make_reply(ReturnCode.EXECUTION_RESULT_ERROR)
    result = results[0]
    if logs:
        result.set_header("job_task_worker_log_count", len(logs))
    return result


def _write_receipt(attempt_dir, receipt):
    path = os.path.join(attempt_dir, "worker_receipt.json")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as stream:
        json.dump(receipt, stream, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    directory_fd = os.open(attempt_dir, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def run(attempt_dir):
    nvflare_fobs_initialize()
    started = time.time()
    cpu_started = os.times()
    with open(os.path.join(attempt_dir, "worker.json")) as stream:
        spec = json.load(stream)
    input_task_name, data = read_artifact(
        attempt_dir,
        attempt=spec["attempt"],
        job_id=spec["job_id"],
        session_id=spec["handoff_scope_id"],
        task_id=spec["task_id"],
        kind="input",
    )
    if input_task_name != spec["task_name"]:
        raise ValueError("input artifact task name does not match the worker specification")
    old_sys_path = sys.path.copy()
    old_state = os.environ.get(STATE_DIR_ENV)
    try:
        # The scheduler may provide its own PYTHONPATH for the NVFlare source
        # overlay.  Make the deployed application modules importable inside the
        # fresh worker independently of that launcher environment.
        sys.path.insert(0, os.path.join(spec["app_root"], "custom"))
        components = _load_components(spec.get("components"))
        engine, fl_ctx = _new_context(spec, components)
        _restore_peer_context(data, fl_ctx)
        os.environ[STATE_DIR_ENV] = spec["state_dir"]
        if spec["kind"] == "executor":
            result = _run_executor(spec, data, engine, fl_ctx)
        elif spec["kind"] == "script":
            result = _run_script(spec, data)
        else:
            raise ValueError(f"unsupported application worker kind {spec['kind']!r}")
    finally:
        sys.path[:] = old_sys_path
        if old_state is None:
            os.environ.pop(STATE_DIR_ENV, None)
        else:
            os.environ[STATE_DIR_ENV] = old_state
    write_artifact(
        attempt_dir,
        attempt=spec["attempt"],
        job_id=spec["job_id"],
        session_id=spec["handoff_scope_id"],
        task_id=spec["task_id"],
        task_name=spec["task_name"],
        kind="result",
        data=result,
    )
    usage = resource.getrusage(resource.RUSAGE_SELF)
    cpu_finished = os.times()
    _write_receipt(
        attempt_dir,
        {
            "attempt": spec["attempt"],
            "job_id": spec["job_id"],
            "handoff_scope_id": spec["handoff_scope_id"],
            "task_id": spec["task_id"],
            "pid": os.getpid(),
            "started": started,
            "finished": time.time(),
            "user_cpu_seconds": cpu_finished.user - cpu_started.user,
            "system_cpu_seconds": cpu_finished.system - cpu_started.system,
            "max_rss_native_units": usage.ru_maxrss,
            "max_rss_unit": "bytes" if sys.platform == "darwin" else "kibibytes",
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("attempt_dir")
    args = parser.parse_args()
    run(args.attempt_dir)


if __name__ == "__main__":
    main()
