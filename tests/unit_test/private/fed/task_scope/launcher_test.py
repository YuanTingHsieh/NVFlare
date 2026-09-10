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
"""Launcher-independent logical lifecycle and control protocol tests."""

import ast
import json
import shlex
import threading
from pathlib import Path
from unittest.mock import Mock

import pytest

from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.job_launcher_spec import JobProcessArgs, JobReturnCode
from nvflare.apis.shareable import Shareable
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey
from nvflare.fuel.f3.cellnet.defs import ReturnCode as CellReturnCode
from nvflare.private.defs import CellChannel, new_cell_message
from nvflare.private.fed import task_scope
from nvflare.private.fed.task_scope.launcher import TaskScopedJobHandle, TaskScopedJobRegistry, launch_task_scope_worker
from nvflare.private.fed.task_scope.protocol import (
    ATTEMPT_OPTION,
    DIRECTORY_OPTION,
    DONE,
    READY,
    STATUS,
    TASK_COMPLETE,
    TASK_TOKEN,
    TERMINAL_TOPIC,
    WORKER_MODULE_CONTEXT_KEY,
    write_receipt,
)


def test_common_worker_bootstrap_is_attempt_scoped_and_restored_on_launch_failure(tmp_path):
    fl_ctx = FLContext()
    original = {JobProcessArgs.OPTIONS: ("--set", "existing=value")}
    fl_ctx.set_prop(FLContextKey.JOB_PROCESS_ARGS, original, private=True, sticky=False)
    directory = tmp_path / "directory with spaces"

    def fail():
        assert fl_ctx.get_prop(WORKER_MODULE_CONTEXT_KEY) == "nvflare.private.fed.task_scope.worker"
        args = fl_ctx.get_prop(FLContextKey.JOB_PROCESS_ARGS)
        options = dict(token.split("=", 1) for token in shlex.split(args[JobProcessArgs.OPTIONS][1]))
        assert options == {
            "existing": "value",
            ATTEMPT_OPTION: "attempt-1",
            DIRECTORY_OPTION: str(directory),
        }
        raise RuntimeError("physical launch failed")

    with pytest.raises(RuntimeError, match="physical launch failed"):
        launch_task_scope_worker(fail, fl_ctx, "attempt-1", str(directory))
    assert fl_ctx.get_prop(FLContextKey.JOB_PROCESS_ARGS) is original
    assert fl_ctx.get_prop(WORKER_MODULE_CONTEXT_KEY) is None


def test_attempts_settle_and_publish_receipts_before_relaunch(tmp_path):
    entered = threading.Event()
    release = threading.Event()
    allocations = []
    trace = []
    probe = Mock(side_effect=[{STATUS: READY, TASK_TOKEN: "one"}, {STATUS: READY, TASK_TOKEN: "two"}, {STATUS: DONE}])
    handle = TaskScopedJobHandle("job-1", str(tmp_path / "job-1"), probe, Mock())

    class Allocation:
        def __init__(self, attempt, directory):
            self.attempt = attempt
            self.directory = directory
            self.finished = False

        def wait(self):
            if allocations[0] is self:
                entered.set()
                assert release.wait(3), "test did not release the first physical worker"
            write_receipt(self.directory, self.attempt, {STATUS: TASK_COMPLETE, "task_id": self.attempt})
            self.finished = True
            trace.append("settled")

        def poll(self):
            return JobReturnCode.SUCCESS if self.finished else JobReturnCode.UNKNOWN

        def terminate(self):
            release.set()

    def launch(attempt, directory):
        assert all(a.finished for a in allocations), "physical workers must not overlap"
        allocation = Allocation(attempt, directory)
        allocations.append(allocation)
        trace.append("launched")
        return allocation

    handle._launch_attempt = Mock(side_effect=launch)
    thread = threading.Thread(target=handle.wait, daemon=True)
    thread.start()
    try:
        assert entered.wait(3)
        assert handle.poll() == JobReturnCode.UNKNOWN
        assert len(allocations) == 1
        assert probe.call_count == 1
    finally:
        release.set()
        thread.join(3)
    assert not thread.is_alive()
    assert handle.poll() == JobReturnCode.SUCCESS
    assert trace == ["launched", "settled", "launched", "settled"]
    assert allocations[0].attempt != allocations[1].attempt
    events = [json.loads(line) for line in Path(handle._root, "events.jsonl").read_text().splitlines()]
    phases = [e["phase"] for e in events if e["phase"] in ("submitting", "allocation_released", "receipt")]
    assert phases == ["submitting", "allocation_released", "receipt"] * 2


@pytest.mark.parametrize("field", ["poll_interval", "probe_timeout", "communication_timeout"])
@pytest.mark.parametrize("value", [0, True, "1"])
def test_registry_rejects_invalid_timeout(field, value):
    with pytest.raises(ValueError, match="positive numbers"):
        TaskScopedJobRegistry(**{field: value})


@pytest.fixture
def registered(tmp_path):
    registry = TaskScopedJobRegistry()
    engine = Mock()
    engine.new_context.side_effect = FLContext

    def create(probe, interval, timeout):
        return TaskScopedJobHandle("job-1", str(tmp_path / "job-1"), probe, Mock(), interval, timeout)

    handle = registry.register("job-1", engine, create)
    return registry, engine, handle, create


def _terminal(origin="server.job-1", job_id="job-1", status=DONE):
    return new_cell_message({MessageHeaderKey.ORIGIN: origin}, Shareable({"job_id": job_id, STATUS: status}))


def test_registry_retains_live_handle_and_releases_only_after_terminal_settlement(registered):
    registry, engine, handle, create = registered
    engine.get_cell().register_request_cb.assert_called_once_with(
        channel=CellChannel.AUX_COMMUNICATION, topic=TERMINAL_TOPIC, cb=registry._terminal_message
    )
    registry.release("job-1")
    assert registry._handles["job-1"] is handle
    with pytest.raises(RuntimeError, match="already active"):
        registry.register("job-1", engine, create)
    reply = registry._terminal_message(_terminal())
    assert reply.get_header(MessageHeaderKey.RETURN_CODE) == CellReturnCode.OK
    # Receipt of DONE does not itself pretend the logical handle has settled.
    registry.release("job-1")
    assert registry._handles["job-1"] is handle
    handle.wait()
    assert handle.poll() == JobReturnCode.SUCCESS
    engine.aux_runner.send_aux_request.assert_not_called()
    registry.release("job-1")
    assert "job-1" not in registry._handles


@pytest.mark.parametrize(
    "message,expected",
    [
        (_terminal(origin="server.other-job"), CellReturnCode.AUTHENTICATION_ERROR),
        (_terminal(job_id="unknown"), CellReturnCode.AUTHENTICATION_ERROR),
        (_terminal(status=READY), CellReturnCode.INVALID_REQUEST),
    ],
)
def test_registry_rejects_unrelated_or_nonterminal_notices(registered, message, expected):
    registry, _, handle, _ = registered
    reply = registry._terminal_message(message)
    assert reply.get_header(MessageHeaderKey.RETURN_CODE) == expected
    assert handle._server_terminal is None
    assert handle.poll() == JobReturnCode.UNKNOWN


def test_private_task_scope_imports_do_not_depend_on_optional_launcher_packages():
    # Inspect every module without importing optional packages or assuming which
    # launcher dependencies happen to be installed in the test environment.
    for source in Path(task_scope.__file__).parent.rglob("*.py"):
        for node in ast.walk(ast.parse(source.read_text(), filename=str(source))):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                modules = [module] + [f"{module}.{alias.name}".strip(".") for alias in node.names]
            else:
                continue
            for module in modules:
                assert module != "nvflare.app_opt" and not module.startswith("nvflare.app_opt."), (source, module)
                assert "slurm" not in module.lower(), (source, module)
