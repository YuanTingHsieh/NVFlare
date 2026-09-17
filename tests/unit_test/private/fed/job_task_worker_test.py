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

import json
import os
import subprocess
import sys
import threading
import time

import pytest

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey, JobConstants, ReturnCode
from nvflare.apis.fl_context import FLContextManager
from nvflare.apis.job_launcher_spec import JobProcessEnv
from nvflare.apis.shareable import ReservedHeaderKey, Shareable
from nvflare.apis.signal import Signal
from nvflare.apis.workspace import Workspace
from nvflare.private.fed.client.client_engine_executor_spec import TaskAssignment
from nvflare.private.fed.client.client_runner import ClientRunner, ClientRunnerConfig, TaskRouter
from nvflare.private.fed.job_task_worker.artifacts import read_artifact, write_artifact
from nvflare.private.fed.job_task_worker.executor import (
    ClientAPIJobTaskWorkerExecutor,
    JobTaskWorkerExecutor,
    LocalProcessHandle,
)
from nvflare.private.fed.job_task_worker.protocol import PUBLICATION_ACK_PROP
from nvflare.private.fed.job_task_worker.slurm import SlurmTaskWorkerLauncher
from nvflare.private.fed.job_task_worker.worker import _run_script
from nvflare.private.fed.job_task_worker.worker import run as run_task_worker
from nvflare.private.fed.utils.fed_utils import nvflare_fobs_initialize

_APP_MODULE = """
import json
import os
import time

from nvflare.apis.executor import Executor
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.shareable import Shareable


class ValueComponent(FLComponent):
    def __init__(self, value):
        super().__init__()
        self.value = value


class ProbeExecutor(Executor):
    def execute(self, task_name, shareable, fl_ctx, abort_signal):
        state_dir = fl_ctx.get_prop("job_task_worker_state_dir")
        state_file = os.path.join(state_dir, "counter.json")
        try:
            with open(state_file) as stream:
                count = json.load(stream)["count"]
        except FileNotFoundError:
            count = 0
        count += 1
        temporary = state_file + ".tmp"
        with open(temporary, "w") as stream:
            json.dump({"count": count}, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, state_file)
        result = Shareable()
        result["value"] = shareable["value"] + 1
        result["count"] = count
        result["pid"] = os.getpid()
        result["component"] = fl_ctx.get_engine().get_component("value").value
        result["credentials"] = sorted(name for name in os.environ if name.startswith("NVFLARE_JOB_"))
        result["python_hash_seed"] = os.environ.get("PYTHONHASHSEED")
        result["has_cell"] = fl_ctx.get_engine().get_cell() is not None
        result["has_federation_session"] = fl_ctx.get_prop("__ssid__") is not None
        return result


class CrashExecutor(Executor):
    def execute(self, task_name, shareable, fl_ctx, abort_signal):
        os._exit(7)


class RaiseExecutor(Executor):
    def execute(self, task_name, shareable, fl_ctx, abort_signal):
        raise ValueError("deliberate application failure")


class SlowExecutor(Executor):
    def execute(self, task_name, shareable, fl_ctx, abort_signal):
        time.sleep(30)
        return Shareable()
"""


class _Engine:
    def __init__(self, workspace, components=None):
        self.workspace = workspace
        self.components = components or {}
        self.assignment = None

    def get_workspace(self):
        return self.workspace

    def get_component(self, component_id):
        return self.components.get(component_id)

    def register_aux_message_handler(self, **kwargs):
        pass

    def get_task_assignment(self, fl_ctx, timeout=None):
        return self.assignment


def _workspace(tmp_path, job_id="job-1", site_name="site-1"):
    root = tmp_path / "workspace"
    (root / "startup").mkdir(parents=True)
    (root / "local").mkdir()
    custom = root / job_id / f"app_{site_name}" / "custom"
    custom.mkdir(parents=True)
    (custom / "worker_app.py").write_text(_APP_MODULE)
    return Workspace(str(root), site_name=site_name)


def _context(workspace, engine, job_id="job-1", task_id="task-1", session_id="session-1"):
    manager = FLContextManager(engine=engine, identity_name="site-1", job_id=job_id)
    fl_ctx = manager.new_context()
    fl_ctx.set_prop(FLContextKey.TASK_ID, task_id, private=True, sticky=False)
    fl_ctx.set_prop(FLContextKey.TASK_NAME, "train", private=True, sticky=False)
    fl_ctx.set_prop(FLContextKey.SSID, session_id, private=True, sticky=False)
    fl_ctx.set_prop(FLContextKey.WORKSPACE_OBJECT, workspace, private=True, sticky=True)
    fl_ctx.set_prop(FLContextKey.JOB_META, {JobConstants.JOB_ID: job_id}, private=True, sticky=True)
    return fl_ctx


def _executor(application_path="worker_app:ProbeExecutor", **kwargs):
    worker_timeout = kwargs.pop("worker_timeout", 5)
    poll_interval = kwargs.pop("poll_interval", 0.01)
    return JobTaskWorkerExecutor(
        application_path,
        components=[{"id": "value", "factory_path": "worker_app:ValueComponent", "args": {"value": 9}}],
        poll_interval=poll_interval,
        worker_timeout=worker_timeout,
        **kwargs,
    )


def _read_events(workspace, job_id="job-1"):
    path = os.path.join(workspace.get_run_dir(job_id), ".job_task_worker_b", "events.jsonl")
    with open(path) as stream:
        return [json.loads(line) for line in stream]


def test_artifact_round_trip_binds_session_and_identity(tmp_path):
    nvflare_fobs_initialize()
    data = Shareable({"value": 1})
    write_artifact(
        str(tmp_path),
        attempt="attempt-1",
        job_id="job-1",
        session_id="session-1",
        task_id="task-1",
        task_name="train",
        kind="input",
        data=data,
    )

    task_name, restored = read_artifact(
        str(tmp_path),
        attempt="attempt-1",
        job_id="job-1",
        session_id="session-1",
        task_id="task-1",
        kind="input",
    )

    assert task_name == "train"
    assert restored["value"] == 1
    with pytest.raises(ValueError, match="stale"):
        read_artifact(
            str(tmp_path),
            attempt="attempt-1",
            job_id="job-1",
            session_id="different-session",
            task_id="task-1",
            kind="input",
        )


def test_artifact_rejects_tampering(tmp_path):
    nvflare_fobs_initialize()
    write_artifact(
        str(tmp_path),
        attempt="attempt-1",
        job_id="job-1",
        session_id="session-1",
        task_id="task-1",
        task_name="train",
        kind="result",
        data=Shareable({"value": 1}),
    )
    with open(tmp_path / "result.fobs", "ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="checksum"):
        read_artifact(
            str(tmp_path),
            attempt="attempt-1",
            job_id="job-1",
            session_id="session-1",
            task_id="task-1",
            kind="result",
        )


@pytest.mark.parametrize("fault", ["missing_manifest", "partial_manifest", "wrong_attempt"])
def test_artifact_rejects_incomplete_or_wrong_attempt(tmp_path, fault):
    nvflare_fobs_initialize()
    write_artifact(
        str(tmp_path),
        attempt="attempt-1",
        job_id="job-1",
        session_id="session-1",
        task_id="task-1",
        task_name="train",
        kind="result",
        data=Shareable({"value": 1}),
    )
    expected_attempt = "attempt-1"
    if fault == "missing_manifest":
        os.unlink(tmp_path / "result.json")
        expected = FileNotFoundError
    elif fault == "partial_manifest":
        (tmp_path / "result.json").write_text("{")
        expected = json.JSONDecodeError
    else:
        expected_attempt = "attempt-2"
        expected = ValueError

    with pytest.raises(expected):
        read_artifact(
            str(tmp_path),
            attempt=expected_attempt,
            job_id="job-1",
            session_id="session-1",
            task_id="task-1",
            kind="result",
        )


def test_artifact_rejects_pass_through(tmp_path):
    nvflare_fobs_initialize()
    data = Shareable()
    data.set_header(ReservedHeaderKey.PASS_THROUGH, True)
    with pytest.raises(ValueError, match="eager"):
        write_artifact(
            str(tmp_path),
            attempt="attempt-1",
            job_id="job-1",
            session_id="session-1",
            task_id="task-1",
            task_name="train",
            kind="input",
            data=data,
        )


def test_fresh_workers_use_registry_and_explicit_state(tmp_path, monkeypatch):
    workspace = _workspace(tmp_path)
    engine = _Engine(workspace)
    fl_ctx = _context(workspace, engine)
    executor = _executor(worker_environment={"PYTHONHASHSEED": "202610"})
    executor.handle_event(EventType.START_RUN, fl_ctx)
    for name in (JobProcessEnv.AUTH_TOKEN, JobProcessEnv.TOKEN_SIGNATURE, JobProcessEnv.SSID):
        monkeypatch.setenv(name, "must-not-leak")

    first = executor.execute("train", Shareable({"value": 3}), fl_ctx, Signal())
    fl_ctx.set_prop(FLContextKey.TASK_ID, "task-2", private=True, sticky=False)
    second = executor.execute("train", Shareable({"value": 5}), fl_ctx, Signal())

    assert (first["value"], second["value"]) == (4, 6)
    assert (first["count"], second["count"]) == (1, 2)
    assert first["pid"] != second["pid"]
    assert first["component"] == second["component"] == 9
    assert first["credentials"] == second["credentials"] == []
    assert first["python_hash_seed"] == second["python_hash_seed"] == "202610"
    assert first["has_cell"] is second["has_cell"] is False
    assert first["has_federation_session"] is second["has_federation_session"] is False


def test_script_worker_bootstraps_job_local_sibling_import_without_pythonpath(tmp_path, monkeypatch):
    custom_dir = tmp_path / "custom"
    custom_dir.mkdir()
    output = tmp_path / "imported.txt"
    (custom_dir / "sibling_probe.py").write_text('VALUE = "job-local"\n')
    script = custom_dir / "client.py"
    script.write_text(
        "from pathlib import Path\n" "from sibling_probe import VALUE\n" f"Path({str(output)!r}).write_text(VALUE)\n"
    )
    monkeypatch.delenv("PYTHONPATH", raising=False)
    isolated_sys_path = [path for path in sys.path if os.path.abspath(path or os.curdir) != str(custom_dir)]
    monkeypatch.setattr(sys, "path", isolated_sys_path.copy())

    result = _run_script(
        {
            "application_path": str(script),
            "arguments": "",
            "state_dir": str(tmp_path / "state"),
            "job_id": "job-1",
            "client_name": "site-1",
            "task_name": "train",
        },
        Shareable(),
    )

    assert output.read_text() == "job-local"
    assert result.get_return_code() == ReturnCode.EXECUTION_RESULT_ERROR
    assert sys.path == isolated_sys_path


def test_executor_worker_bootstraps_job_custom_modules_without_pythonpath(tmp_path, monkeypatch):
    workspace = _workspace(tmp_path)
    attempt_dir = tmp_path / "attempt"
    attempt_dir.mkdir()
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    spec = {
        "version": 1,
        "attempt": "attempt-1",
        "job_id": "job-1",
        "handoff_scope_id": "scope-1",
        "task_id": "task-1",
        "task_name": "train",
        "client_name": "site-1",
        "workspace_root": workspace.get_root_dir(),
        "app_root": workspace.get_app_dir("job-1"),
        "state_dir": str(state_dir),
        "kind": "executor",
        "application_path": "worker_app:ProbeExecutor",
        "application_args": {},
        "components": [{"id": "value", "factory_path": "worker_app:ValueComponent", "args": {"value": 9}}],
    }
    write_artifact(
        str(attempt_dir),
        attempt="attempt-1",
        job_id="job-1",
        session_id="scope-1",
        task_id="task-1",
        task_name="train",
        kind="input",
        data=Shareable({"value": 3}),
    )
    (attempt_dir / "worker.json").write_text(json.dumps(spec))
    monkeypatch.delenv("PYTHONPATH", raising=False)
    monkeypatch.delitem(sys.modules, "worker_app", raising=False)

    run_task_worker(str(attempt_dir))

    task_name, result = read_artifact(
        str(attempt_dir),
        attempt="attempt-1",
        job_id="job-1",
        session_id="scope-1",
        task_id="task-1",
        kind="result",
    )
    assert task_name == "train"
    assert result["value"] == 4
    assert result["component"] == 9
    assert workspace.get_app_custom_dir("job-1") not in sys.path
    sys.modules.pop("worker_app", None)


def test_event_order_places_settlement_before_cpu_publication(tmp_path):
    workspace = _workspace(tmp_path)
    engine = _Engine(workspace)
    fl_ctx = _context(workspace, engine)
    executor = _executor()
    executor.handle_event(EventType.START_RUN, fl_ctx)
    executor.execute("train", Shareable({"value": 1}), fl_ctx, Signal())
    executor.handle_event(EventType.AFTER_TASK_RESULT_FILTER, fl_ctx)
    executor.handle_event(EventType.BEFORE_SEND_TASK_RESULT, fl_ctx)
    fl_ctx.set_prop(PUBLICATION_ACK_PROP, True, private=True, sticky=False)
    executor.handle_event(EventType.AFTER_SEND_TASK_RESULT, fl_ctx)

    names = [event["event"] for event in _read_events(workspace)]
    assert names.index("input_committed") < names.index("worker_launched")
    assert names.index("worker_launched") < names.index("allocation_settled")
    assert names.index("allocation_settled") < names.index("durable_result_loaded")
    assert names.index("durable_result_loaded") < names.index("cpu_cj_result_filters_complete")
    assert names.index("cpu_cj_result_filters_complete") < names.index("cpu_cj_publication_started")
    assert names.index("cpu_cj_publication_started") < names.index("cpu_cj_publication_acknowledged")


def test_missing_ack_is_not_recorded_as_success(tmp_path):
    workspace = _workspace(tmp_path)
    engine = _Engine(workspace)
    fl_ctx = _context(workspace, engine)
    executor = _executor()
    executor.handle_event(EventType.START_RUN, fl_ctx)
    executor.execute("train", Shareable({"value": 1}), fl_ctx, Signal())
    executor.handle_event(EventType.BEFORE_SEND_TASK_RESULT, fl_ctx)
    fl_ctx.set_prop(PUBLICATION_ACK_PROP, False, private=True, sticky=False)
    executor.handle_event(EventType.AFTER_SEND_TASK_RESULT, fl_ctx)

    names = [event["event"] for event in _read_events(workspace)]
    assert "cpu_cj_publication_not_acknowledged" in names
    assert "cpu_cj_publication_acknowledged" not in names
    not_ack = next(
        event for event in _read_events(workspace) if event["event"] == "cpu_cj_publication_not_acknowledged"
    )
    attempt_dir = os.path.join(workspace.get_run_dir("job-1"), ".job_task_worker_b", "attempts", not_ack["attempt"])
    assert os.path.isfile(os.path.join(attempt_dir, "result.fobs"))
    assert os.path.isfile(os.path.join(attempt_dir, "result.json"))


def test_compute_crash_has_no_durable_result(tmp_path):
    workspace = _workspace(tmp_path)
    engine = _Engine(workspace)
    fl_ctx = _context(workspace, engine)
    executor = _executor("worker_app:CrashExecutor")
    executor.handle_event(EventType.START_RUN, fl_ctx)

    with pytest.raises(RuntimeError, match="code 7"):
        executor.execute("train", Shareable(), fl_ctx, Signal())

    root = os.path.join(workspace.get_run_dir("job-1"), ".job_task_worker_b", "attempts")
    attempt = os.listdir(root)[0]
    assert not os.path.exists(os.path.join(root, attempt, "result.json"))


def test_application_exception_has_no_durable_result(tmp_path):
    workspace = _workspace(tmp_path)
    engine = _Engine(workspace)
    fl_ctx = _context(workspace, engine)
    executor = _executor("worker_app:RaiseExecutor")
    executor.handle_event(EventType.START_RUN, fl_ctx)

    with pytest.raises(RuntimeError, match="exited with code"):
        executor.execute("train", Shareable(), fl_ctx, Signal())

    root = os.path.join(workspace.get_run_dir("job-1"), ".job_task_worker_b", "attempts")
    attempt = os.listdir(root)[0]
    assert not os.path.exists(os.path.join(root, attempt, "result.json"))


def test_abort_cleans_worker_and_rejects_later_admission(tmp_path):
    workspace = _workspace(tmp_path)
    engine = _Engine(workspace)
    fl_ctx = _context(workspace, engine)
    executor = _executor("worker_app:SlowExecutor", worker_timeout=60)
    executor.handle_event(EventType.START_RUN, fl_ctx)
    outcome = []

    def run():
        try:
            executor.execute("train", Shareable(), fl_ctx, Signal())
        except Exception as e:
            outcome.append(e)

    thread = threading.Thread(target=run)
    thread.start()
    deadline = time.time() + 5
    while executor._active is None and time.time() < deadline:
        time.sleep(0.01)
    assert executor._active is not None
    worker_pid = executor._active.process.pid
    executor.handle_event(EventType.ABORT_TASK, fl_ctx)
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert outcome and "cancelled" in str(outcome[0])
    with pytest.raises(ProcessLookupError):
        os.kill(worker_pid, 0)
    with pytest.raises(RuntimeError, match="rejects new"):
        executor.execute("train", Shareable(), fl_ctx, Signal())


def test_local_handle_kills_descendants_before_settlement(tmp_path):
    child_file = tmp_path / "child.pid"
    code = (
        "import pathlib,subprocess,sys; "
        f"p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); "
        f"pathlib.Path({str(child_file)!r}).write_text(str(p.pid))"
    )
    process = subprocess.Popen([sys.executable, "-c", code], start_new_session=True)
    handle = LocalProcessHandle(process, stop_grace=1.0, descendant_settle_timeout=0.05)

    with pytest.raises(RuntimeError, match="live descendants"):
        handle.wait()

    child_pid = int(child_file.read_text())
    assert handle.settled
    with pytest.raises(ProcessLookupError):
        os.kill(child_pid, 0)


@pytest.mark.parametrize(
    "kwargs, expected_mode, expected_script",
    [
        (
            {"execution_mode": "in_process", "task_script_path": "client.py", "launch_once": False},
            "in_process",
            "client.py",
        ),
        (
            {
                "execution_mode": "external_process",
                "command": ["python3", "-u", "custom/client.py", "--x", "1"],
                "launch_once": False,
            },
            "external_process",
            "client.py",
        ),
    ],
)
def test_client_api_adapter_maps_existing_modes_to_fresh_script_worker(kwargs, expected_mode, expected_script):
    executor = ClientAPIJobTaskWorkerExecutor(**kwargs)

    assert executor.kind == "script"
    assert executor.source_mode == expected_mode
    assert executor.application_path == expected_script
    assert executor.state_id == "client-api"
    assert executor._launch_once is False


def test_client_api_adapter_rejects_attach():
    with pytest.raises(ValueError, match="not Attach"):
        ClientAPIJobTaskWorkerExecutor(execution_mode="attach")


def test_worker_environment_rejects_federation_credentials():
    with pytest.raises(ValueError, match="credentials"):
        _executor(worker_environment={JobProcessEnv.AUTH_TOKEN: "must-not-forward"})


class _RunnerEngine:
    def __init__(self, submitted):
        self.assignment = TaskAssignment("train", "task-1", Shareable())
        self.submitted = submitted

    def register_aux_message_handler(self, **kwargs):
        pass

    def get_task_assignment(self, fl_ctx, timeout=None):
        return self.assignment


@pytest.mark.parametrize("submitted", [True, False])
def test_standard_client_runner_exposes_actual_publication_ack(submitted):
    engine = _RunnerEngine(submitted)
    runner = ClientRunner(
        client_config={},
        config=ClientRunnerConfig(TaskRouter(), {}, {}),
        job_id="job-1",
        engine=engine,
    )
    runner._process_task = lambda task, fl_ctx: Shareable()
    runner._send_task_result = lambda result, task_id, fl_ctx: submitted
    observed = []
    runner.fire_event = lambda event_type, fl_ctx: observed.append((event_type, fl_ctx.get_prop(PUBLICATION_ACK_PROP)))
    manager = FLContextManager(engine=engine, identity_name="site-1", job_id="job-1")

    runner.fetch_and_run_one_task(manager.new_context())

    assert observed[-1] == (EventType.AFTER_SEND_TASK_RESULT, submitted)


def test_slurm_plan_sizes_worker_not_resident_cj(tmp_path):
    workspace = _workspace(tmp_path)
    os.chmod(workspace.get_root_dir(), 0o700)
    launcher = SlurmTaskWorkerLauncher(
        workspace_path=workspace.get_root_dir(),
        sandbox="none",
        python_path="/usr/bin/python3",
        executables={name: "/usr/bin/true" for name in ("sbatch", "squeue", "sacct", "scancel")},
        worker_resources={"nodes": 1, "gpus_per_node": 2, "cpus_per_node": 4, "mem_per_node": 16384},
        task_environment={"PYTHONHASHSEED": "202610"},
    )
    fl_ctx = _context(workspace, _Engine(workspace))
    attempt = os.path.join(workspace.get_run_dir("job-1"), ".job_task_worker_b", "attempts", "attempt-1")
    os.makedirs(attempt)

    plan = launcher._build_task_plan(attempt, fl_ctx)

    assert plan.exe_module == "nvflare.private.fed.job_task_worker.worker"
    assert plan.module_args == (attempt,)
    assert plan.resources.gpus_per_node == 2
    assert plan.resources.cpus_per_node == 4
    assert plan.resources.mem_per_node == 16384
    assert plan.study_env["PYTHONHASHSEED"] == "202610"
    assert plan.study_secret_env == {}
    assert plan.additional_node_command == ()


def test_slurm_launcher_rejects_credential_environment(tmp_path):
    workspace = _workspace(tmp_path)
    os.chmod(workspace.get_root_dir(), 0o700)
    launcher = SlurmTaskWorkerLauncher(
        workspace_path=workspace.get_root_dir(),
        sandbox="none",
        python_path="/usr/bin/python3",
        executables={name: "/usr/bin/true" for name in ("sbatch", "squeue", "sacct", "scancel")},
    )
    attempt = os.path.join(workspace.get_run_dir("job-1"), "attempt")
    os.makedirs(attempt)
    command = [sys.executable, "-m", "nvflare.private.fed.job_task_worker.worker", attempt]

    with pytest.raises(Exception, match="credential"):
        launcher.launch(
            command,
            cwd=workspace.get_run_dir("job-1"),
            env={JobProcessEnv.AUTH_TOKEN: "secret"},
            fl_ctx=_context(workspace, _Engine(workspace)),
        )
