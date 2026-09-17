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

import errno
import json
import os
import shutil
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call

import numpy as np
import pytest

from nvflare.apis.filter import Filter
from nvflare.apis.impl.wf_comm_server import WFCommServer
from nvflare.apis.job_launcher_spec import JobProcessEnv
from nvflare.apis.shareable import ReservedHeaderKey, Shareable
from nvflare.app_common.abstract.fl_model import FLModel
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.np.constants import NPConstants
from nvflare.app_common.utils.fl_model_utils import FLModelUtils
from nvflare.client.config import ConfigKey, ExchangeFormat, TransferType
from nvflare.fuel.f3.cellnet.cell import Cell
from nvflare.private.defs import CellChannel
from nvflare.private.fed.deployment_supervisor import (
    ControllerTaskServiceAdapter,
    DeploymentTaskSupervisor,
    LocalProcessLauncher,
    WorkerDefinition,
    artifacts,
)
from nvflare.private.fed.deployment_supervisor.artifacts import read_artifact, write_artifact
from nvflare.private.fed.deployment_supervisor.config import load_worker_routes
from nvflare.private.fed.deployment_supervisor.federation import (
    DeploymentFederationJobHandle,
    DeploymentTaskService,
    FederatedTaskServiceClient,
)
from nvflare.private.fed.deployment_supervisor.protocol import (
    ACK,
    ACQUIRE_TOPIC,
    ATTEMPT,
    DATA,
    ERROR,
    PUBLISH_TOPIC,
    STATUS,
    TASK_ID,
    TASK_NAME,
)
from nvflare.private.fed.deployment_supervisor.slurm import SlurmTaskWorkerHandle, SlurmTaskWorkerLauncher

ROOT = Path(__file__).resolve().parents[4]
EXAMPLE = ROOT / "examples" / "advanced" / "deployment-supervisor-a"


def _supervisor(tmp_path, service, routes, **kwargs):
    env_path = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = os.pathsep.join(p for p in (str(ROOT), str(EXAMPLE), env_path) if p)
    return DeploymentTaskSupervisor(service=service, workspace=tmp_path, routes=routes, **kwargs)


def _shareable(value):
    result = Shareable()
    result["value"] = value
    return result


def test_broadcast_and_send_across_clients_and_rounds(tmp_path):
    service = ControllerTaskServiceAdapter()
    ids = []
    for round_number in range(3):
        ids.extend(
            service.broadcast(
                job_id="job",
                task_name="train",
                data=_shareable(round_number),
                clients=["site-1", "site-2"],
                round_number=round_number,
            )
        )
    ids.append(service.send(job_id="job", task_name="validate", data=_shareable(9), client="site-2"))
    worker = WorkerDefinition(kind="executor", entrypoint="components:IncrementExecutor")
    for site in ("site-1", "site-2"):
        assert _supervisor(tmp_path / site, service, {"*": worker}).run_until_idle(site) in (3, 4)
    assert [service.result(task_id)["value"] for task_id in ids] == [1, 1, 2, 2, 3, 3, 10]
    assert service.is_idle()


def test_explicit_state_survives_fresh_workers(tmp_path):
    service = ControllerTaskServiceAdapter()
    ids = [
        service.send(job_id="job", task_name="train", data=_shareable(i), client="site-1", round_number=i)
        for i in range(2)
    ]
    worker = WorkerDefinition(kind="executor", entrypoint="components:DeclaredStateExecutor", state_id="training-state")
    assert _supervisor(tmp_path, service, {"train": worker}).run_until_idle("site-1") == 2
    assert [service.result(task_id)["count"] for task_id in ids] == [1, 2]


def test_task_local_component_and_declared_state_survive_fresh_workers(tmp_path):
    service = ControllerTaskServiceAdapter()
    task_ids = [
        service.send(job_id="job", task_name="train", data=_shareable(value), client="site-1", round_number=value)
        for value in (1, 2)
    ]
    worker = WorkerDefinition(
        kind="executor",
        entrypoint="components:ComponentStateExecutor",
        state_id="component-state",
        task_components=[{"id": "offset", "path": "components:OffsetComponent", "args": {"amount": 10}}],
    )
    assert _supervisor(tmp_path, service, {"train": worker}).run_until_idle("site-1") == 2
    assert [service.result(task_id)["value"] for task_id in task_ids] == [11, 12]
    assert [service.result(task_id)["count"] for task_id in task_ids] == [1, 2]


def test_t4_workspace_state_is_continuous_and_isolated_across_fresh_workers(tmp_path):
    service = ControllerTaskServiceAdapter()
    task_ids = {site: [] for site in ("site-1", "site-2")}
    for round_number in range(3):
        for site in task_ids:
            task_ids[site].append(
                service.send(
                    job_id="t4-job",
                    task_name="count",
                    data=Shareable({"round": round_number}),
                    client=site,
                    round_number=round_number,
                )
            )
    for site in task_ids:
        worker = WorkerDefinition(
            kind="executor",
            entrypoint="t4_components:WorkspaceCounterExecutor",
            factory_args={"compute_seconds": 0.0},
            state_id=site,
        )
        assert _supervisor(tmp_path / site, service, {"count": worker}).run_until_idle(site) == 3

    for site, ids in task_ids.items():
        results = [service.result(task_id) for task_id in ids]
        assert [(result["round"], result["value"]) for result in results] == [(0, 1), (1, 2), (2, 3)]
        assert len({result["pid"] for result in results}) == 3
        checkpoint = tmp_path / site / "state" / "t4-job" / site / "task_scope_counter.json"
        assert json.loads(checkpoint.read_text())["value"] == 3
    site_1_checkpoint = tmp_path / "site-1" / "state" / "t4-job" / "site-1" / "task_scope_counter.json"
    site_2_checkpoint = tmp_path / "site-2" / "state" / "t4-job" / "site-2" / "task_scope_counter.json"
    assert site_1_checkpoint != site_2_checkpoint
    assert json.loads(site_1_checkpoint.read_text())["round"] == 2
    assert json.loads(site_2_checkpoint.read_text())["round"] == 2


@pytest.mark.parametrize("legacy_mode", ["in_process", "external_process"])
def test_existing_hello_numpy_script_runs_through_managed_worker(tmp_path, legacy_mode):
    """The same hello script used by both legacy modes needs no phase or persistence code."""
    service = ControllerTaskServiceAdapter()
    model = FLModel(
        params={NPConstants.NUMPY_KEY: np.array([1.0, 2.0])},
        current_round=1,
        total_rounds=2,
    )
    task_data = FLModelUtils.to_shareable(model)
    task_data.add_cookie(AppConstants.CONTRIBUTION_ROUND, 1)
    task_id = service.send(
        job_id=f"hello-{legacy_mode}",
        task_name="train",
        data=task_data,
        client="site-1",
        round_number=1,
    )
    exchange = {
        ConfigKey.EXCHANGE_FORMAT: ExchangeFormat.NUMPY,
        ConfigKey.SERVER_EXPECTED_FORMAT: ExchangeFormat.NUMPY,
        ConfigKey.TRANSFER_TYPE: TransferType.FULL,
        ConfigKey.TRAIN_TASK_NAME: "train",
        ConfigKey.EVAL_TASK_NAME: "validate",
        ConfigKey.SUBMIT_MODEL_TASK_NAME: "submit_model",
    }
    worker = WorkerDefinition(
        kind="script",
        entrypoint=str(ROOT / "examples" / "hello-world" / "hello-numpy" / "client.py"),
        task_exchange=exchange,
    )
    assert _supervisor(tmp_path, service, {"train": worker}).run_once("site-1")
    output = FLModelUtils.from_shareable(service.result(task_id))
    np.testing.assert_array_equal(output.params[NPConstants.NUMPY_KEY], np.array([2.0, 3.0]))
    assert output.metrics["weight_mean"] == 2.5
    assert service.result(task_id).get_cookie(AppConstants.CONTRIBUTION_ROUND) == 1


@pytest.mark.parametrize(("mode", "expected_code"), [("exception", 1), ("nonzero", 7)])
def test_g06_fault_executor_fails_without_result_publication(tmp_path, mode, expected_code):
    service = ControllerTaskServiceAdapter()
    task_id = service.send(
        job_id=f"g06-{mode}",
        task_name="train",
        data=_shareable(1),
        client="site-1",
    )
    custom_dir = tmp_path / "custom"
    custom_dir.mkdir()
    custom_module = "g06_fault_executor_custom_only"
    shutil.copyfile(EXAMPLE / "g06_fault_executor.py", custom_dir / f"{custom_module}.py")
    worker = WorkerDefinition(
        kind="executor",
        entrypoint=f"{custom_module}:G06FaultExecutor",
        factory_args={"mode": mode},
        module_search_paths=[str(custom_dir)],
    )

    assert _supervisor(tmp_path, service, {"train": worker}).run_once("site-1")
    assert service.result(task_id) is None
    assert service.failure(task_id) == f"task worker exited with code {expected_code}"
    assert not list(tmp_path.rglob("result.json"))
    assert not list(tmp_path.rglob("result.fobs"))


class _TraceFilter(Filter):
    def __init__(self, name, calls):
        super().__init__()
        self.trace_name = name
        self.calls = calls

    def process(self, shareable, fl_ctx):
        self.calls.append(self.trace_name)
        shareable.setdefault("trace", []).append(self.trace_name)
        return shareable


def test_site_then_job_filter_order_is_preserved_in_both_directions(tmp_path):
    service = ControllerTaskServiceAdapter()
    calls = []
    task_id = service.send(job_id="job", task_name="train", data=_shareable(0), client="site-1")
    supervisor = _supervisor(
        tmp_path,
        service,
        {"train": WorkerDefinition(kind="executor", entrypoint="components:IncrementExecutor")},
        site_input_filters=[_TraceFilter("site-in", calls)],
        job_input_filters=[_TraceFilter("job-in", calls)],
        site_result_filters=[_TraceFilter("site-out", calls)],
        job_result_filters=[_TraceFilter("job-out", calls)],
    )
    supervisor.run_once("site-1")
    assert calls == ["site-in", "job-in", "site-out", "job-out"]
    assert service.result(task_id)["trace"] == ["site-out", "job-out"]


def test_job_config_filters_execute_inside_fresh_worker(tmp_path):
    service = ControllerTaskServiceAdapter()
    task_id = service.send(job_id="job", task_name="train", data=_shareable(1), client="site-1")
    definition = {"path": "components.AddFilter", "args": {"amount": 10}}
    worker = WorkerDefinition(
        kind="executor",
        entrypoint="components:IncrementExecutor",
        input_filters=[definition],
        result_filters=[definition],
    )
    assert _supervisor(tmp_path, service, {"train": worker}).run_once("site-1")
    assert service.result(task_id)["value"] == 22


def test_publish_happens_only_after_handle_settlement(tmp_path):
    class ObservedService(ControllerTaskServiceAdapter):
        handle = None

        def publish(self, assignment, result):
            assert self.handle.settled
            super().publish(assignment, result)

    class ObservedLauncher(LocalProcessLauncher):
        def launch(self, command, *, cwd, env):
            handle = super().launch(command, cwd=cwd, env=env)
            service.handle = handle
            return handle

    service = ObservedService()
    task_id = service.send(job_id="job", task_name="train", data=_shareable(1), client="site-1")
    supervisor = _supervisor(
        tmp_path,
        service,
        {"train": WorkerDefinition(kind="executor", entrypoint="components:IncrementExecutor")},
        launcher=ObservedLauncher(),
    )
    supervisor.run_once("site-1")
    assert service.result(task_id)["value"] == 2
    events = [json.loads(line)["event"] for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert events.index("allocation_settled") < events.index("published")


def test_supervisor_reads_terminal_status_from_poll_when_wait_returns_none(tmp_path):
    class NoneReturningWaitHandle:
        def __init__(self, handle):
            self.handle = handle

        def poll(self):
            return self.handle.poll()

        def wait(self):
            self.handle.wait()
            return None

        def terminate(self):
            return self.handle.terminate()

    class Launcher:
        def launch(self, command, *, cwd, env):
            handle = LocalProcessLauncher().launch(command, cwd=cwd, env=env)
            return NoneReturningWaitHandle(handle)

    service = ControllerTaskServiceAdapter()
    task_id = service.send(job_id="job", task_name="train", data=_shareable(1), client="site-1")
    supervisor = _supervisor(
        tmp_path,
        service,
        {"train": WorkerDefinition(kind="executor", entrypoint="components:IncrementExecutor")},
        launcher=Launcher(),
    )

    assert supervisor.run_once("site-1")
    assert service.result(task_id)["value"] == 2
    assert service.failure(task_id) is None


def test_publication_handlers_observe_real_publish_boundary(tmp_path):
    calls = []

    class Handler:
        def before_publish(self, assignment, result, fl_ctx):
            calls.append(("before", assignment.task_id, result["value"]))

        def after_publish(self, assignment, result, fl_ctx):
            calls.append(("after", assignment.task_id, result["value"]))

    service = ControllerTaskServiceAdapter()
    task_id = service.send(job_id="job", task_name="train", data=_shareable(1), client="site-1")
    supervisor = _supervisor(
        tmp_path,
        service,
        {"train": WorkerDefinition(kind="executor", entrypoint="components:IncrementExecutor")},
        publication_handlers=[Handler()],
    )
    supervisor.run_once("site-1")
    assert calls == [("before", task_id, 2), ("after", task_id, 2)]


def test_task_worker_does_not_inherit_federation_credentials(tmp_path, monkeypatch):
    for name in (JobProcessEnv.AUTH_TOKEN, JobProcessEnv.TOKEN_SIGNATURE, JobProcessEnv.SSID):
        monkeypatch.setenv(name, "secret")
    service = ControllerTaskServiceAdapter()
    task_id = service.send(job_id="job", task_name="train", data=_shareable(1), client="site-1")
    supervisor = _supervisor(
        tmp_path,
        service,
        {"train": WorkerDefinition(kind="executor", entrypoint="components:CredentialProbeExecutor")},
    )
    supervisor.run_once("site-1")
    assert service.result(task_id)["credentials"] == []


def test_timeout_is_bounded_and_failure_is_not_published(tmp_path):
    script = tmp_path / "slow.py"
    script.write_text("import time\ntime.sleep(30)\n")
    service = ControllerTaskServiceAdapter()
    task_id = service.send(job_id="job", task_name="train", data=_shareable(1), client="site-1")
    supervisor = _supervisor(
        tmp_path / "workspace",
        service,
        {"train": WorkerDefinition(kind="script", entrypoint=str(script))},
        worker_timeout=0.2,
        poll_interval=0.01,
        launcher=LocalProcessLauncher(stop_grace=0.1),
    )
    supervisor.run_once("site-1")
    assert service.result(task_id) is None
    assert "code" in service.failure(task_id)


def test_cancellation_terminates_worker_process_group(tmp_path):
    script = tmp_path / "slow.py"
    script.write_text("import time\ntime.sleep(30)\n")
    service = ControllerTaskServiceAdapter()
    task_id = service.send(job_id="job", task_name="train", data=_shareable(1), client="site-1")
    supervisor = _supervisor(
        tmp_path / "workspace",
        service,
        {"train": WorkerDefinition(kind="script", entrypoint=str(script))},
        worker_timeout=30,
        poll_interval=0.01,
        launcher=LocalProcessLauncher(stop_grace=0.1),
    )
    thread = threading.Thread(target=supervisor.run_once, args=("site-1",))
    thread.start()
    deadline = time.monotonic() + 3
    while supervisor._active is None and time.monotonic() < deadline:
        time.sleep(0.01)
    supervisor.cancel()
    thread.join(3)
    assert not thread.is_alive()
    assert service.result(task_id) is None
    assert service.failure(task_id) == "task cancelled"


def test_leader_exit_with_live_descendant_is_cleaned_and_not_settled_as_success(tmp_path):
    child_pid_file = tmp_path / "child.pid"
    code = (
        "import pathlib, subprocess, sys; "
        "p=subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)']); "
        f"pathlib.Path({str(child_pid_file)!r}).write_text(str(p.pid))"
    )
    handle = LocalProcessLauncher(stop_grace=0.2, descendant_settle_timeout=0.05).launch(
        [sys.executable, "-c", code], cwd=str(tmp_path), env=os.environ.copy()
    )
    with pytest.raises(RuntimeError, match="live descendants"):
        handle.wait()
    child_pid = int(child_pid_file.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(child_pid, 0)
    assert handle.settled


def test_exception_after_launch_forces_handle_cleanup(tmp_path):
    class BrokenPollHandle:
        def __init__(self):
            self.terminated = False
            self.waited = False

        def poll(self):
            raise RuntimeError("poll failed")

        def terminate(self):
            self.terminated = True

        def wait(self):
            self.waited = True
            return -1

    class BrokenPollLauncher:
        def __init__(self):
            self.handle = BrokenPollHandle()

        def launch(self, command, *, cwd, env):
            return self.handle

    launcher = BrokenPollLauncher()
    service = ControllerTaskServiceAdapter()
    task_id = service.send(job_id="job", task_name="train", data=_shareable(1), client="site-1")
    supervisor = _supervisor(
        tmp_path,
        service,
        {"train": WorkerDefinition(kind="executor", entrypoint="components:IncrementExecutor")},
        launcher=launcher,
    )
    supervisor.run_once("site-1")
    assert launcher.handle.terminated
    assert launcher.handle.waited
    assert service.result(task_id) is None
    assert service.failure(task_id) == "poll failed"


def test_artifact_rejects_tampering_and_stale_identity(tmp_path):
    write_artifact(
        tmp_path,
        attempt="attempt",
        job_id="job",
        task_id="task",
        task_name="train",
        kind="input",
        data=_shareable(1),
    )
    with pytest.raises(ValueError, match="stale"):
        read_artifact(tmp_path, attempt="other", job_id="job", task_id="task", kind="input")
    with open(tmp_path / "input.fobs", "ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="checksum"):
        read_artifact(tmp_path, attempt="attempt", job_id="job", task_id="task", kind="input")


def test_artifact_retries_transient_nonblocking_open_and_restores_blocking_io(tmp_path, monkeypatch):
    write_artifact(
        tmp_path,
        attempt="attempt",
        job_id="job",
        task_id="task",
        task_name="train",
        kind="input",
        data=_shareable(1),
    )
    real_open = artifacts.os.open
    calls = 0

    def transient_open(path, flags, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise BlockingIOError(errno.EAGAIN, "transient shared-filesystem open")
        return real_open(path, flags, *args, **kwargs)

    set_blocking = Mock(wraps=artifacts.os.set_blocking)
    monkeypatch.setattr(artifacts.os, "open", transient_open)
    monkeypatch.setattr(artifacts.os, "set_blocking", set_blocking)
    monkeypatch.setattr(artifacts.time, "sleep", lambda delay: None)

    task_name, data = read_artifact(tmp_path, attempt="attempt", job_id="job", task_id="task", kind="input")
    assert task_name == "train"
    assert data["value"] == 1
    assert calls >= 3  # retried manifest open, then opened the payload
    assert set_blocking.call_count == 2


def test_artifact_transient_open_retry_is_bounded(tmp_path, monkeypatch):
    path = tmp_path / "input.json"
    path.write_text("{}")
    times = iter((0.0, artifacts._OPEN_RETRY_TIMEOUT + 1.0))
    monkeypatch.setattr(artifacts.time, "monotonic", lambda: next(times))
    monkeypatch.setattr(
        artifacts.os,
        "open",
        lambda *args, **kwargs: (_ for _ in ()).throw(BlockingIOError(errno.EAGAIN, "still unavailable")),
    )

    with pytest.raises(BlockingIOError, match="still unavailable"):
        artifacts._open_regular(path)


def test_artifact_rejects_live_pass_through_reference(tmp_path):
    data = _shareable(1)
    data.set_header(ReservedHeaderKey.PASS_THROUGH, True)
    with pytest.raises(ValueError, match="eager data"):
        write_artifact(
            tmp_path,
            attempt="attempt",
            job_id="job",
            task_id="task",
            task_name="train",
            kind="input",
            data=data,
        )


def test_resident_requirement_is_rejected_with_migration_direction(tmp_path):
    service = ControllerTaskServiceAdapter()
    task_id = service.send(job_id="job", task_name="train", data=_shareable(1), client="site-1")
    worker = WorkerDefinition(kind="executor", entrypoint="components:IncrementExecutor", lifetime="resident")
    _supervisor(tmp_path, service, {"train": worker}).run_once("site-1")
    assert service.result(task_id) is None
    assert "unchanged resident CJ path" in service.failure(task_id)


@pytest.mark.parametrize(
    "mode,mode_args,expected_args",
    [
        ("in_process", {"task_script_path": "client.py", "task_script_args": "--epochs 1"}, "--epochs 1"),
        ("external_process", {"command": "python -u custom/client.py --epochs 1"}, "--epochs 1"),
    ],
)
def test_deployed_client_api_modes_map_to_same_managed_worker_contract(
    tmp_path, monkeypatch, mode, mode_args, expected_args
):
    config = {
        "components": [{"id": "offset", "path": "pkg.Offset", "args": {"amount": 4}}],
        "executors": [
            {
                "tasks": ["train"],
                "executor": {
                    "path": "nvflare.app_common.executors.client_api_executor.ClientAPIExecutor",
                    "args": {
                        "execution_mode": mode,
                        "params_exchange_format": "pytorch",
                        "params_transfer_type": "DIFF",
                        "train_task_name": "train",
                        "memory_gc_rounds": 2,
                        "cuda_empty_cache": True,
                        **mode_args,
                    },
                },
            }
        ],
        "task_data_filters": [{"tasks": ["train"], "filters": [{"path": "pkg.Input"}]}],
        "task_result_filters": [{"tasks": ["*"], "filters": [{"path": "pkg.Output"}]}],
    }

    class Loaded:
        def to_dict(self):
            return config

    monkeypatch.setattr(
        "nvflare.private.fed.deployment_supervisor.config.ConfigFactory.load_config", lambda *args: Loaded()
    )
    workspace = SimpleNamespace(
        get_app_config_dir=lambda job_id: str(tmp_path / "config"),
        get_app_custom_dir=lambda job_id: str(tmp_path / "custom"),
    )
    worker = load_worker_routes(workspace, "job")["train"]
    assert worker.kind == "script"
    assert worker.entrypoint == str(tmp_path / "custom" / "client.py")
    assert worker.arguments == expected_args
    assert worker.source_mode == mode
    assert worker.module_search_paths == [str(tmp_path / "custom")]
    assert worker.task_exchange == {
        ConfigKey.EXCHANGE_FORMAT: "pytorch",
        ConfigKey.SERVER_EXPECTED_FORMAT: "numpy",
        ConfigKey.TRANSFER_TYPE: "DIFF",
        ConfigKey.TRAIN_TASK_NAME: "train",
        ConfigKey.EVAL_TASK_NAME: "validate",
        ConfigKey.SUBMIT_MODEL_TASK_NAME: "submit_model",
        ConfigKey.TRAIN_WITH_EVAL: False,
    }
    assert worker.memory_gc_rounds == 2
    assert worker.cuda_empty_cache is True
    assert worker.task_components == [{"id": "offset", "path": "pkg.Offset", "args": {"amount": 4}}]
    assert worker.input_filters == [{"path": "pkg.Input"}]
    assert worker.result_filters == [{"path": "pkg.Output"}]


def test_deployed_custom_executor_imports_inside_fresh_worker(tmp_path, monkeypatch):
    custom_dir = tmp_path / "custom"
    custom_dir.mkdir()
    module_name = "deployed_g06_fault_executor_custom_only"
    shutil.copyfile(EXAMPLE / "g06_fault_executor.py", custom_dir / f"{module_name}.py")
    config = {
        "executors": [
            {
                "tasks": ["train"],
                "executor": {
                    "path": f"{module_name}:G06FaultExecutor",
                    "args": {"mode": "nonzero"},
                },
            }
        ]
    }

    class Loaded:
        def to_dict(self):
            return config

    monkeypatch.setattr(
        "nvflare.private.fed.deployment_supervisor.config.ConfigFactory.load_config", lambda *args: Loaded()
    )
    workspace = SimpleNamespace(
        get_app_config_dir=lambda job_id: str(tmp_path / "config"),
        get_app_custom_dir=lambda job_id: str(custom_dir),
    )
    worker = load_worker_routes(workspace, "job")["train"]
    assert worker.module_search_paths == [str(custom_dir)]

    service = ControllerTaskServiceAdapter()
    task_id = service.send(job_id="job", task_name="train", data=_shareable(1), client="site-1")
    assert _supervisor(tmp_path / "supervisor", service, {"train": worker}).run_once("site-1")
    assert service.result(task_id) is None
    assert service.failure(task_id) == "task worker exited with code 7"


def test_server_publication_ack_requires_controller_acceptance(monkeypatch):
    communicator = WFCommServer()
    client = SimpleNamespace(name="site-1")
    task = SimpleNamespace(name="train")
    client_task = SimpleNamespace(client=client, task=task, result_received_time=None)
    communicator._client_task_map["task-1"] = client_task
    runner = SimpleNamespace(
        status="started",
        current_wf=SimpleNamespace(controller=SimpleNamespace(communicator=communicator)),
        wf_lock=threading.RLock(),
    )

    def accept(client, task_name, task_id, result, fl_ctx):
        client_task.result_received_time = time.time()

    runner.process_submission = accept
    fl_ctx = SimpleNamespace(get_prop=lambda key: runner)
    service = DeploymentTaskService()
    service._attempts[("site-1", "task-1")] = "attempt-1"
    monkeypatch.setattr(service, "_authenticated", lambda request: (client, fl_ctx, None))
    payload = Shareable()
    payload.update({TASK_NAME: "train", TASK_ID: "task-1", ATTEMPT: "attempt-1", DATA: Shareable({"result": 1})})
    response = service._handle_publish(SimpleNamespace(payload=payload)).payload
    assert response[STATUS] == ACK
    assert ("site-1", "task-1") not in service._attempts


def test_server_service_start_registers_on_cell_stream_dispatch_and_is_idempotent():
    # Use Cell's real dynamic dispatch boundary. AUX is a streaming channel:
    # keyword registration must select _register_request_cb, which installs
    # both CoreCell and blob callbacks. Positional registration selects the
    # raw CoreCell method instead and caused the live topic-unknown failure.
    cell = Cell.__new__(Cell)
    cell.logger = Mock()
    cell.core_cell = SimpleNamespace(register_request_cb=Mock())
    cell._register_request_cb = Mock()
    engine = SimpleNamespace(get_cell=lambda: cell)
    service = DeploymentTaskService()

    service.start(engine, "job-1")
    service.start(engine, "job-1")

    assert cell._register_request_cb.call_args_list == [
        call(channel=CellChannel.AUX_COMMUNICATION, topic=ACQUIRE_TOPIC, cb=service._handle_acquire),
        call(channel=CellChannel.AUX_COMMUNICATION, topic=PUBLISH_TOPIC, cb=service._handle_publish),
    ]
    cell.core_cell.register_request_cb.assert_not_called()


def test_federated_task_client_targets_the_job_scoped_server_service():
    captured = {}

    class AuxRunner:
        def send_aux_request(self, **kwargs):
            captured.update(kwargs)
            return {"server": Shareable()}

    class Context:
        def put(self, *args, **kwargs):
            pass

    class ContextFactory:
        def __enter__(self):
            return Context()

        def __exit__(self, *args):
            return False

    engine = SimpleNamespace(aux_runner=AuxRunner(), new_context=ContextFactory)
    client = FederatedTaskServiceClient(engine, "job-1", "site-1")
    client._request(ACQUIRE_TOPIC, Shareable())

    target = captured["targets"][0]
    assert target.name == "server"
    assert target.fqcn == "server.job-1"
    assert target.job_scoped is False


def test_server_service_rejects_rebinding_to_another_run():
    engine = SimpleNamespace(get_cell=lambda: SimpleNamespace(register_request_cb=Mock()))
    service = DeploymentTaskService()
    service.start(engine, "job-1")

    with pytest.raises(RuntimeError, match="already bound"):
        service.start(engine, "job-2")


def test_server_publication_does_not_ack_a_dropped_result(monkeypatch):
    communicator = WFCommServer()
    client = SimpleNamespace(name="site-1")
    communicator._client_task_map["task-1"] = SimpleNamespace(
        client=client, task=SimpleNamespace(name="train"), result_received_time=None
    )
    runner = SimpleNamespace(
        status="started",
        current_wf=SimpleNamespace(controller=SimpleNamespace(communicator=communicator)),
        wf_lock=threading.RLock(),
        process_submission=lambda *args: None,
    )
    service = DeploymentTaskService()
    service._attempts[("site-1", "task-1")] = "attempt-1"
    monkeypatch.setattr(
        service, "_authenticated", lambda request: (client, SimpleNamespace(get_prop=lambda key: runner), None)
    )
    payload = Shareable()
    payload.update({TASK_NAME: "train", TASK_ID: "task-1", ATTEMPT: "attempt-1", DATA: Shareable({"result": 1})})
    response = service._handle_publish(SimpleNamespace(payload=payload)).payload
    assert response[STATUS] == ERROR
    assert service._attempts[("site-1", "task-1")] == "attempt-1"


def test_job_abort_prevents_another_task_admission():
    class Supervisor:
        def __init__(self):
            self.calls = 0
            self.cancelled = False

        def run_once(self, client_name):
            self.calls += 1
            return False

        def cancel(self):
            self.cancelled = True

    supervisor = Supervisor()
    handle = DeploymentFederationJobHandle(
        supervisor,
        SimpleNamespace(terminal=False, last_error=None),
        "site-1",
        poll_interval=0.001,
        communication_timeout=1.0,
    )
    handle.request_abort()
    assert handle.wait() == 9
    assert supervisor.calls == 0
    assert supervisor.cancelled


def test_cancelled_supervisor_does_not_acquire_next_task(tmp_path):
    class Service:
        def __init__(self):
            self.acquire_calls = 0

        def acquire(self, client_name):
            self.acquire_calls += 1
            return None

    service = Service()
    supervisor = _supervisor(tmp_path, service, {})
    supervisor.cancel()
    assert not supervisor.run_once("site-1")
    assert service.acquire_calls == 0


def test_slurm_task_worker_plan_removes_federation_credentials(tmp_path):
    @dataclass(frozen=True)
    class Resources:
        nodes: int = 1

    @dataclass(frozen=True)
    class Plan:
        resources: Resources
        additional_node_command: tuple
        study_secret_env: dict
        exe_module: str = "legacy.module"
        module_args: tuple = ()
        node_app_dir: str = "app"

    class Manager:
        def launch(self, plan):
            self.plan = plan
            return SimpleNamespace(
                terminate=lambda: None,
                poll=lambda: 0,
                wait=lambda: None,
            )

    manager = Manager()
    job_launcher = SimpleNamespace(
        manager=manager,
        _build_launch_plan=lambda job_meta, fl_ctx: Plan(
            resources=Resources(),
            additional_node_command=(),
            study_secret_env={
                JobProcessEnv.AUTH_TOKEN: "secret",
                JobProcessEnv.TOKEN_SIGNATURE: "signature",
                JobProcessEnv.SSID: "ssid",
                "SAFE": "value",
            },
        ),
    )
    launcher = SlurmTaskWorkerLauncher(job_launcher, {"job_id": "job"}, SimpleNamespace())
    attempt_dir = str(tmp_path.resolve())
    handle = launcher.launch(
        [sys.executable, "-m", "nvflare.private.fed.deployment_supervisor.worker", attempt_dir],
        cwd=attempt_dir,
        env={},
    )
    assert isinstance(handle, SlurmTaskWorkerHandle)
    assert handle.wait() == 0
    assert manager.plan.exe_module == "nvflare.private.fed.deployment_supervisor.worker"
    assert manager.plan.module_args == (attempt_dir,)
    assert manager.plan.study_secret_env == {"SAFE": "value"}
    assert manager.plan.node_app_dir is None
