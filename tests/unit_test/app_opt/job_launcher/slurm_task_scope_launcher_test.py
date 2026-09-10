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
"""Control-state tests with a fake scheduler; these do not execute Slurm."""

import json
import shlex
import threading
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import pytest

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey, ReservedKey, ReturnCode
from nvflare.apis.fl_context import FLContext, FLContextManager
from nvflare.apis.job_launcher_spec import JobProcessArgs, JobReturnCode
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.app_opt.job_launcher.slurm.config import JobResources, LaunchPlan, SlurmLauncherError
from nvflare.app_opt.job_launcher.slurm.launcher import ClientSlurmJobLauncher, SlurmJobLauncher
from nvflare.fuel.common.exit_codes import ProcessExitCode
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey
from nvflare.fuel.f3.cellnet.defs import ReturnCode as CellReturnCode
from nvflare.private.defs import CellChannel, new_cell_message
from nvflare.private.fed.task_scope import launcher as launcher_module
from nvflare.private.fed.task_scope.launcher import TaskScopedJobHandle
from nvflare.private.fed.task_scope.protocol import (
    ATTEMPT_OPTION,
    DIRECTORY_OPTION,
    DONE,
    END_RUN,
    ERROR,
    IDLE,
    PROBE_TOPIC,
    READY,
    STATUS,
    TASK_COMPLETE,
    TASK_TOKEN,
    TERMINAL_TOPIC,
    WAIT,
    write_receipt,
)


def _plan(tmp_path):
    run_dir = tmp_path / "job-1"
    run_dir.mkdir()
    return LaunchPlan(
        job_id="job-1",
        site_name="site-1",
        run_dir=str(run_dir),
        exe_module="nvflare.private.fed.app.client.worker_process",
        module_args=(),
        resources=JobResources(gpus_per_node=1),
        directives={},
        sandbox="none",
        image=None,
        setup="",
        study_env={"EXISTING": "preserved"},
        study_secret_env={},
        mounts=(),
        python_path="/usr/bin/python3",
        python_env="",
        forward_env=(),
    )


class _FakeAllocation:
    def __init__(self, manager, plan, spec):
        self.manager = manager
        self.plan = plan
        self.spec = spec
        self.job_id = str(1000 + len(manager.allocations))
        self.finished = False
        self.entered_wait = threading.Event()
        self.release = threading.Event()
        self.terminations = 0

    def terminate(self):
        self.terminations += 1
        self.release.set()

    def wait(self):
        if self.finished:
            return
        self.entered_wait.set()
        self.manager.trace.append(("wait", self.job_id))
        if self.spec.get("block"):
            assert self.release.wait(3), "test did not release its fake allocation"
        if self.spec.get("on_wait"):
            self.spec["on_wait"](self)
        receipt = self.spec.get("receipt", {STATUS: TASK_COMPLETE, "task_id": f"task-{self.job_id}"})
        if receipt is not None:
            attempt = self.plan.study_env[ATTEMPT_OPTION]
            if self.spec.get("stale_receipt"):
                attempt = "previous-attempt"
            write_receipt(self.plan.study_env[DIRECTORY_OPTION], attempt, receipt)
        if "legacy_rc" in self.spec:
            Path(self.plan.run_dir, "_process_rc.txt").write_text(str(self.spec["legacy_rc"]))
        self.finished = True
        self.manager.trace.append(("released", self.job_id))

    def poll(self):
        return self.spec.get("rc", JobReturnCode.SUCCESS) if self.finished else JobReturnCode.UNKNOWN


class _FakeSlurmManager:
    """Require every physical allocation to finish before another is submitted."""

    def __init__(self, specs=()):
        self.specs = list(specs)
        self.plans = []
        self.allocations = []
        self.trace = []
        self.logger = Mock()
        self.on_launch = None

    def launch(self, plan):
        assert all(a.finished for a in self.allocations), "overlapping Slurm allocations"
        self.plans.append(plan)
        spec = self.specs.pop(0)
        if spec.get("launch_error"):
            raise SlurmLauncherError("fake sbatch failure")
        allocation = _FakeAllocation(self, plan, spec)
        self.allocations.append(allocation)
        self.trace.append(("launch", allocation.job_id))
        if self.on_launch:
            self.on_launch(allocation)
        return allocation


def _reply(status, task_token="pending-task"):
    result = make_reply(ReturnCode.OK)
    result[STATUS] = status
    if status == READY:
        result[TASK_TOKEN] = task_token
    return result


def _handle(tmp_path, statuses=(DONE,), specs=()):
    manager = _FakeSlurmManager(specs)
    plan = _plan(tmp_path)
    replies = []
    for index, status in enumerate(statuses):
        if status is None or isinstance(status, Shareable):
            replies.append(status)
        else:
            replies.append(_reply(status, f"pending-task-{index}"))
    probe = Mock(side_effect=replies)

    def launch_attempt(attempt, directory):
        attempt_plan = replace(
            plan, study_env=dict(plan.study_env, **{ATTEMPT_OPTION: attempt, DIRECTORY_OPTION: directory})
        )
        return manager.launch(attempt_plan)

    handle = TaskScopedJobHandle(
        plan.job_id,
        plan.run_dir,
        probe,
        manager.logger,
        poll_interval=0.001,
        communication_timeout=0.05,
        launch_attempt=launch_attempt,
        allocation_details=lambda allocation: {"slurm_id": allocation.job_id},
    )
    handle.plan = plan
    return handle, manager, probe


def _events(handle):
    return [
        json.loads(line) for line in Path(handle.plan.run_dir, ".task_scope", "events.jsonl").read_text().splitlines()
    ]


def _start(handle):
    thread = threading.Thread(target=handle.wait, daemon=True)
    thread.start()
    return thread


def _join(thread):
    thread.join(3)
    assert not thread.is_alive(), "logical handle did not settle"


def test_idle_probes_keep_gpu_unallocated(tmp_path):
    handle, manager, probe = _handle(tmp_path, [WAIT, WAIT, WAIT, DONE])
    handle.wait()
    assert handle.poll() == JobReturnCode.SUCCESS
    assert manager.plans == []
    assert probe.call_count == 4


def test_two_tasks_use_distinct_sequential_allocations_and_matching_receipts(tmp_path):
    handle, manager, _ = _handle(tmp_path, [WAIT, READY, WAIT, READY, WAIT, DONE], [{}, {}])
    handle.wait()
    assert handle.poll() == JobReturnCode.SUCCESS
    assert len(manager.allocations) == 2
    assert [p.job_id for p in manager.plans] == ["job-1", "job-1"]
    assert len({p.study_env[ATTEMPT_OPTION] for p in manager.plans}) == 2
    assert len({p.study_env[DIRECTORY_OPTION] for p in manager.plans}) == 2
    assert all(p.study_env["EXISTING"] == "preserved" for p in manager.plans)
    assert manager.trace == [
        ("launch", "1000"),
        ("wait", "1000"),
        ("released", "1000"),
        ("launch", "1001"),
        ("wait", "1001"),
        ("released", "1001"),
    ]
    phases = [event["phase"] for event in _events(handle)]
    assert phases.index("allocation_released") < phases.index("receipt")


def test_receipt_is_read_only_after_scheduler_wait_confirms_allocation_release(tmp_path, monkeypatch):
    handle, manager, _ = _handle(tmp_path, [READY, DONE], [{}])
    real_read = launcher_module.read_receipt

    def checked_read(directory, attempt):
        assert all(a.finished for a in manager.allocations)
        return real_read(directory, attempt)

    monkeypatch.setattr(launcher_module, "read_receipt", checked_read)
    handle.wait()
    assert handle.poll() == JobReturnCode.SUCCESS


@pytest.mark.parametrize("receipt", [None, {STATUS: TASK_COMPLETE, "task_id": "t"}])
def test_missing_or_stale_receipt_never_recycles_worker(tmp_path, receipt):
    handle, manager, probe = _handle(tmp_path, [READY, READY, DONE], [{"receipt": receipt, "stale_receipt": True}, {}])
    handle.wait()
    assert handle.poll() == ProcessExitCode.INFRASTRUCTURE_ERROR
    assert len(manager.plans) == 1
    assert probe.call_count == 1


@pytest.mark.parametrize("code", [1, 137, ProcessExitCode.EXCEPTION, ProcessExitCode.INFRASTRUCTURE_ERROR])
def test_failed_allocation_never_restarts_even_with_valid_receipt(tmp_path, code):
    handle, manager, probe = _handle(tmp_path, [READY, READY, DONE], [{"rc": code}, {}])
    handle.wait()
    assert handle.poll() in (ProcessExitCode.EXCEPTION, ProcessExitCode.INFRASTRUCTURE_ERROR)
    assert len(manager.plans) == 1
    assert probe.call_count == 1


def test_stale_legacy_success_cannot_mask_sbatch_failure(tmp_path):
    handle, manager, _ = _handle(tmp_path, [READY], [{"launch_error": True}])
    Path(handle.plan.run_dir, "_process_rc.txt").write_text("0")
    handle.wait()
    assert handle.poll() == ProcessExitCode.INFRASTRUCTURE_ERROR
    assert not Path(handle.plan.run_dir, "_process_rc.txt").exists()
    archives = list(Path(handle.plan.run_dir, ".task_scope").glob("*/previous_process_rc.txt"))
    assert len(archives) == 1
    assert archives[0].read_text() == "0"
    assert len(manager.plans) == 1


def test_current_legacy_success_cannot_mask_scheduler_failure(tmp_path):
    handle, _, _ = _handle(tmp_path, [READY], [{"rc": 1, "legacy_rc": 0}])
    handle.wait()
    assert handle.poll() == ProcessExitCode.EXCEPTION


def test_cancel_while_idle_releases_logical_job_without_sbatch(tmp_path):
    handle, manager, probe = _handle(tmp_path)

    def cancel_on_probe():
        handle.terminate()
        return _reply(WAIT)

    probe.side_effect = cancel_on_probe
    handle.wait()
    assert handle.poll() == JobReturnCode.ABORTED
    assert not manager.plans


def test_cancel_during_submission_terminates_new_handle_and_waits_for_release(tmp_path):
    handle, manager, _ = _handle(tmp_path, [READY, DONE], [{"receipt": None}])
    manager.on_launch = lambda allocation: handle.terminate()
    handle.wait()
    assert handle.poll() == JobReturnCode.ABORTED
    assert len(manager.allocations) == 1
    allocation = manager.allocations[0]
    assert allocation.terminations == 1
    assert allocation.finished


def test_cancel_active_worker_waits_until_allocation_is_released(tmp_path):
    handle, manager, _ = _handle(tmp_path, [READY, DONE], [{"block": True, "receipt": None}])
    launched = threading.Event()
    manager.on_launch = lambda allocation: launched.set()
    thread = _start(handle)
    assert launched.wait(1)
    allocation = manager.allocations[0]
    assert allocation.entered_wait.wait(1)
    handle.terminate()
    _join(thread)
    assert allocation.finished
    assert allocation.terminations >= 1
    assert handle.poll() == JobReturnCode.ABORTED


@pytest.mark.parametrize("receipt,expected", [({STATUS: TASK_COMPLETE, "task_id": "t"}, 0), (None, 104)])
def test_terminal_notice_during_active_allocation_still_requires_exit_and_receipt(tmp_path, receipt, expected):
    handle, manager, probe = _handle(tmp_path, [READY], [{"block": True, "receipt": receipt}])
    launched = threading.Event()
    manager.on_launch = lambda allocation: launched.set()
    thread = _start(handle)
    assert launched.wait(1)
    allocation = manager.allocations[0]
    assert allocation.entered_wait.wait(1)
    handle.notify_terminal(DONE)
    assert handle.poll() == JobReturnCode.UNKNOWN
    assert allocation.terminations == 0
    allocation.release.set()
    _join(thread)
    assert handle.poll() == expected
    assert allocation.finished
    assert probe.call_count == 1


def test_probe_loss_fails_without_interpreting_server_disappearance_as_success(tmp_path):
    handle, manager, probe = _handle(tmp_path)
    probe.side_effect = None
    probe.return_value = None
    handle.communication_timeout = 0.005
    handle.wait()
    assert handle.poll() == ProcessExitCode.INFRASTRUCTURE_ERROR
    assert not manager.plans


@pytest.mark.parametrize("reply", [{}, {STATUS: "unknown"}, [READY], _reply(ERROR)])
def test_invalid_or_rejected_probe_fails_without_allocating(tmp_path, reply):
    handle, manager, probe = _handle(tmp_path)
    probe.side_effect = None
    probe.return_value = reply
    handle.wait()
    assert handle.poll() == ProcessExitCode.INFRASTRUCTURE_ERROR
    assert not manager.plans


def test_error_notification_cancels_active_allocation_and_preserves_failure(tmp_path):
    handle, manager, _ = _handle(tmp_path, [READY], [{"block": True, "receipt": None}])
    launched = threading.Event()
    manager.on_launch = lambda allocation: launched.set()
    thread = _start(handle)
    assert launched.wait(1)
    allocation = manager.allocations[0]
    assert allocation.entered_wait.wait(1)
    handle.notify_terminal(ERROR)
    _join(thread)
    assert allocation.finished
    assert allocation.terminations >= 1
    assert handle.poll() not in (JobReturnCode.SUCCESS, JobReturnCode.UNKNOWN)


def test_evidence_write_failure_after_submit_still_reclaims_allocation(tmp_path):
    handle, manager, _ = _handle(tmp_path, [READY], [{"receipt": None}])
    real_record = handle._record

    def fail_allocation_record(phase, **details):
        if phase == "allocated":
            raise OSError("fake evidence write failure")
        real_record(phase, **details)

    handle._record = fail_allocation_record
    handle.wait()
    assert handle.poll() == ProcessExitCode.INFRASTRUCTURE_ERROR
    assert len(manager.allocations) == 1
    assert manager.allocations[0].finished
    assert manager.allocations[0].terminations == 1


def test_wait_is_idempotent_after_logical_completion(tmp_path):
    handle, manager, probe = _handle(tmp_path, [READY, DONE], [{}])
    handle.wait()
    handle.wait()
    assert handle.poll() == JobReturnCode.SUCCESS
    assert probe.call_count == 2
    assert len(manager.plans) == 1


def test_explicit_end_run_receipt_finishes_without_another_probe(tmp_path):
    handle, manager, probe = _handle(tmp_path, [READY], [{"receipt": {STATUS: END_RUN}}])
    handle.wait()
    assert handle.poll() == JobReturnCode.SUCCESS
    assert len(manager.plans) == 1
    assert probe.call_count == 1


def test_done_racing_ready_probe_does_not_submit_new_allocation(tmp_path):
    handle, manager, probe = _handle(tmp_path, specs=[{}])

    def ending_probe():
        handle.notify_terminal(DONE)
        return _reply(READY)

    probe.side_effect = ending_probe
    handle.wait()
    assert handle.poll() == JobReturnCode.SUCCESS
    assert not manager.plans


def test_idle_receipt_suppresses_same_task_token_until_new_work_is_advertised(tmp_path):
    handle, manager, probe = _handle(
        tmp_path,
        [
            _reply(READY, "stale-work"),
            WAIT,
            _reply(READY, "stale-work"),
            _reply(READY, "stale-work"),
            _reply(READY, "new-work"),
            DONE,
        ],
        [{"receipt": {STATUS: IDLE}}, {}],
    )
    handle.wait()
    assert handle.poll() == JobReturnCode.SUCCESS
    assert len(manager.plans) == 2
    assert all(a.finished for a in manager.allocations)
    assert probe.call_count == 6


def _launcher(tmp_path, task_scoped=True):
    return ClientSlurmJobLauncher(
        workspace_path=str(tmp_path),
        sandbox="none",
        python_path="/usr/bin/python3",
        executables={name: "/usr/bin/true" for name in ("sbatch", "squeue", "sacct", "scancel")},
        task_scoped=task_scoped,
    )


def _launcher_context():
    engine = Mock()

    def context():
        fl_ctx = FLContext()
        fl_ctx.set_prop(ReservedKey.ENGINE, engine, private=True, sticky=False)
        fl_ctx.set_prop(ReservedKey.IDENTITY_NAME, "site-1", private=False, sticky=False)
        return fl_ctx

    engine.new_context.side_effect = context
    return engine, context()


def test_launcher_registers_parent_terminal_endpoint_and_probes_explicit_server_job(tmp_path):
    launcher = _launcher(tmp_path)
    plan = _plan(tmp_path)
    launcher._build_launch_plan = Mock(return_value=plan)
    launcher.manager = _FakeSlurmManager()
    engine, fl_ctx = _launcher_context()
    parent_contexts = FLContextManager(
        engine=engine,
        identity_name="site-1",
        job_id="",
        public_stickers={FLContextKey.CURRENT_JOB_ID: ""},
    )
    engine.new_context.side_effect = parent_contexts.new_context
    engine.aux_runner.send_aux_request.return_value = {"server": _reply(DONE)}
    handle = launcher.launch_job({}, fl_ctx)
    assert not launcher.manager.plans
    engine.get_cell.return_value.register_request_cb.assert_called_once_with(
        channel=CellChannel.AUX_COMMUNICATION, topic=TERMINAL_TOPIC, cb=launcher._task_scope._terminal_message
    )
    handle.wait()
    assert handle.poll() == JobReturnCode.SUCCESS
    kwargs = engine.aux_runner.send_aux_request.call_args.kwargs
    assert kwargs["targets"][0].fqcn == "server.job-1"
    assert kwargs["targets"][0].job_scoped is False
    assert kwargs["topic"] == PROBE_TOPIC
    assert kwargs["fl_ctx"].get_prop(FLContextKey.CURRENT_RUN) == "job-1"
    assert kwargs["fl_ctx"].get_prop(FLContextKey.CURRENT_JOB_ID) == "job-1"
    # Probe-local overrides must not turn the deployment-scoped CP context into
    # a context for this job; other logical jobs may probe concurrently.
    assert parent_contexts.new_context().get_prop(FLContextKey.CURRENT_RUN) == ""
    assert parent_contexts.new_context().get_prop(FLContextKey.CURRENT_JOB_ID) == ""


def test_task_scoped_mode_reuses_physical_slurm_launch_and_restores_common_bootstrap(tmp_path, monkeypatch):
    launcher = _launcher(tmp_path)
    plan = _plan(tmp_path)
    launcher._build_launch_plan = Mock(return_value=plan)
    launcher.manager = Mock()
    engine, fl_ctx = _launcher_context()
    original_args = {
        JobProcessArgs.EXE_MODULE: ("-m", ClientSlurmJobLauncher.EXE_MODULE),
        JobProcessArgs.OPTIONS: ("--set", "existing=value"),
    }
    fl_ctx.set_prop(FLContextKey.JOB_PROCESS_ARGS, original_args, private=True, sticky=False)
    engine.aux_runner.send_aux_request.side_effect = [
        {"server": _reply(READY, "task-1")},
        {"server": _reply(DONE)},
    ]
    launches = []

    class Allocation:
        job_id = "1234"

        def wait(self):
            options = dict(token.split("=", 1) for token in shlex.split(launches[0][JobProcessArgs.OPTIONS][1]))
            write_receipt(
                options[DIRECTORY_OPTION],
                options[ATTEMPT_OPTION],
                {STATUS: TASK_COMPLETE, "task_id": "task-1"},
            )

        def poll(self):
            return JobReturnCode.SUCCESS

        def terminate(self):
            raise AssertionError("successful physical allocation must not be terminated")

    def physical_launch(_launcher, _job_meta, received_ctx):
        launches.append(dict(received_ctx.get_prop(FLContextKey.JOB_PROCESS_ARGS)))
        return Allocation()

    monkeypatch.setattr(SlurmJobLauncher, "launch_job", physical_launch)
    handle = launcher.launch_job({}, fl_ctx)
    handle.wait()

    assert handle.poll() == JobReturnCode.SUCCESS
    assert len(launches) == 1
    assert launches[0][JobProcessArgs.EXE_MODULE][1] == ClientSlurmJobLauncher.EXE_MODULE
    assert fl_ctx.get_prop(FLContextKey.JOB_PROCESS_ARGS) is original_args


def test_launcher_rejects_duplicate_logical_job_without_submitting(tmp_path):
    launcher = _launcher(tmp_path)
    launcher._build_launch_plan = Mock(return_value=_plan(tmp_path))
    launcher.manager = _FakeSlurmManager()
    _, fl_ctx = _launcher_context()
    launcher.launch_job({}, fl_ctx)
    with pytest.raises(RuntimeError, match="already active"):
        launcher.launch_job({}, fl_ctx)
    assert not launcher.manager.plans


def test_launcher_rejects_multinode_before_creating_logical_job(tmp_path):
    launcher = _launcher(tmp_path)
    launcher._build_launch_plan = Mock(return_value=replace(_plan(tmp_path), resources=JobResources(nodes=2)))
    _, fl_ctx = _launcher_context()
    with pytest.raises(SlurmLauncherError, match="single-node"):
        launcher.launch_job({}, fl_ctx)


@pytest.mark.parametrize(
    "origin,job,status,expected",
    [
        ("server.job-1", "job-1", DONE, CellReturnCode.OK),
        ("site-2", "job-1", DONE, CellReturnCode.AUTHENTICATION_ERROR),
        ("server.other", "other", DONE, CellReturnCode.AUTHENTICATION_ERROR),
        ("server.job-1", "job-1", READY, CellReturnCode.INVALID_REQUEST),
    ],
)
def test_terminal_endpoint_validates_server_origin_job_and_status(tmp_path, origin, job, status, expected):
    launcher = _launcher(tmp_path)
    handle = Mock()
    launcher._task_scope._handles["job-1"] = handle
    message = new_cell_message({MessageHeaderKey.ORIGIN: origin}, Shareable({"job_id": job, STATUS: status}))
    reply = launcher._task_scope._terminal_message(message)
    assert reply.get_header(MessageHeaderKey.RETURN_CODE) == expected
    if expected == CellReturnCode.OK:
        handle.notify_terminal.assert_called_once_with(DONE)
    else:
        handle.notify_terminal.assert_not_called()


def test_default_client_slurm_launcher_still_submits_immediately(tmp_path):
    launcher = _launcher(tmp_path, task_scoped=False)
    plan = _plan(tmp_path)
    launcher._build_launch_plan = Mock(return_value=plan)
    launcher.manager = Mock()
    _, fl_ctx = _launcher_context()
    assert launcher.launch_job({}, fl_ctx) is launcher.manager.launch.return_value
    launcher.manager.launch.assert_called_once_with(plan)
    assert launcher._task_scope is None


@pytest.mark.parametrize("result", [JobReturnCode.SUCCESS, JobReturnCode.UNKNOWN])
def test_job_completed_releases_only_finished_logical_handle(tmp_path, result):
    launcher = _launcher(tmp_path)
    handle = Mock()
    handle.poll.return_value = result
    launcher._task_scope._handles["job-1"] = handle
    _, ctx = _launcher_context()
    ctx.set_prop(FLContextKey.CURRENT_JOB_ID, "job-1", private=True, sticky=False)
    launcher.handle_event(EventType.JOB_COMPLETED, ctx)
    assert ("job-1" in launcher._task_scope._handles) == (result == JobReturnCode.UNKNOWN)
