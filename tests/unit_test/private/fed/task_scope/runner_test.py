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

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import ReservedHeaderKey, Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.fuel.utils.fobs.decomposers.via_downloader import LazyDownloadRef
from nvflare.private.defs import SpecialTaskName
from nvflare.private.fed.client.client_app_runner import ClientAppRunner
from nvflare.private.fed.client.client_engine_executor_spec import TaskAssignment
from nvflare.private.fed.client.client_runner import ClientRunner, TaskRouter
from nvflare.private.fed.task_scope import worker
from nvflare.private.fed.task_scope.protocol import (
    ATTEMPT_OPTION,
    DIRECTORY_OPTION,
    END_RUN,
    IDLE,
    RECEIPT_FILE,
    TASK_COMPLETE,
    read_receipt,
)
from nvflare.private.fed.task_scope.runner import TaskScopedClientAppRunner, TaskScopedClientRunner


def _task(name="train", task_id="task-1"):
    return TaskAssignment(name, task_id, Shareable())


@pytest.fixture
def runner():
    result = TaskScopedClientRunner.__new__(TaskScopedClientRunner)
    result.task_router = TaskRouter()
    result.task_router.add_executor(["train"], SimpleNamespace(supports_task_scoped_process=True))
    result.engine = MagicMock()
    result.engine.new_context.return_value.__enter__.return_value = FLContext()
    result.engine.get_task_assignment.return_value = _task()
    result.run_abort_signal = Signal()
    result._run_abort_requested = False
    result.task_lock = threading.Lock()
    result.running_tasks = {}
    result.get_task_timeout = 5.0
    result.log_debug = MagicMock()
    result.log_info = MagicMock()
    result.log_error = MagicMock()
    result.log_exception = MagicMock()
    result.fire_event = MagicMock()
    result.init_run = MagicMock()
    result.end_run_events_sequence = MagicMock()
    result._send_job_heartbeat = MagicMock()
    result._process_task = MagicMock(return_value=Shareable())
    result._send_task_result = MagicMock(return_value=True)
    with (
        patch("nvflare.private.fed.client.client_runner.ReliableMessage.shutdown"),
        patch("nvflare.private.fed.client.client_runner.DownloadService.shutdown"),
    ):
        yield result


def test_one_task_waits_for_ack_and_end_run_before_exposing_outcome(runner):
    args = SimpleNamespace()
    order = []

    def send_result(*_args):
        assert args.task_scope_outcome is None
        order.append("ack")
        return True

    def end_run():
        assert args.task_scope_outcome is None
        order.append("end_run")

    runner._send_task_result.side_effect = send_result
    runner.end_run_events_sequence.side_effect = end_run
    runner.run("app", args)

    assert args.task_scope_outcome == {"status": TASK_COMPLETE, "task_id": "task-1"}
    assert order == ["ack", "end_run"]
    runner.engine.get_task_assignment.assert_called_once()
    runner._process_task.assert_called_once()
    runner._send_task_result.assert_called_once()
    runner.engine.shutdown_streamer.assert_called_once()
    assert runner.run_abort_signal.triggered


@pytest.mark.parametrize("name, status", [(SpecialTaskName.TRY_AGAIN, IDLE), (SpecialTaskName.END_RUN, END_RUN)])
def test_explicit_control_response_exits_without_executing_or_submitting(runner, name, status):
    args = SimpleNamespace()
    runner.engine.get_task_assignment.return_value = _task(name)

    runner.run("app", args)

    assert args.task_scope_outcome == {"status": status, "task_id": None}
    runner.engine.get_task_assignment.assert_called_once()
    runner._process_task.assert_not_called()
    runner._send_task_result.assert_not_called()
    runner.end_run_events_sequence.assert_called_once()


def test_missing_assignment_is_a_failure_not_idle(runner):
    args = SimpleNamespace()
    runner.engine.get_task_assignment.return_value = None

    with pytest.raises(RuntimeError, match="client execution failed") as error:
        runner.run("app", args)

    assert "explicit control response" in str(error.value.__cause__)
    assert args.task_scope_outcome is None
    runner._process_task.assert_not_called()


@pytest.mark.parametrize("submitted", [False, None])
def test_unacknowledged_result_never_publishes_success(runner, submitted):
    args = SimpleNamespace()
    runner._send_task_result.return_value = submitted

    with pytest.raises(RuntimeError, match="client execution failed") as error:
        runner.run("app", args)

    assert "result-submission ACK" in str(error.value.__cause__)
    assert args.task_scope_outcome is None
    runner.end_run_events_sequence.assert_called_once()


@pytest.mark.parametrize(
    "return_code", [ReturnCode.EXECUTION_EXCEPTION, ReturnCode.TASK_ABORTED, ReturnCode.UNSAFE_JOB]
)
def test_acknowledged_failure_result_is_not_a_completed_task(runner, return_code):
    args = SimpleNamespace()
    runner._process_task.return_value = make_reply(return_code)

    with pytest.raises(RuntimeError, match="client execution failed") as error:
        runner.run("app", args)

    assert "failure code" in str(error.value.__cause__)
    runner._send_task_result.assert_called_once()
    assert args.task_scope_outcome is None


def test_processing_exception_is_not_swallowed_into_clean_outcome(runner):
    args = SimpleNamespace()
    failure = RuntimeError("executor could not restore its artifact")
    runner._process_task.side_effect = failure

    with pytest.raises(RuntimeError, match="client execution failed") as error:
        runner.run("app", args)

    assert error.value.__cause__ is failure
    runner.log_exception.assert_called_once()
    runner.end_run_events_sequence.assert_called_once()
    assert args.task_scope_outcome is None


def test_fatal_event_after_send_invalidates_success(runner):
    args = SimpleNamespace()

    def fire_event(event, fl_ctx):
        if event == EventType.AFTER_SEND_TASK_RESULT:
            runner.handle_event(EventType.FATAL_SYSTEM_ERROR, fl_ctx)

    runner.fire_event.side_effect = fire_event
    with pytest.raises(RuntimeError, match="client execution failed"):
        runner.run("app", args)

    assert runner._run_abort_requested
    assert args.task_scope_outcome is None


@pytest.mark.parametrize("boundary", ["ack", "after_send", "end_run"])
def test_normal_server_end_run_after_ack_preserves_completed_task(runner, boundary):
    args = SimpleNamespace()

    def server_end_run():
        runner._handle_end_run("end_run", Shareable(), FLContext())

    if boundary == "ack":

        def send_result(*_args):
            server_end_run()
            return True

        runner._send_task_result.side_effect = send_result
    elif boundary == "after_send":

        def fire_event(event, _fl_ctx):
            if event == EventType.AFTER_SEND_TASK_RESULT:
                server_end_run()

        runner.fire_event.side_effect = fire_event
    else:
        runner.end_run_events_sequence.side_effect = server_end_run

    runner.run("app", args)

    assert args.task_scope_outcome == {"status": TASK_COMPLETE, "task_id": "task-1"}
    assert not runner._run_abort_requested


def test_abort_during_end_run_invalidates_success(runner):
    args = SimpleNamespace()
    runner.end_run_events_sequence.side_effect = lambda: runner.abort("abort during teardown")

    with pytest.raises(RuntimeError, match="aborted without a clean task outcome"):
        runner.run("app", args)

    assert args.task_scope_outcome is None


def test_end_run_failure_does_not_expose_an_outcome(runner):
    args = SimpleNamespace()
    runner.end_run_events_sequence.side_effect = RuntimeError("end-run checkpoint failed")

    with pytest.raises(RuntimeError, match="checkpoint failed"):
        runner.run("app", args)

    assert args.task_scope_outcome is None


@pytest.mark.parametrize("wrapped", [False, True])
def test_lazy_result_is_rejected_before_send(runner, wrapped):
    args = SimpleNamespace()
    lazy = LazyDownloadRef("site-1.job-1", "download-1", "tensor-1")
    value = SimpleNamespace(nested=(lazy,)) if wrapped else {"weights": [lazy]}
    runner._process_task.return_value = Shareable({"result": value})

    with pytest.raises(RuntimeError, match="client execution failed") as error:
        runner.run("app", args)

    assert "requires eager task results" in str(error.value.__cause__)
    runner._send_task_result.assert_not_called()
    assert args.task_scope_outcome is None


def test_pass_through_result_is_rejected(runner):
    args = SimpleNamespace()
    runner._process_task.return_value.set_header(ReservedHeaderKey.PASS_THROUGH, True)

    with pytest.raises(RuntimeError, match="client execution failed"):
        runner.run("app", args)

    runner._send_task_result.assert_not_called()
    assert args.task_scope_outcome is None


@pytest.mark.parametrize("executor", [object(), SimpleNamespace(supports_task_scoped_process="true")])
def test_every_executor_must_explicitly_declare_restart_safety(runner, executor):
    args = SimpleNamespace()
    runner.task_router.add_executor(["validate*"], executor)

    with pytest.raises(RuntimeError, match="supports_task_scoped_process=True"):
        runner.run("app", args)

    runner.init_run.assert_not_called()
    runner.engine.get_task_assignment.assert_not_called()
    assert args.task_scope_outcome is None


def test_aux_task_is_rejected_without_invoking_executor(runner):
    result = runner._handle_do_task("do_task", Shareable(), FLContext())

    assert result.get_return_code() == ReturnCode.TASK_UNKNOWN
    runner._process_task.assert_not_called()


def test_default_client_runner_selection_is_unchanged():
    assert ClientAppRunner.CLIENT_RUNNER_CLASS is ClientRunner
    assert TaskScopedClientAppRunner.CLIENT_RUNNER_CLASS is TaskScopedClientRunner


@pytest.fixture
def attempt_dir(tmp_path):
    attempt = "a" * 32
    directory = tmp_path / ".task_scope" / attempt
    directory.mkdir(parents=True)
    return directory, attempt


def _attempt_args(directory, attempt, **values):
    return SimpleNamespace(
        set=[f"{ATTEMPT_OPTION}={attempt}", f"{DIRECTORY_OPTION}={directory}"],
        **values,
    )


def test_worker_receipt_follows_standard_main_finally_cleanup(attempt_dir, monkeypatch):
    directory, attempt = attempt_dir
    args = _attempt_args(directory, attempt)
    order = []

    def run_worker(received_args, app_runner_class):
        assert received_args is args
        assert app_runner_class is TaskScopedClientAppRunner
        try:
            args.task_scope_outcome = {"status": TASK_COMPLETE, "task_id": "task-1"}
            order.append("run")
        finally:
            assert not (directory / RECEIPT_FILE).exists()
            order.append("cleanup")

    monkeypatch.setattr(worker, "run_worker", run_worker)
    assert worker.main(args) == 0

    assert order == ["run", "cleanup"]
    assert read_receipt(str(directory), attempt) == {
        "attempt": attempt,
        "status": TASK_COMPLETE,
        "task_id": "task-1",
    }


def test_worker_cleanup_failure_never_writes_receipt(attempt_dir, monkeypatch):
    directory, attempt = attempt_dir

    def run_worker(args, app_runner_class):
        try:
            args.task_scope_outcome = {"status": TASK_COMPLETE, "task_id": "task-1"}
        finally:
            raise RuntimeError("archive upload failed")

    monkeypatch.setattr(worker, "run_worker", run_worker)
    with pytest.raises(RuntimeError, match="archive upload failed"):
        worker.main(_attempt_args(directory, attempt))

    assert not (directory / RECEIPT_FILE).exists()


def test_worker_does_not_reuse_stale_args_outcome(attempt_dir, monkeypatch):
    directory, attempt = attempt_dir
    monkeypatch.setattr(worker, "run_worker", MagicMock())
    args = _attempt_args(directory, attempt, task_scope_outcome={"status": TASK_COMPLETE, "task_id": "old-task"})

    with pytest.raises(RuntimeError, match="without a clean runner outcome"):
        worker.main(args)

    assert not (directory / RECEIPT_FILE).exists()


def test_worker_requires_attempt_environment_before_startup(monkeypatch):
    run_worker = MagicMock()
    monkeypatch.setattr(worker, "run_worker", run_worker)

    with pytest.raises(RuntimeError, match="requires an attempt ID"):
        worker.main(SimpleNamespace(set=[]))

    run_worker.assert_not_called()
