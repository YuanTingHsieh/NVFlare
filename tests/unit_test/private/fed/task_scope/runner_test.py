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
from nvflare.apis.fl_constant import FilterKey, FLContextKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import ReservedHeaderKey, Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.fuel.utils.fobs.decomposers.via_downloader import LazyDownloadRef
from nvflare.private.defs import SpecialTaskName
from nvflare.private.fed.app import job_process_cleanup
from nvflare.private.fed.app.client import worker_process as worker
from nvflare.private.fed.client.client_app_runner import ClientAppRunner
from nvflare.private.fed.client.client_engine_executor_spec import TaskAssignment
from nvflare.private.fed.client.client_runner import ClientRunner, TaskRouter
from nvflare.private.fed.task_scope import runner as runner_module
from nvflare.private.fed.task_scope.artifacts import read_artifact, write_artifact
from nvflare.private.fed.task_scope.protocol import (
    ATTEMPT_OPTION,
    COMPUTE,
    DIRECTORY_OPTION,
    END_RUN,
    IDLE,
    INPUT_READY,
    PHASE_OPTION,
    PULL,
    PUSH,
    RECEIPT_FILE,
    RESULT_READY,
    TASK_COMPLETE,
    read_receipt,
)
from nvflare.private.fed.task_scope.runner import TaskScopedClientAppRunner, TaskScopedClientRunner
from nvflare.private.fed.utils.fed_utils import fobs_initialize


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


@pytest.fixture
def worker_runtime(tmp_path, attempt_dir, monkeypatch):
    """Execute canonical worker startup/finally with external runtime services stubbed."""
    (tmp_path / "startup").mkdir()
    (tmp_path / "local").mkdir()
    args = SimpleNamespace(
        workspace=str(tmp_path),
        job_id="job-1",
        client_name="site-1",
        token="token",
        token_signature="signature",
        ssid="session",
        set=[],
    )
    directory, attempt = attempt_dir
    runtime = SimpleNamespace(
        args=args,
        directory=directory,
        attempt=attempt,
        order=[],
        outcome={"status": TASK_COMPLETE, "task_id": "task-1"},
    )

    def record(stage):
        # The real main must not publish until even the final process cleanup.
        assert not list(directory.rglob(RECEIPT_FILE))
        runtime.order.append(stage)

    for name in (
        "download_workspace",
        "refresh_custom_dir_import_path",
        "set_stats_pool_config_for_job",
        "fobs_initialize",
        "security_init_for_job",
        "register_ext_decomposers",
        "configure_logging",
    ):
        monkeypatch.setattr(worker, name, MagicMock())
    monkeypatch.setattr(worker.ConfigService, "get_str_var", MagicMock(return_value=None))
    monkeypatch.setattr(worker, "get_script_logger", MagicMock(return_value=MagicMock()))

    federated_client = MagicMock()
    federated_client.stop_cell.side_effect = lambda: record("cell")
    federated_client.terminate.side_effect = lambda: record("client")
    runtime.deployer = MagicMock()
    runtime.deployer.create_fed_client.return_value = federated_client
    runtime.deployer.close.side_effect = lambda: record("deployer")
    conf = MagicMock()
    conf.base_deployer = runtime.deployer
    monkeypatch.setattr(worker, "FLClientStarterConfiger", MagicMock(return_value=conf))

    def start_run(_root, received_args, *_args):
        assert received_args is args
        record("run")
        if runtime.outcome is not None:
            received_args.task_scope_outcome = runtime.outcome

    runtime.app = MagicMock()
    runtime.app.start_run.side_effect = start_run
    runtime.app.close.side_effect = lambda: record("commands")
    runtime.app.wait_for_command_callbacks.side_effect = lambda timeout: record("callbacks") or True
    runtime.default_factory = MagicMock(return_value=runtime.app)
    runtime.scoped_factory = MagicMock(return_value=runtime.app)
    monkeypatch.setattr(worker, "ClientAppRunner", runtime.default_factory)
    monkeypatch.setattr(runner_module, "TaskScopedClientAppRunner", runtime.scoped_factory)

    runtime.upload = MagicMock(side_effect=lambda *_args, **_kwargs: record("upload"))
    monkeypatch.setattr(worker, "upload_results_on_shutdown", runtime.upload)
    monkeypatch.setattr(
        worker, "create_stats_pool_files_for_job", MagicMock(side_effect=lambda *_args: record("stats"))
    )
    monkeypatch.setattr(job_process_cleanup, "shutdown_f3_streaming", lambda: record("streaming"))
    monkeypatch.setattr(job_process_cleanup, "security_close", lambda: record("security"))
    thread = MagicMock()
    thread.is_alive.return_value = True
    thread_factory = MagicMock(return_value=thread)

    def join():
        assert thread_factory.call_args.kwargs["args"][2].is_set()
        record("join")

    thread.join.side_effect = join
    monkeypatch.setattr(worker.threading, "Thread", thread_factory)
    return runtime


def _worker_scope(runtime, phase=None):
    runtime.args.set = [
        f"{ATTEMPT_OPTION}={runtime.attempt}",
        f"{DIRECTORY_OPTION}={runtime.directory}",
    ]
    if phase is not None:
        runtime.args.set.append(f"{PHASE_OPTION}={phase}")
        (runtime.directory / phase).mkdir()
    return runtime.args


def test_default_worker_still_uses_standard_runner_and_uploads(worker_runtime):
    runtime = worker_runtime
    assert worker.main(runtime.args) is None
    runtime.default_factory.assert_called_once()
    runtime.scoped_factory.assert_not_called()
    runtime.upload.assert_called_once()
    assert runtime.order == [
        "run",
        "commands",
        "stats",
        "upload",
        "streaming",
        "cell",
        "callbacks",
        "security",
        "deployer",
        "client",
        "join",
    ]
    assert not list(runtime.directory.rglob(RECEIPT_FILE))


def test_worker_receipt_follows_standard_main_finally_cleanup(worker_runtime):
    runtime = worker_runtime
    assert worker.main(_worker_scope(runtime)) == 0
    runtime.scoped_factory.assert_called_once()
    runtime.default_factory.assert_not_called()
    assert runtime.order == [
        "run",
        "commands",
        "stats",
        "upload",
        "streaming",
        "cell",
        "callbacks",
        "security",
        "deployer",
        "client",
        "join",
    ]
    assert read_receipt(str(runtime.directory), runtime.attempt) == dict(runtime.outcome, attempt=runtime.attempt)


def test_worker_does_not_reuse_stale_args_outcome(worker_runtime):
    runtime = worker_runtime
    runtime.outcome = None
    args = _worker_scope(runtime)
    args.task_scope_outcome = {"status": TASK_COMPLETE, "task_id": "old-task"}
    with pytest.raises(RuntimeError, match="without a clean runner outcome"):
        worker.main(args)
    assert not list(runtime.directory.rglob(RECEIPT_FILE))
    assert runtime.order[-3:] == ["deployer", "client", "join"]


@pytest.mark.parametrize(
    "options",
    [
        [f"{ATTEMPT_OPTION}=attempt"],
        [f"{DIRECTORY_OPTION}=/unused"],
        [f"{PHASE_OPTION}={PULL}"],
        [f"{ATTEMPT_OPTION}=", f"{DIRECTORY_OPTION}=/unused"],
    ],
)
def test_worker_rejects_incomplete_task_scope_options_before_startup(worker_runtime, options):
    worker_runtime.args.set = options
    with pytest.raises((ValueError, RuntimeError), match="requires an attempt ID"):
        worker.main(worker_runtime.args)
    worker.download_workspace.assert_not_called()
    worker_runtime.default_factory.assert_not_called()
    worker_runtime.scoped_factory.assert_not_called()


@pytest.fixture
def phased(runner, attempt_dir):
    fobs_initialize()
    directory, attempt = attempt_dir
    runner.job_id = "job-1"
    fl_ctx = runner.engine.new_context.return_value.__enter__.return_value
    peer_ctx = FLContext()
    peer_ctx.set_prop(FLContextKey.CURRENT_RUN, runner.job_id, private=False)
    peer_ctx.set_prop("round", 3, private=False)
    fl_ctx.set_peer_context(peer_ctx)
    task = runner.engine.get_task_assignment.return_value
    task.data["weight"] = 2
    task.data.add_cookie("round", 3)
    task.data.set_peer_context(peer_ctx)
    task.data.set_peer_props(peer_ctx.get_all_public_props())

    def phase_args(phase):
        args = _attempt_args(directory, attempt)
        args.set.append(f"{PHASE_OPTION}={phase}")
        return args

    return directory, attempt, phase_args


def test_pull_persists_full_input_context_without_executing_or_submitting(runner, phased):
    directory, attempt, phase_args = phased
    args = phase_args(PULL)
    runner.run("app", args)

    artifact = read_artifact(str(directory), attempt, runner.job_id, "input")
    assert artifact["data"]["weight"] == 2
    assert artifact["data"].get_cookie("round") == 3
    assert artifact["data"].get_peer_context().get_job_id() == runner.job_id
    assert artifact["data"].get_peer_props()["round"] == 3
    assert args.task_scope_outcome == {"status": INPUT_READY, "task_id": "task-1"}
    runner._process_task.assert_not_called()
    runner._send_task_result.assert_not_called()


@pytest.mark.parametrize("peer_job", [None, "another-job"])
def test_pull_cannot_commit_input_without_matching_server_context(runner, phased, peer_job):
    directory, _, phase_args = phased
    fl_ctx = runner.engine.new_context.return_value.__enter__.return_value
    if peer_job is None:
        fl_ctx.set_peer_context(None)
    else:
        fl_ctx.get_peer_context().set_prop(FLContextKey.CURRENT_RUN, peer_job, private=False)

    with pytest.raises(RuntimeError, match="client execution failed"):
        runner.run("app", phase_args(PULL))
    assert not (directory / "input.json").exists()


def test_compute_restores_context_runs_filters_and_commits_before_any_submission(runner, phased):
    directory, attempt, phase_args = phased
    task = runner.engine.get_task_assignment.return_value
    write_artifact(str(directory), attempt, runner.job_id, "input", task.name, task.task_id, task.data)
    fl_ctx = runner.engine.new_context.return_value.__enter__.return_value
    fl_ctx.set_peer_context(None)
    order = []

    def filter_input(data, context):
        assert context.get_peer_context().get_job_id() == runner.job_id
        assert context.get_peer_context().get_prop("round") == 3
        order.append("input_filter")
        data["weight"] += 1
        return data

    def execute(name, data, context, abort_signal):
        order.append("execute")
        assert name == "train"
        assert data["weight"] == 3
        return Shareable({"weight": data["weight"] * 2})

    def filter_result(data, context):
        order.append("result_filter")
        data["weight"] += 1
        return data

    runner.task_router.task_table["train"].execute = execute
    runner.task_data_filters = {f"train{FilterKey.DELIMITER}{FilterKey.IN}": [SimpleNamespace(process=filter_input)]}
    runner.task_result_filters = {
        f"train{FilterKey.DELIMITER}{FilterKey.OUT}": [SimpleNamespace(process=filter_result)]
    }
    runner.fire_event_with_data = lambda event, ctx, key, value: ctx.set_prop(key, value, private=True, sticky=False)
    runner._process_task = ClientRunner._process_task.__get__(runner)
    args = phase_args(COMPUTE)
    runner.run("app", args)

    result = read_artifact(str(directory), attempt, runner.job_id, "result")["data"]
    assert result["weight"] == 7
    assert result.get_cookie("round") == 3
    assert result.get_header(ReservedHeaderKey.TASK_ID) == "task-1"
    assert result.get_header(ReservedHeaderKey.TASK_NAME) == "train"
    assert order == ["input_filter", "execute", "result_filter"]
    assert EventType.BEFORE_SEND_TASK_RESULT not in [call.args[0] for call in runner.fire_event.call_args_list]
    assert args.task_scope_outcome == {"status": RESULT_READY, "task_id": "task-1"}
    runner.engine.get_task_assignment.assert_not_called()
    runner._send_task_result.assert_not_called()
    runner.engine.send_task_result.assert_not_called()


@pytest.mark.parametrize("failure", ["exception", "failed_result", "lazy_result", "pass_through"])
def test_compute_failure_never_commits_result_or_submits(runner, phased, failure):
    directory, attempt, phase_args = phased
    task = runner.engine.get_task_assignment.return_value
    write_artifact(str(directory), attempt, runner.job_id, "input", task.name, task.task_id, task.data)
    if failure == "exception":
        runner._process_task.side_effect = RuntimeError("trainer failed")
    elif failure == "failed_result":
        runner._process_task.return_value = make_reply(ReturnCode.EXECUTION_EXCEPTION)
    elif failure == "lazy_result":
        runner._process_task.return_value = Shareable({"weights": LazyDownloadRef("source", "download", "ref")})
    else:
        runner._process_task.return_value.set_header(ReservedHeaderKey.PASS_THROUGH, True)

    args = phase_args(COMPUTE)
    with pytest.raises(RuntimeError, match="client execution failed"):
        runner.run("app", args)
    assert args.task_scope_outcome is None
    assert not (directory / "result.json").exists()
    runner.engine.get_task_assignment.assert_not_called()
    runner._send_task_result.assert_not_called()


def test_pull_rejects_lazy_input_instead_of_committing_live_source_dependency(runner, phased):
    directory, _, phase_args = phased
    runner.engine.get_task_assignment.return_value.data["weight"] = LazyDownloadRef("source", "download", "ref")
    args = phase_args(PULL)
    with pytest.raises(RuntimeError, match="client execution failed"):
        runner.run("app", args)
    assert not (directory / "input.json").exists()
    assert args.task_scope_outcome is None
    runner._process_task.assert_not_called()


@pytest.mark.parametrize("phase,kind", [(COMPUTE, "input"), (PUSH, "result")])
def test_missing_handoff_fails_without_fetching_another_task(runner, phased, phase, kind):
    _, _, phase_args = phased
    args = phase_args(phase)
    with pytest.raises(RuntimeError, match="client execution failed") as error:
        runner.run("app", args)
    assert isinstance(error.value.__cause__, FileNotFoundError)
    assert args.task_scope_outcome is None
    runner.engine.get_task_assignment.assert_not_called()
    runner._process_task.assert_not_called()
    runner._send_task_result.assert_not_called()


@pytest.mark.parametrize("ack", [False, True])
def test_push_only_submits_persisted_payload_and_requires_ack(runner, phased, ack):
    directory, attempt, phase_args = phased
    result = Shareable({"weight": 7})
    result.set_header(ReservedHeaderKey.TASK_NAME, "train")
    result.set_header(ReservedHeaderKey.TASK_ID, "task-1")
    write_artifact(str(directory), attempt, runner.job_id, "result", "train", "task-1", result)

    def submit(data, task_id, context):
        assert data["weight"] == 7
        assert task_id == context.get_prop(FLContextKey.TASK_ID) == "task-1"
        assert context.get_prop(FLContextKey.TASK_NAME) == "train"
        if ack:
            # Normal END_RUN can arrive as soon as the last result is accepted.
            runner._handle_end_run("end_run", Shareable(), context)
        return ack

    runner._send_task_result.side_effect = submit
    args = phase_args(PUSH)
    if ack:
        runner.run("app", args)
        assert args.task_scope_outcome == {"status": TASK_COMPLETE, "task_id": "task-1"}
    else:
        with pytest.raises(RuntimeError, match="client execution failed"):
            runner.run("app", args)
        assert args.task_scope_outcome is None
    runner.engine.get_task_assignment.assert_not_called()
    runner._process_task.assert_not_called()
    runner._send_task_result.assert_called_once()
    assert [call.args[0] for call in runner.fire_event.call_args_list] == [
        EventType.BEFORE_SEND_TASK_RESULT,
        EventType.AFTER_SEND_TASK_RESULT,
    ]


@pytest.mark.parametrize("phase,status", [(PULL, INPUT_READY), (COMPUTE, RESULT_READY), (PUSH, TASK_COMPLETE)])
def test_phase_receipt_follows_cleanup_and_only_push_uploads_workspace(worker_runtime, phase, status):
    runtime = worker_runtime
    runtime.outcome = {"status": status, "task_id": "task-1"}
    assert worker.main(_worker_scope(runtime, phase)) == 0
    runtime.scoped_factory.assert_called_once()
    runtime.default_factory.assert_not_called()
    if phase == PUSH:
        runtime.upload.assert_called_once()
        assert runtime.order.index("upload") < runtime.order.index("streaming")
    else:
        runtime.upload.assert_not_called()
    assert runtime.order[-3:] == ["deployer", "client", "join"]
    assert read_receipt(str(runtime.directory / phase), runtime.attempt) == {
        "attempt": runtime.attempt,
        "phase": phase,
        "status": status,
        "task_id": "task-1",
    }
    assert not (runtime.directory / RECEIPT_FILE).exists()


@pytest.mark.parametrize("phase", [None, PUSH])
def test_worker_archive_failure_withholds_receipt_and_runs_remaining_cleanup(worker_runtime, phase):
    runtime = worker_runtime
    runtime.upload.side_effect = RuntimeError("archive upload failed")
    with pytest.raises(RuntimeError, match="archive upload failed"):
        worker.main(_worker_scope(runtime, phase))
    assert runtime.order[-6:] == ["cell", "callbacks", "security", "deployer", "client", "join"]
    assert not list(runtime.directory.rglob(RECEIPT_FILE))


@pytest.mark.parametrize("phase", [None, PULL, COMPUTE, PUSH])
def test_worker_runner_failure_withholds_receipt_and_still_cleans_up(worker_runtime, phase):
    runtime = worker_runtime
    runtime.app.start_run.side_effect = RuntimeError("runner failed")
    with pytest.raises(RuntimeError, match="runner failed"):
        worker.main(_worker_scope(runtime, phase))
    assert runtime.order[-3:] == ["deployer", "client", "join"]
    assert not list(runtime.directory.rglob(RECEIPT_FILE))


def test_final_worker_cleanup_failure_withholds_receipt(worker_runtime):
    runtime = worker_runtime
    runtime.deployer.close.side_effect = RuntimeError("deployer cleanup failed")
    with pytest.raises(RuntimeError, match="deployer cleanup failed"):
        worker.main(_worker_scope(runtime))
    assert not list(runtime.directory.rglob(RECEIPT_FILE))


def test_unknown_phase_is_rejected_before_any_worker_initialization(runner, worker_runtime):
    args = _worker_scope(worker_runtime, "bogus")
    with pytest.raises(ValueError, match="invalid task-scope phase"):
        worker.main(args)
    with pytest.raises(ValueError, match="invalid task-scope phase"):
        runner.run("app", args)
    worker.download_workspace.assert_not_called()
    worker_runtime.scoped_factory.assert_not_called()
    worker_runtime.default_factory.assert_not_called()
    runner.init_run.assert_not_called()
