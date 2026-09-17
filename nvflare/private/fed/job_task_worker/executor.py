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
"""Infrastructure Executor that supervises one fresh application worker per task."""

import json
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
import uuid

from nvflare.apis.event_type import EventType
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.job_launcher_spec import JobHandleSpec, JobProcessEnv, JobReturnCode
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.client.config import ConfigKey
from nvflare.private.fed.job_task_worker.artifacts import read_artifact, write_artifact
from nvflare.private.fed.job_task_worker.protocol import PUBLICATION_ACK_PROP, WORKER_MODULE
from nvflare.private.fed.utils.fed_utils import nvflare_fobs_initialize

_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,255}$")
_SAFE_ENVIRONMENT_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_CREDENTIAL_ENVIRONMENT = {JobProcessEnv.AUTH_TOKEN, JobProcessEnv.TOKEN_SIGNATURE, JobProcessEnv.SSID}


def _safe_component(name, value):
    if not isinstance(value, str) or not _SAFE_COMPONENT.fullmatch(value) or value in (".", ".."):
        raise ValueError(f"{name} must be a safe path component")
    return value


def _is_running(return_code):
    return return_code is None or return_code == JobReturnCode.UNKNOWN


def validate_worker_environment(environment):
    if environment is None:
        return {}
    if not isinstance(environment, dict):
        raise TypeError("worker_environment must be a dict")
    result = {}
    for name, value in environment.items():
        if not isinstance(name, str) or not _SAFE_ENVIRONMENT_NAME.fullmatch(name):
            raise ValueError("worker_environment names must be valid environment variable names")
        if name in _CREDENTIAL_ENVIRONMENT:
            raise ValueError("worker_environment must not contain federation credentials")
        if not isinstance(value, str) or "\x00" in value:
            raise ValueError("worker_environment values must be strings without NUL")
        result[name] = value
    return result


class LocalProcessHandle(JobHandleSpec):
    """A process-group handle whose wait boundary includes descendant cleanup."""

    def __init__(self, process, stop_grace, descendant_settle_timeout):
        self.process = process
        self.process_group = process.pid
        self.stop_grace = stop_grace
        self.descendant_settle_timeout = descendant_settle_timeout
        self.settled = False

    def poll(self):
        return self.process.poll()

    def _group_exists(self):
        try:
            os.killpg(self.process_group, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True

    def _signal_group(self, sig):
        try:
            os.killpg(self.process_group, sig)
            return True
        except (ProcessLookupError, PermissionError):
            return False

    def _wait_for_group_exit(self, timeout):
        deadline = time.monotonic() + timeout
        while self._group_exists():
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.01)
        return True

    def wait(self):
        result = self.process.wait()
        if self._group_exists():
            self._signal_group(signal.SIGTERM)
            if not self._wait_for_group_exit(self.descendant_settle_timeout):
                self._signal_group(signal.SIGKILL)
                self._wait_for_group_exit(self.stop_grace)
            self.settled = not self._group_exists()
            raise RuntimeError("application task-worker leader exited with live descendants")
        self.settled = True
        return result

    def terminate(self):
        group_signalled = self._signal_group(signal.SIGTERM)
        if not group_signalled and self.process.poll() is None:
            self.process.terminate()
        try:
            self.process.wait(timeout=self.stop_grace)
        except subprocess.TimeoutExpired:
            pass
        self._wait_for_group_exit(self.stop_grace)
        if self._group_exists():
            group_signalled = self._signal_group(signal.SIGKILL)
            if not group_signalled and self.process.poll() is None:
                self.process.kill()
            self._wait_for_group_exit(self.stop_grace)
        try:
            self.process.wait(timeout=self.stop_grace)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        self.settled = not self._group_exists()


class LocalProcessLauncher:
    """Local launcher with the same settled-handle contract required of Slurm."""

    def __init__(self, stop_grace=2.0, descendant_settle_timeout=0.25):
        self.stop_grace = stop_grace
        self.descendant_settle_timeout = descendant_settle_timeout

    def launch(self, command, *, cwd, env, fl_ctx):
        process = subprocess.Popen(command, cwd=cwd, env=env, start_new_session=True)
        return LocalProcessHandle(process, self.stop_grace, self.descendant_settle_timeout)


class JobTaskWorkerExecutor(Executor):
    """Keep federation in the resident CPU CJ and run application code elsewhere.

    The object is infrastructure, not the application Executor. It persists the
    already-filtered task, launches a fresh application worker, waits for process
    or allocation settlement, validates a durable result, and returns that result
    to the normal CJ runner. The runner retains result filters, publication retry,
    and ACK ownership.
    """

    def __init__(
        self,
        application_path: str,
        *,
        kind: str = "executor",
        application_args=None,
        arguments: str = "",
        components=None,
        state_id: str = "application",
        launcher_id: str = "",
        worker_timeout: float = 3600.0,
        poll_interval: float = 0.1,
        stop_grace: float = 2.0,
        descendant_settle_timeout: float = 0.25,
        task_exchange=None,
        memory_gc_rounds: int = 0,
        cuda_empty_cache: bool = False,
        source_mode: str = "",
        worker_environment=None,
    ):
        super().__init__()
        if kind not in ("executor", "script"):
            raise ValueError("kind must be executor or script")
        if not isinstance(application_path, str) or not application_path:
            raise ValueError("application_path must be a nonempty string")
        if not isinstance(worker_timeout, (int, float)) or worker_timeout <= 0:
            raise ValueError("worker_timeout must be positive")
        if not isinstance(poll_interval, (int, float)) or poll_interval <= 0:
            raise ValueError("poll_interval must be positive")
        _safe_component("state_id", state_id)
        if launcher_id:
            _safe_component("launcher_id", launcher_id)
        self.application_path = application_path
        self.kind = kind
        self.application_args = application_args or {}
        self.arguments = arguments
        self.components = components or []
        self.state_id = state_id
        self.launcher_id = launcher_id
        self.worker_timeout = float(worker_timeout)
        self.poll_interval = float(poll_interval)
        self.local_launcher = LocalProcessLauncher(stop_grace, descendant_settle_timeout)
        self.task_exchange = task_exchange or {}
        self.memory_gc_rounds = memory_gc_rounds
        self.cuda_empty_cache = cuda_empty_cache
        self.source_mode = source_mode
        self.worker_environment = validate_worker_environment(worker_environment)
        self._state_lock = threading.RLock()
        self._event_lock = threading.Lock()
        self._execution_lock = threading.Lock()
        self._active = None
        self._stopping = False
        self._event_file = None
        self._pending_publication = {}
        self._cj_instance_id = None
        nvflare_fobs_initialize()

    def _initialize_paths(self, fl_ctx):
        if self._event_file:
            return
        workspace = fl_ctx.get_workspace()
        job_id = _safe_component("job_id", fl_ctx.get_job_id())
        root = os.path.join(workspace.get_run_dir(job_id), ".job_task_worker_b")
        os.makedirs(root, mode=0o700, exist_ok=True)
        self._event_file = os.path.join(root, "events.jsonl")

    def _event(self, name, fl_ctx, **details):
        self._initialize_paths(fl_ctx)
        event = {
            "architecture": "B",
            "event": name,
            "time": time.time(),
            "monotonic": time.monotonic(),
            "wall_utc_ns": time.time_ns(),
            "monotonic_ns": time.monotonic_ns(),
            "job_id": fl_ctx.get_job_id(),
            "site": fl_ctx.get_identity_name(),
            **details,
        }
        with self._event_lock:
            with open(self._event_file, "a") as stream:
                stream.write(json.dumps(event, sort_keys=True) + "\n")

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.START_RUN:
            with self._state_lock:
                self._stopping = False
                # This is deliberately not the federation SSID.  It scopes
                # durable handoffs to one resident-CJ incarnation without
                # disclosing a federation session identifier to the worker.
                self._cj_instance_id = f"cj-{uuid.uuid4().hex}"
            self._event(
                "resident_cj_started",
                fl_ctx,
                resident_pid=os.getpid(),
                owner_instance_id=self._cj_instance_id,
            )
        elif event_type in (EventType.ABORT_TASK, EventType.END_RUN):
            with self._state_lock:
                self._stopping = True
                active = self._active
            if active is not None:
                active.terminate()
            self._event("resident_cj_stopping", fl_ctx, reason=event_type)
        elif event_type == EventType.AFTER_TASK_RESULT_FILTER:
            self._event(
                "cpu_cj_result_filters_complete",
                fl_ctx,
                task_id=fl_ctx.get_prop(FLContextKey.TASK_ID),
            )
        elif event_type == EventType.BEFORE_SEND_TASK_RESULT:
            task_id = fl_ctx.get_prop(FLContextKey.TASK_ID)
            self._event(
                "cpu_cj_publication_started",
                fl_ctx,
                task_id=task_id,
                attempt=self._pending_publication.get(task_id),
            )
        elif event_type == EventType.AFTER_SEND_TASK_RESULT:
            task_id = fl_ctx.get_prop(FLContextKey.TASK_ID)
            acknowledged = fl_ctx.get_prop(PUBLICATION_ACK_PROP) is True
            self._event(
                "cpu_cj_publication_acknowledged" if acknowledged else "cpu_cj_publication_not_acknowledged",
                fl_ctx,
                task_id=task_id,
                attempt=self._pending_publication.pop(task_id, None),
                acknowledged=acknowledged,
            )

    def _launcher(self, fl_ctx):
        if not self.launcher_id:
            return self.local_launcher
        launcher = fl_ctx.get_engine().get_component(self.launcher_id)
        if launcher is None or not callable(getattr(launcher, "launch", None)):
            raise RuntimeError(f"task-worker launcher component {self.launcher_id!r} is unavailable")
        return launcher

    def _worker_environment(self, custom_dir):
        env = os.environ.copy()
        for name in _CREDENTIAL_ENVIRONMENT:
            env.pop(name, None)
        env.update(self.worker_environment)
        source_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
        paths = [custom_dir, source_root]
        if env.get("PYTHONPATH"):
            paths.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(paths)
        return env

    def _write_spec(self, path, spec):
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as stream:
            json.dump(spec, stream, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        directory_fd = os.open(os.path.dirname(path), os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)

    def _application_entrypoint(self, custom_dir):
        if self.kind == "script" and not os.path.isabs(self.application_path):
            path = self.application_path
            if path.startswith("custom/"):
                path = path[len("custom/") :]
            return os.path.join(custom_dir, path)
        return self.application_path

    def _request_termination(self, handle, fl_ctx, attempt, reason):
        self._event("worker_termination_requested", fl_ctx, attempt=attempt, reason=reason)
        handle.terminate()

    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        if not self._execution_lock.acquire(blocking=False):
            raise RuntimeError("Architecture B prototype admits only one application task at a time per CJ")
        try:
            return self._execute(task_name, shareable, fl_ctx, abort_signal)
        finally:
            self._execution_lock.release()

    def _execute(self, task_name, shareable, fl_ctx, abort_signal):
        if not isinstance(shareable, Shareable):
            raise TypeError("task-worker input must be a Shareable")
        job_id = _safe_component("job_id", fl_ctx.get_job_id())
        task_id = fl_ctx.get_prop(FLContextKey.TASK_ID)
        if not isinstance(task_id, str) or not task_id:
            raise RuntimeError("task-worker execution requires the current task ID")
        with self._state_lock:
            if self._cj_instance_id is None:
                self._cj_instance_id = f"cj-{uuid.uuid4().hex}"
            handoff_scope_id = self._cj_instance_id
            if self._stopping or abort_signal.triggered:
                raise RuntimeError("resident CJ is stopping and rejects new application task admission")

        workspace = fl_ctx.get_workspace()
        run_dir = workspace.get_run_dir(job_id)
        root = os.path.join(run_dir, ".job_task_worker_b")
        attempts_root = os.path.join(root, "attempts")
        state_dir = os.path.join(root, "state", self.state_id)
        os.makedirs(attempts_root, mode=0o700, exist_ok=True)
        os.makedirs(state_dir, mode=0o700, exist_ok=True)
        attempt = uuid.uuid4().hex
        attempt_dir = os.path.join(attempts_root, attempt)
        os.mkdir(attempt_dir, mode=0o700)
        self._event("task_staging_started", fl_ctx, task_id=task_id, task_name=task_name, attempt=attempt)
        write_artifact(
            attempt_dir,
            attempt=attempt,
            job_id=job_id,
            session_id=handoff_scope_id,
            task_id=task_id,
            task_name=task_name,
            kind="input",
            data=shareable,
        )
        custom_dir = workspace.get_app_custom_dir(job_id)
        spec = {
            "version": 1,
            "attempt": attempt,
            "job_id": job_id,
            "handoff_scope_id": handoff_scope_id,
            "task_id": task_id,
            "task_name": task_name,
            "client_name": fl_ctx.get_identity_name(),
            "workspace_root": workspace.get_root_dir(),
            "app_root": workspace.get_app_dir(job_id),
            "state_dir": state_dir,
            "kind": self.kind,
            "application_path": self._application_entrypoint(custom_dir),
            "application_args": self.application_args,
            "arguments": self.arguments,
            "components": self.components,
            "task_exchange": self.task_exchange,
            "memory_gc_rounds": self.memory_gc_rounds,
            "cuda_empty_cache": self.cuda_empty_cache,
            "source_mode": self.source_mode,
        }
        self._write_spec(os.path.join(attempt_dir, "worker.json"), spec)
        self._event("input_committed", fl_ctx, task_id=task_id, attempt=attempt)
        command = [sys.executable, "-m", WORKER_MODULE, attempt_dir]
        launcher = self._launcher(fl_ctx)
        worker_env = self._worker_environment(custom_dir)
        with self._state_lock:
            if self._stopping or abort_signal.triggered:
                raise RuntimeError("resident CJ stopped before application worker launch")
        handle = launcher.launch(command, cwd=run_dir, env=worker_env, fl_ctx=fl_ctx)
        with self._state_lock:
            self._active = handle
            stopping = self._stopping
        details = {"task_id": task_id, "attempt": attempt}
        scheduler_id = getattr(handle, "job_id", None)
        worker_pid = getattr(getattr(handle, "process", None), "pid", None)
        if scheduler_id is not None:
            details["scheduler_id"] = scheduler_id
        if worker_pid is not None:
            details["worker_pid"] = worker_pid
        self._event("worker_launched", fl_ctx, **details)
        deadline = time.monotonic() + self.worker_timeout
        termination_requested = False
        try:
            while _is_running(handle.poll()):
                reason = None
                if stopping or self._stopping or abort_signal.triggered:
                    reason = "abort"
                elif time.monotonic() >= deadline:
                    reason = "timeout"
                if reason and not termination_requested:
                    termination_requested = True
                    self._request_termination(handle, fl_ctx, attempt, reason)
                time.sleep(self.poll_interval)
            waited_code = handle.wait()
            return_code = handle.poll()
            if return_code is None:
                return_code = waited_code
            self._event(
                "allocation_settled",
                fl_ctx,
                task_id=task_id,
                attempt=attempt,
                return_code=return_code,
                settled=getattr(handle, "settled", True),
            )
        finally:
            with self._state_lock:
                if self._active is handle:
                    self._active = None
        if termination_requested or self._stopping or abort_signal.triggered:
            raise RuntimeError("application task worker was cancelled")
        if return_code != 0:
            raise RuntimeError(f"application task worker exited with code {return_code}")
        result_task_name, result = read_artifact(
            attempt_dir,
            attempt=attempt,
            job_id=job_id,
            session_id=handoff_scope_id,
            task_id=task_id,
            kind="result",
        )
        if result_task_name != task_name:
            raise ValueError("result artifact task name does not match its assignment")
        receipt_path = os.path.join(attempt_dir, "worker_receipt.json")
        if not os.path.isfile(receipt_path):
            raise RuntimeError("application worker did not commit its completion receipt")
        with open(receipt_path) as stream:
            receipt = json.load(stream)
        if any(
            receipt.get(key) != value
            for key, value in (
                ("attempt", attempt),
                ("job_id", job_id),
                ("handoff_scope_id", handoff_scope_id),
                ("task_id", task_id),
            )
        ):
            raise RuntimeError("application worker receipt identity mismatch")
        self._event(
            "durable_result_loaded",
            fl_ctx,
            task_id=task_id,
            attempt=attempt,
            worker_pid=receipt.get("pid"),
            worker_cpu_seconds=receipt.get("user_cpu_seconds", 0) + receipt.get("system_cpu_seconds", 0),
            worker_max_rss_native_units=receipt.get("max_rss_native_units"),
            worker_max_rss_unit=receipt.get("max_rss_unit"),
            input_bytes=os.path.getsize(os.path.join(attempt_dir, "input.fobs")),
            result_bytes=os.path.getsize(os.path.join(attempt_dir, "result.fobs")),
        )
        self._pending_publication[task_id] = attempt
        self._event("result_returned_to_cpu_cj", fl_ctx, task_id=task_id, attempt=attempt)
        return result


class ClientAPIJobTaskWorkerExecutor(JobTaskWorkerExecutor):
    """Translate current in-process/external-process ClientAPI configs to B workers.

    The source mode is retained as evidence, but both modes intentionally execute
    the application script inside the one fresh task worker. Attach and a resident
    launch-once trainer are not preserved by this disposable-worker contract.
    """

    def __init__(
        self,
        execution_mode: str,
        command=None,
        task_script_path=None,
        task_script_args: str = "",
        launch_once: bool = True,
        launch_timeout=None,
        shutdown_timeout=None,
        stop_grace_period: float = 30.0,
        heartbeat_interval: float = 5.0,
        heartbeat_timeout: float = 30.0,
        task_wait_timeout=None,
        result_wait_timeout=None,
        train_task_name="train",
        evaluate_task_name="validate",
        submit_model_task_name="submit_model",
        train_with_evaluation: bool = False,
        params_exchange_format="raw",
        server_expected_format="numpy",
        params_transfer_type="full",
        memory_gc_rounds: int = 0,
        cuda_empty_cache: bool = False,
        attach_id=None,
        attach_timeout=None,
        allow_reconnect: bool = False,
        allow_insecure_attach: bool = False,
        *,
        launcher_id: str = "",
        state_id: str = "client-api",
        worker_timeout: float = 3600.0,
        poll_interval: float = 0.1,
        resources=None,
        script_resource: str = "",
        worker_environment=None,
    ):
        self._execution_mode = execution_mode
        self._command = command
        self._task_script_path = task_script_path
        self._task_script_args = task_script_args
        self._launch_once = launch_once
        self._launch_timeout = launch_timeout
        self._shutdown_timeout = shutdown_timeout
        self._stop_grace_period = stop_grace_period
        self._heartbeat_interval = heartbeat_interval
        self._heartbeat_timeout = heartbeat_timeout
        self._task_wait_timeout = task_wait_timeout
        self._result_wait_timeout = result_wait_timeout
        self._train_task_name = train_task_name
        self._evaluate_task_name = evaluate_task_name
        self._submit_model_task_name = submit_model_task_name
        self._train_with_evaluation = train_with_evaluation
        self._params_exchange_format = params_exchange_format
        self._server_expected_format = server_expected_format
        self._params_transfer_type = params_transfer_type
        self._memory_gc_rounds = memory_gc_rounds
        self._cuda_empty_cache = cuda_empty_cache
        self._attach_id = attach_id
        self._attach_timeout = attach_timeout
        self._allow_reconnect = allow_reconnect
        self._allow_insecure_attach = allow_insecure_attach
        if execution_mode == "in_process":
            script = task_script_path
            arguments = task_script_args
            if command:
                raise ValueError("in_process task-worker config does not accept command")
        elif execution_mode == "external_process":
            if task_script_path:
                raise ValueError("external_process task-worker config does not accept task_script_path")
            tokens = shlex.split(command) if isinstance(command, str) else list(command or [])
            script_index = next((index for index, token in enumerate(tokens) if token.endswith(".py")), None)
            if script_index is None:
                raise ValueError("external_process task-worker command must identify a Python task script")
            script = tokens[script_index]
            arguments = shlex.join(tokens[script_index + 1 :])
        else:
            raise ValueError("Architecture B task workers support in_process or external_process config, not Attach")
        if not isinstance(script, str) or not script:
            raise ValueError("Client API task-worker adapter requires a Python task script")
        task_exchange = {
            ConfigKey.EXCHANGE_FORMAT: params_exchange_format,
            ConfigKey.SERVER_EXPECTED_FORMAT: server_expected_format,
            ConfigKey.TRANSFER_TYPE: params_transfer_type,
            ConfigKey.TRAIN_TASK_NAME: train_task_name,
            ConfigKey.EVAL_TASK_NAME: evaluate_task_name,
            ConfigKey.SUBMIT_MODEL_TASK_NAME: submit_model_task_name,
            ConfigKey.TRAIN_WITH_EVAL: train_with_evaluation,
        }
        resource_path = script[len("custom/") :] if script.startswith("custom/") else script
        deployed_script = os.path.basename(script) if os.path.isabs(script) else resource_path
        if execution_mode == "in_process":
            self._task_script_path = deployed_script
        super().__init__(
            application_path=deployed_script,
            kind="script",
            arguments=arguments,
            state_id=state_id,
            launcher_id=launcher_id,
            worker_timeout=worker_timeout,
            poll_interval=poll_interval,
            task_exchange=task_exchange,
            memory_gc_rounds=memory_gc_rounds,
            cuda_empty_cache=cuda_empty_cache,
            source_mode=execution_mode,
            worker_environment=worker_environment,
        )
        self._resource_for_job = script_resource or resource_path
        self._resources_for_job = resources or []

    def add_to_fed_job(self, job, ctx, **kwargs):
        job.check_kwargs(args_to_check=kwargs, args_expected={"tasks": False})
        tasks = kwargs.get("tasks", ["*"])
        job.add_executor(self, tasks=tasks, ctx=ctx)
        job.add_file_source(
            src_path=self._resource_for_job,
            dest_dir=None,
            app_folder_type="custom",
            ctx=ctx,
        )
        if self._resources_for_job:
            job.add_resources(resources=self._resources_for_job, ctx=ctx)
        return {}
