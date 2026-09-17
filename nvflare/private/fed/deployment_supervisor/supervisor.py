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
"""Deployment-level acquisition, worker supervision, and result publication."""

import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass

from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContextManager
from nvflare.apis.job_launcher_spec import JobHandleSpec, JobProcessEnv
from nvflare.apis.shareable import Shareable
from nvflare.private.fed.deployment_supervisor.artifacts import read_artifact, write_artifact
from nvflare.private.fed.utils.fed_utils import nvflare_fobs_initialize

_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,255}$")


def _safe_component(name, value):
    if not isinstance(value, str) or not _SAFE_COMPONENT.fullmatch(value) or value in (".", ".."):
        raise ValueError(f"{name} must be a safe path component")
    return value


@dataclass(frozen=True)
class WorkerDefinition:
    kind: str
    entrypoint: str
    arguments: str = ""
    lifetime: str = "task"
    state_id: str = ""
    task_exchange: dict = None
    memory_gc_rounds: int = 0
    cuda_empty_cache: bool = False
    factory_args: dict = None
    task_components: list = None
    input_filters: list = None
    result_filters: list = None
    source_mode: str = ""
    module_search_paths: list = None

    def validate(self):
        if self.kind not in ("executor", "script"):
            raise ValueError("worker kind must be executor or script")
        if not self.entrypoint:
            raise ValueError("worker entrypoint is required")
        for path in self.module_search_paths or []:
            if not isinstance(path, str) or not os.path.isabs(path):
                raise ValueError("worker module search paths must be absolute paths")
        if self.lifetime != "task":
            raise ValueError(
                "architecture A prototype supports fresh task workers only; use the unchanged resident CJ path "
                "or migrate cross-task state with state_id"
            )


class LocalProcessHandle(JobHandleSpec):
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
            # Darwin can return EPERM for a still-existing orphaned process
            # group after its original leader has been reaped.
            return True

    def _signal_group(self, sig):
        try:
            os.killpg(self.process_group, sig)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
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
            raise RuntimeError("task worker leader exited with live descendants")
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
    """Local launcher with the same settle-before-return JobHandle contract."""

    def __init__(self, stop_grace=2.0, descendant_settle_timeout=0.25):
        self.stop_grace = stop_grace
        self.descendant_settle_timeout = descendant_settle_timeout

    def launch(self, command, *, cwd, env):
        process = subprocess.Popen(command, cwd=cwd, env=env, start_new_session=True)
        return LocalProcessHandle(process, self.stop_grace, self.descendant_settle_timeout)


class DeploymentTaskSupervisor:
    """A site-deployment owner shared across jobs; workers never acquire/publish."""

    def __init__(
        self,
        *,
        service,
        workspace,
        routes,
        launcher=None,
        worker_timeout=300.0,
        poll_interval=0.05,
        site_input_filters=None,
        job_input_filters=None,
        site_result_filters=None,
        job_result_filters=None,
        publication_handlers=None,
        event_sink=None,
    ):
        self.service = service
        self.workspace = os.path.abspath(workspace)
        self.routes = routes
        self.launcher = launcher or LocalProcessLauncher()
        self.worker_timeout = worker_timeout
        self.poll_interval = poll_interval
        self.site_input_filters = site_input_filters or []
        self.job_input_filters = job_input_filters or []
        self.site_result_filters = site_result_filters or []
        self.job_result_filters = job_result_filters or []
        self.publication_handlers = publication_handlers or []
        self.event_sink = event_sink
        self._cancel = threading.Event()
        self._active = None
        nvflare_fobs_initialize()
        os.makedirs(self.workspace, mode=0o700, exist_ok=True)

    def cancel(self):
        self._cancel.set()
        if self._active is not None:
            self._active.terminate()

    def _event(self, name, **details):
        event = {"event": name, "time": time.time(), **details}
        path = os.path.join(self.workspace, "events.jsonl")
        with open(path, "a") as stream:
            stream.write(json.dumps(event, sort_keys=True) + "\n")
        if self.event_sink:
            self.event_sink(event)

    def _apply_filters(self, filters, data, fl_ctx, stage):
        value = data
        for task_filter in filters:
            self._event(f"{stage}_filter", filter=task_filter.__class__.__name__)
            value = task_filter.process(value, fl_ctx)
            if not isinstance(value, Shareable):
                raise TypeError(f"{stage} filter returned {type(value)} instead of Shareable")
        return value

    def _context(self, assignment):
        manager = FLContextManager(identity_name=assignment.client_name, job_id=assignment.job_id)
        fl_ctx = manager.new_context()
        fl_ctx.set_prop(FLContextKey.TASK_NAME, assignment.task_name, private=True, sticky=False)
        fl_ctx.set_prop(FLContextKey.TASK_ID, assignment.task_id, private=True, sticky=False)
        return fl_ctx

    def run_once(self, client_name):
        if self._cancel.is_set():
            return False
        assignment = self.service.acquire(client_name)
        if assignment is None:
            return False
        if self._cancel.is_set():
            self.service.fail(assignment, "task cancelled before worker launch")
            return True
        worker = self.routes.get(assignment.task_name) or self.routes.get("*")
        if not isinstance(worker, WorkerDefinition):
            self.service.fail(assignment, f"no worker route for {assignment.task_name}")
            return True
        try:
            worker.validate()
            job_id = _safe_component("job_id", assignment.job_id)
            attempt = _safe_component("attempt", assignment.attempt)
            state_id = _safe_component("state_id", worker.state_id or assignment.client_name)
            attempt_dir = os.path.join(self.workspace, job_id, attempt)
            os.makedirs(attempt_dir, mode=0o700)
            state_dir = os.path.join(self.workspace, "state", job_id, state_id)
            os.makedirs(state_dir, mode=0o700, exist_ok=True)
            fl_ctx = self._context(assignment)
            self._event("acquired", task_id=assignment.task_id, attempt=assignment.attempt)
            staged = self._apply_filters(
                [*self.site_input_filters, *self.job_input_filters], assignment.data, fl_ctx, "input"
            )
            write_artifact(
                attempt_dir,
                attempt=assignment.attempt,
                job_id=assignment.job_id,
                task_id=assignment.task_id,
                task_name=assignment.task_name,
                kind="input",
                data=staged,
            )
            spec = {
                **asdict(worker),
                "task_exchange": worker.task_exchange or {},
                "attempt": assignment.attempt,
                "job_id": assignment.job_id,
                "task_id": assignment.task_id,
                "task_name": assignment.task_name,
                "client_name": assignment.client_name,
                "round_number": assignment.round_number,
                "state_dir": state_dir,
            }
            spec_path = os.path.join(attempt_dir, "worker.json")
            fd = os.open(spec_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "w") as stream:
                json.dump(spec, stream, sort_keys=True)
                stream.flush()
                os.fsync(stream.fileno())
            command = [sys.executable, "-m", "nvflare.private.fed.deployment_supervisor.worker", attempt_dir]
            self._event("launching", task_id=assignment.task_id, attempt=assignment.attempt)
            worker_env = os.environ.copy()
            for name in (JobProcessEnv.AUTH_TOKEN, JobProcessEnv.TOKEN_SIGNATURE, JobProcessEnv.SSID):
                worker_env.pop(name, None)
            self._active = self.launcher.launch(command, cwd=os.getcwd(), env=worker_env)
            self._event("launched", task_id=assignment.task_id, attempt=assignment.attempt)
            deadline = time.monotonic() + self.worker_timeout
            while self._active.poll() is None:
                if self._cancel.is_set() or time.monotonic() >= deadline:
                    self._active.terminate()
                    break
                time.sleep(self.poll_interval)
            # JobHandleSpec.wait() is a completion barrier; it does not promise
            # to return the terminal code (SlurmJobHandle.wait returns None).
            # Read the settled status through poll(), which is the status API.
            self._active.wait()
            return_code = self._active.poll()
            self._event(
                "allocation_settled",
                task_id=assignment.task_id,
                attempt=assignment.attempt,
                return_code=return_code,
            )
            self._active = None
            if self._cancel.is_set():
                raise RuntimeError("task cancelled")
            if return_code != 0:
                raise RuntimeError(f"task worker exited with code {return_code}")
            result_task_name, result = read_artifact(
                attempt_dir,
                attempt=assignment.attempt,
                job_id=assignment.job_id,
                task_id=assignment.task_id,
                kind="result",
            )
            if result_task_name != assignment.task_name:
                raise ValueError("result artifact task name does not match its assignment")
            result = self._apply_filters(
                [*self.site_result_filters, *self.job_result_filters], result, fl_ctx, "result"
            )
            for handler in self.publication_handlers:
                handler.before_publish(assignment, result, fl_ctx)
            self.service.publish(assignment, result)
            self._event("published", task_id=assignment.task_id, attempt=assignment.attempt)
            for handler in self.publication_handlers:
                try:
                    handler.after_publish(assignment, result, fl_ctx)
                except Exception as e:
                    self._event(
                        "publication_handler_failed",
                        task_id=assignment.task_id,
                        attempt=assignment.attempt,
                        reason=str(e),
                    )
        except Exception as e:
            active = self._active
            if active is not None:
                try:
                    active.terminate()
                    active.wait()
                except Exception as cleanup_error:
                    self._event(
                        "allocation_cleanup_failed",
                        task_id=assignment.task_id,
                        attempt=assignment.attempt,
                        reason=str(cleanup_error),
                    )
                finally:
                    self._active = None
            self.service.fail(assignment, e)
            self._event("failed", task_id=assignment.task_id, attempt=assignment.attempt, reason=str(e))
        return True

    def run_until_idle(self, client_name):
        count = 0
        while self.run_once(client_name):
            count += 1
        return count
