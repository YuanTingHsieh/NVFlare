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
"""Internal task-scoped participation, independent of the physical launcher."""

import json
import os
import shlex
import threading
import time
import uuid

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey, JobConstants, ReturnCode
from nvflare.apis.job_launcher_spec import JobHandleSpec, JobProcessArgs, JobReturnCode
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.fuel.common.exit_codes import PROCESS_EXIT_REASON, ProcessExitCode
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey
from nvflare.fuel.f3.cellnet.defs import ReturnCode as CellReturnCode
from nvflare.private.aux_runner import AuxMsgTarget
from nvflare.private.defs import CellChannel, new_cell_message
from nvflare.private.fed.task_scope.protocol import (
    ATTEMPT_OPTION,
    COMPUTE,
    DIRECTORY_OPTION,
    DONE,
    END_RUN,
    ERROR,
    IDLE,
    INPUT_READY,
    PHASE_OPTION,
    PHASES,
    PROBE_TOPIC,
    PULL,
    PUSH,
    READY,
    REASON,
    RESULT_READY,
    STATUS,
    TASK_COMPLETE,
    TASK_TOKEN,
    TERMINAL_TOPIC,
    WAIT,
    read_receipt,
)
from nvflare.private.fed.utils.fed_utils import get_return_code


def launch_task_scope_worker(physical_launch, fl_ctx, attempt, directory, phase=None):
    """Invoke an existing launcher with common one-task worker bootstrap arguments."""
    job_args = fl_ctx.get_prop(FLContextKey.JOB_PROCESS_ARGS)
    if not isinstance(job_args, dict):
        raise RuntimeError("task-scoped launch requires prepared job process arguments")
    scoped = dict(job_args)
    option, value = scoped.get(JobProcessArgs.OPTIONS, ("--set", ""))
    if option != "--set" or not isinstance(value, str):
        raise RuntimeError("task-scoped launch requires valid job process set options")
    additions = " ".join((f"{ATTEMPT_OPTION}={shlex.quote(attempt)}", f"{DIRECTORY_OPTION}={shlex.quote(directory)}"))
    if phase is not None:
        if phase not in PHASES:
            raise ValueError("invalid task phase")
        additions += f" {PHASE_OPTION}={phase}"
    scoped[JobProcessArgs.OPTIONS] = (option, f"{value} {additions}".strip())
    original_phase = fl_ctx.get_prop(PHASE_OPTION)
    fl_ctx.set_prop(FLContextKey.JOB_PROCESS_ARGS, scoped, private=True, sticky=False)
    if phase is not None:
        fl_ctx.set_prop(PHASE_OPTION, phase, private=True, sticky=False)
    try:
        return physical_launch()
    finally:
        fl_ctx.set_prop(FLContextKey.JOB_PROCESS_ARGS, job_args, private=True, sticky=False)
        if phase is not None:
            fl_ctx.set_prop(PHASE_OPTION, original_phase, private=True, sticky=False)


class TaskScopedJobHandle(JobHandleSpec):
    """CP-owned logical job spanning sequential physical worker allocations.

    The launcher adapter supplies physical handles whose wait() settles allocation
    cleanup before returning. The shared workspace must survive worker exit.
    Parent restart/adoption and semantic server commit receipts are NOT provided.
    """

    def __init__(
        self,
        job_id,
        run_dir,
        probe,
        logger,
        poll_interval=2.0,
        communication_timeout=120.0,
        *,
        launch_attempt=None,
        allocation_details=None,
        phased=False,
    ):
        self.job_id = job_id
        self.run_dir = run_dir
        self.probe = probe
        self.logger = logger
        self.poll_interval = poll_interval
        self.communication_timeout = communication_timeout
        self.launch_attempt = launch_attempt
        self.allocation_details = allocation_details
        self.phased = phased
        self.active = None
        self.result = None
        self._server_terminal = None
        self._idle_task_token = None
        self._cancel = threading.Event()
        self._lock = threading.RLock()
        self._wait_lock = threading.Lock()
        self._root = os.path.join(run_dir, ".task_scope")
        os.makedirs(self._root, mode=0o700, exist_ok=True)

    def _record(self, phase, **details):
        event = dict(time=time.time(), monotonic=time.monotonic(), job_id=self.job_id, phase=phase, **details)
        try:
            with open(os.path.join(self._root, "events.jsonl"), "a") as stream:
                stream.write(json.dumps(event) + "\n")
        except OSError:
            self.logger.warning("could not append task-scope diagnostic evidence", exc_info=True)
        self.logger.info("task-scoped job %s", event)

    def notify_terminal(self, status):
        if status not in (DONE, ERROR):
            raise ValueError("invalid terminal notification")
        with self._lock:
            if self._server_terminal != ERROR:
                self._server_terminal = status
        if status == ERROR:
            self.terminate()

    def terminate(self):
        self._cancel.set()
        with self._lock:
            active = self.active
        if active is not None:
            active.terminate()

    def _terminate_for_heartbeat_cleanup(self):
        # Missing server job is not proof of successful execution/publication.
        self.terminate()

    def poll(self):
        with self._lock:
            return JobReturnCode.UNKNOWN if self.result is None else self.result

    def _launch_attempt(self, attempt, directory, phase=None):
        """Launch the one-task worker with its attempt ID and receipt directory."""
        if self.launch_attempt is None:
            raise NotImplementedError
        if phase is None:
            return self.launch_attempt(attempt, directory)
        return self.launch_attempt(attempt, directory, phase)

    def _allocation_details(self, active):
        """Optional launcher-specific diagnostic identifiers."""
        return self.allocation_details(active) if self.allocation_details else {}

    def _run_allocation(self, attempt, directory, phase=None):
        receipt_dir = os.path.join(directory, phase) if phase is not None else directory
        if phase is not None:
            os.mkdir(receipt_dir, mode=0o700)
        details = {"task_phase": phase} if phase is not None else {}
        # Consume any old legacy marker before submission, including when the
        # next allocation fails before Python/MPM gets a chance to clear it.
        stale_rc = os.path.join(self.run_dir, "_process_rc.txt")
        if os.path.exists(stale_rc):
            os.replace(stale_rc, os.path.join(receipt_dir, "previous_process_rc.txt"))
        self._record("submitting", attempt=attempt, **details)
        active = (
            self._launch_attempt(attempt, directory, phase)
            if phase is not None
            else self._launch_attempt(attempt, directory)
        )
        with self._lock:
            self.active = active
        self._record("allocated", attempt=attempt, **details, **self._allocation_details(active))
        if self._cancel.is_set():
            active.terminate()
        # The adapter must settle allocation accounting and artifact cleanup;
        # only then may the same job FQCN and launcher paths be used again.
        active.wait()
        raw_rc = active.poll()
        rc = get_return_code(active, self.job_id, os.path.dirname(self.run_dir), self.logger)
        with self._lock:
            self.active = None
        self._record(
            "allocation_released",
            attempt=attempt,
            rc=rc,
            launcher_rc=raw_rc,
            **details,
            **self._allocation_details(active),
        )
        if self._cancel.is_set():
            return JobReturnCode.ABORTED, None
        # In this experiment a successful file cannot hide scheduler failure.
        if raw_rc != JobReturnCode.SUCCESS or rc != JobReturnCode.SUCCESS:
            code = raw_rc if raw_rc != JobReturnCode.SUCCESS else rc
            # The parent reports only its known failure-code domain. Do not let
            # an unclassified OS code become a non-reportable logical outcome.
            return (
                code if code in PROCESS_EXIT_REASON or code == JobReturnCode.ABORTED else ProcessExitCode.EXCEPTION
            ), None
        receipt = read_receipt(receipt_dir, attempt)
        if phase is not None and receipt.get("phase") != phase:
            raise ValueError("receipt does not match the launched phase")
        if phase is None and "phase" in receipt:
            raise ValueError("baseline worker cannot return a phase receipt")
        self._record("receipt", attempt=attempt, receipt=receipt, **details)
        return None, receipt

    def _run_attempt(self, task_token):
        attempt = uuid.uuid4().hex
        directory = os.path.join(self._root, attempt)
        os.mkdir(directory, mode=0o700)
        task_id = None
        for phase in PHASES if self.phased else (None,):
            if self._cancel.is_set():
                return JobReturnCode.ABORTED
            # Completion while a handoff is pending is not proof of publication.
            if phase in (COMPUTE, PUSH) and self._server_terminal is not None:
                return ProcessExitCode.EXCEPTION
            rc, receipt = self._run_allocation(attempt, directory, phase)
            if rc is not None:
                return rc
            if phase is not None:
                if phase == PULL and receipt[STATUS] in (IDLE, END_RUN):
                    break
                expected = {PULL: INPUT_READY, COMPUTE: RESULT_READY, PUSH: TASK_COMPLETE}[phase]
                if receipt[STATUS] != expected:
                    raise ValueError("unexpected phase receipt status")
                if task_id is not None and receipt["task_id"] != task_id:
                    raise ValueError("task identity changed between phases")
                task_id = receipt["task_id"]
        if receipt[STATUS] == IDLE:
            # A readiness hint can become stale while the launcher queues the CJ.
            # Do not reacquire resources repeatedly for the same invalidated hint.
            self._idle_task_token = task_token
        if receipt[STATUS] == END_RUN:
            self.notify_terminal(DONE)
        return None

    def _run(self):
        last_response = time.monotonic()
        self._record("waiting_for_work")
        while not self._cancel.is_set():
            with self._lock:
                terminal = self._server_terminal
            if terminal is not None:
                return JobReturnCode.SUCCESS if terminal == DONE else ProcessExitCode.EXCEPTION
            reply = self.probe()
            # A terminal notice may arrive while the request was in flight.
            with self._lock:
                terminal = self._server_terminal
            if self._cancel.is_set():
                break
            if terminal is not None:
                return JobReturnCode.SUCCESS if terminal == DONE else ProcessExitCode.EXCEPTION
            if reply is None:
                if time.monotonic() - last_response >= self.communication_timeout:
                    raise RuntimeError("SJ unavailable without an acknowledged terminal outcome")
            else:
                last_response = time.monotonic()
                status = reply.get(STATUS)
                if status == DONE:
                    return JobReturnCode.SUCCESS
                if status == ERROR:
                    raise RuntimeError(reply.get(REASON, "task availability probe rejected"))
                if status == READY:
                    task_token = reply.get(TASK_TOKEN)
                    if not isinstance(task_token, str) or not task_token:
                        raise RuntimeError("readiness reply requires a task identity")
                    if task_token == self._idle_task_token:
                        self._cancel.wait(self.poll_interval)
                        continue
                    if self._cancel.is_set():
                        break
                    rc = self._run_attempt(task_token)
                    if rc is not None:
                        return rc
                    self._record("waiting_for_work")
                    last_response = time.monotonic()
                    continue
                if status != WAIT:
                    raise RuntimeError("invalid task availability reply")
            self._cancel.wait(self.poll_interval)
        return ProcessExitCode.EXCEPTION if self._server_terminal == ERROR else JobReturnCode.ABORTED

    def wait(self):
        with self._wait_lock:
            if self.result is not None:
                return
            try:
                result = self._run()
            except Exception:
                self.logger.exception("experimental task-scoped lifecycle failed")
                # If submission succeeded, never orphan its allocation because
                # evidence writing or receipt processing raised.
                self.terminate()
                if self.active is not None:
                    self.active.wait()
                result = ProcessExitCode.INFRASTRUCTURE_ERROR
            with self._lock:
                self.result = result
            self._record("logical_job_finished", rc=result)


class TaskScopedJobRegistry:
    """CP control endpoints and logical handles, shared by launcher adapters."""

    def __init__(self, poll_interval=2.0, probe_timeout=5.0, communication_timeout=120.0):
        for value in (poll_interval, probe_timeout, communication_timeout):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
                raise ValueError("task-scope timeouts must be positive numbers")
        self.poll_interval = poll_interval
        self.probe_timeout = probe_timeout
        self.communication_timeout = communication_timeout
        self._handles = {}
        self._lock = threading.Lock()
        self._terminal_cell = None

    def release(self, job_id):
        with self._lock:
            handle = self._handles.get(job_id)
            if handle is not None and handle.poll() != JobReturnCode.UNKNOWN:
                self._handles.pop(job_id)

    def _terminal_message(self, message):
        data = message.payload
        job_id = data.get("job_id") if isinstance(data, Shareable) else None
        with self._lock:
            handle = self._handles.get(job_id)
        if handle is None or message.get_header(MessageHeaderKey.ORIGIN) != f"server.{job_id}":
            return new_cell_message({MessageHeaderKey.RETURN_CODE: CellReturnCode.AUTHENTICATION_ERROR})
        if data.get(STATUS) not in (DONE, ERROR):
            return new_cell_message({MessageHeaderKey.RETURN_CODE: CellReturnCode.INVALID_REQUEST})
        handle.notify_terminal(data[STATUS])
        return new_cell_message({MessageHeaderKey.RETURN_CODE: CellReturnCode.OK}, make_reply(ReturnCode.OK))

    def register(self, job_id, engine, create_handle):
        """Create a logical handle with the shared probe and terminal protocol."""
        cell = engine.get_cell()
        with self._lock:
            previous = self._handles.get(job_id)
            if previous is not None and previous.poll() == JobReturnCode.UNKNOWN:
                raise RuntimeError("logical task-scoped job is already active")
            if self._terminal_cell is not cell:
                cell.register_request_cb(
                    channel=CellChannel.AUX_COMMUNICATION, topic=TERMINAL_TOPIC, cb=self._terminal_message
                )
                self._terminal_cell = cell

            def probe():
                with engine.new_context() as ctx:
                    # A CP context has an empty, public-sticky run ID. Override it
                    # only for this probe; changing the CP sticker would leak one
                    # job's identity into concurrent job probes.
                    ctx.put(FLContextKey.CURRENT_RUN, job_id, private=False, sticky=False)
                    ctx.put(FLContextKey.CURRENT_JOB_ID, job_id, private=False, sticky=False)
                    replies = engine.aux_runner.send_aux_request(
                        targets=[AuxMsgTarget("server", f"server.{job_id}", job_scoped=False)],
                        topic=PROBE_TOPIC,
                        request=Shareable(),
                        timeout=self.probe_timeout,
                        fl_ctx=ctx,
                        optional=True,
                    )
                reply = replies.get("server") if isinstance(replies, dict) else None
                if not isinstance(reply, Shareable) or reply.get_return_code() != ReturnCode.OK:
                    return None
                return reply

            handle = create_handle(probe, self.poll_interval, self.communication_timeout)
            self._handles[job_id] = handle
            return handle


class TaskScopedJobLauncherMixin:
    """Add opt-in logical task scope while delegating physical launches to the existing launcher."""

    def __init__(
        self,
        *,
        task_scoped=False,
        task_phased=False,
        task_probe_interval=2.0,
        task_probe_timeout=5.0,
        task_communication_timeout=120.0,
        **kwargs,
    ):
        if not isinstance(task_scoped, bool):
            raise ValueError("task_scoped must be bool")
        if not isinstance(task_phased, bool) or (task_phased and not task_scoped):
            raise ValueError("task_phased requires task_scoped=True and must be bool")
        self.task_scoped = task_scoped
        self.task_phased = task_phased
        self._task_scope = (
            TaskScopedJobRegistry(task_probe_interval, task_probe_timeout, task_communication_timeout)
            if task_scoped
            else None
        )
        super().__init__(**kwargs)

    def handle_event(self, event_type, fl_ctx):
        super().handle_event(event_type, fl_ctx)
        if self._task_scope and event_type == EventType.JOB_COMPLETED:
            self._task_scope.release(fl_ctx.get_prop(FLContextKey.CURRENT_JOB_ID))

    def _prepare_task_scoped_job(self, job_meta, fl_ctx):
        job_id = job_meta.get(JobConstants.JOB_ID)
        workspace = fl_ctx.get_prop(FLContextKey.WORKSPACE_OBJECT)
        if not isinstance(job_id, str) or not job_id or workspace is None:
            raise RuntimeError("task-scoped launch requires a job ID and workspace")
        return job_id, workspace.get_run_dir(job_id)

    def _task_scope_allocation_details(self, handle):
        return {}

    def launch_job(self, job_meta, fl_ctx):
        if not self.task_scoped:
            return super().launch_job(job_meta, fl_ctx)
        job_id, run_dir = self._prepare_task_scoped_job(job_meta, fl_ctx)

        def create_handle(probe, poll_interval, communication_timeout):
            def launch_attempt(attempt, directory, phase=None):
                return launch_task_scope_worker(
                    lambda: super(TaskScopedJobLauncherMixin, self).launch_job(job_meta, fl_ctx),
                    fl_ctx,
                    attempt,
                    directory,
                    phase,
                )

            return TaskScopedJobHandle(
                job_id,
                run_dir,
                probe,
                self.logger,
                poll_interval,
                communication_timeout,
                launch_attempt=launch_attempt,
                allocation_details=self._task_scope_allocation_details,
                phased=self.task_phased,
            )

        return self._task_scope.register(job_id, fl_ctx.get_engine(), create_handle)
