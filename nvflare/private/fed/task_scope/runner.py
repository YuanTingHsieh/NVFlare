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
"""Experimental client incarnation that submits at most one ordinary task."""

import threading

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import ReservedHeaderKey, Shareable, make_reply
from nvflare.fuel.utils.argument_utils import parse_vars
from nvflare.fuel.utils.fobs.decomposers.via_downloader import contains_lazy_download_ref
from nvflare.private.defs import SpecialTaskName
from nvflare.private.fed.app.fl_conf import create_privacy_manager
from nvflare.private.fed.client.client_app_runner import ClientAppRunner
from nvflare.private.fed.client.client_engine_executor_spec import TaskAssignment
from nvflare.private.fed.client.client_runner import ClientRunner
from nvflare.private.fed.task_scope.artifacts import read_artifact, write_artifact
from nvflare.private.fed.task_scope.config import PUBLICATION_ACK_PROP, TaskScopeTransferConfigurator
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
    RESULT_READY,
    TASK_COMPLETE,
)


class TaskScopedClientRunner(ClientRunner):
    """Run one synchronous, eager task in a disposable CJ process.

    Every configured executor must declare ``supports_task_scoped_process = True``.
    The application graph is constructed only for compute; pull uses the framework
    transfer graph and push adds only explicitly registered publication components.
    State needed by a later task must be restored from the task or durable
    shared-workspace artifacts before execute() runs. END_RUN events do not
    represent completion of the logical federated job. The phased variant runs
    pull, compute, and push in separate incarnations, with inputs and fully filtered
    results persisted between them. Only push archives workspace results; compute
    never sends its task result to the server. Send events run in the CPU push
    process and observe the real result-submission ACK.

    Aux DO_TASK RPCs, work that outlives execute(), unresolved lazy results, and
    components that rely on continuing process-local state are unsupported. A
    successful outcome records the existing result-submission ACK, not an
    additional durable server commit.
    """

    def run(self, app_root, args):
        args.task_scope_outcome = None
        self._task_scope_outcome = None
        self._task_scope_error = None
        options = parse_vars(getattr(args, "set", None))
        self._task_scope_phase = options.get(PHASE_OPTION)
        self._task_scope_directory = options.get(DIRECTORY_OPTION)
        self._task_scope_attempt = options.get(ATTEMPT_OPTION)
        if self._task_scope_phase is not None:
            if self._task_scope_phase not in (PULL, COMPUTE, PUSH):
                raise ValueError(f"invalid task-scope phase: {self._task_scope_phase}")
            if not self._task_scope_directory or not self._task_scope_attempt:
                raise ValueError("task-scope phase requires an attempt ID and artifact directory")
        for executor in self.task_router.task_table.values():
            if getattr(executor, "supports_task_scoped_process", False) is not True:
                raise RuntimeError(
                    f"{type(executor).__name__} must declare supports_task_scoped_process=True "
                    "and support per-incarnation START_RUN/END_RUN before using experimental task-scoped execution"
                )
            validate = getattr(executor, "validate_task_scoped_process", None)
            if callable(validate):
                error = validate()
                if error:
                    raise RuntimeError(f"{type(executor).__name__} is incompatible with task-scoped execution: {error}")

        # Retain the standard initialization, END_RUN, and streaming cleanup.
        # ClientRunner.run logs and swallows _try_run exceptions, so the sentinel
        # below must prevent a failed incarnation from publishing a clean receipt.
        super().run(app_root, args)
        if self._task_scope_error is not None:
            raise RuntimeError("task-scoped client execution failed") from self._task_scope_error
        if self._run_abort_requested or self._task_scope_outcome is None:
            raise RuntimeError("task-scoped client aborted without a clean task outcome")
        args.task_scope_outcome = self._task_scope_outcome

    def requires_materialized_task_result(self, task_name):
        """The compute runner persists and therefore locally consumes the concrete result."""
        return self._task_scope_phase == COMPUTE

    def _try_run(self):
        heartbeat_thread = threading.Thread(target=self._send_job_heartbeat, daemon=True)
        heartbeat_thread.start()
        try:
            if self.run_abort_signal.triggered:
                raise RuntimeError("task-scoped client was stopped before fetching a task")
            with self.engine.new_context() as fl_ctx:
                if self._task_scope_phase == COMPUTE:
                    self._task_scope_outcome = self._compute_task(fl_ctx)
                elif self._task_scope_phase == PUSH:
                    self._task_scope_outcome = self._push_result(fl_ctx)
                else:
                    self._task_scope_outcome = self._run_one_task(fl_ctx)
        except BaseException as e:
            self._task_scope_error = e
            raise
        finally:
            # Stop after the phase handoff (or send/ACK for push and baseline),
            # so teardown cannot interrupt its own result publication.
            self.run_abort_signal.trigger(True)
            heartbeat_thread.join(timeout=1.0)

    def _run_one_task(self, fl_ctx):
        task = self.engine.get_task_assignment(fl_ctx, self.get_task_timeout)
        if task is None:
            raise RuntimeError("task-scoped client did not receive a task or an explicit control response")
        if task.name == SpecialTaskName.END_RUN:
            return {"status": END_RUN, "task_id": None}
        if task.name == SpecialTaskName.TRY_AGAIN:
            return {"status": IDLE, "task_id": None}
        if not isinstance(task.task_id, str) or not task.task_id:
            raise RuntimeError("task-scoped task assignment requires a nonempty task ID")
        if not isinstance(task.data, Shareable):
            raise TypeError("task-scoped task data must be a Shareable")

        if self._task_scope_phase == PULL:
            peer_ctx = fl_ctx.get_peer_context()
            if not isinstance(peer_ctx, FLContext) or peer_ctx.get_job_id() != self.job_id:
                raise RuntimeError("task-scoped input requires the authenticated server's matching job context")
            task.data.set_peer_props(peer_ctx.get_all_public_props())
            task_ssid = fl_ctx.get_prop(FLContextKey.SSID)
            self._require_task_session(task_ssid)
            self._write_task_artifact("input", task, task.data, task_ssid)
            self.log_info(
                fl_ctx, f"task-scope input committed: attempt={self._task_scope_attempt}, task={task.task_id}"
            )
            return {"status": INPUT_READY, "task_id": task.task_id}

        self.log_info(fl_ctx, f"executing task-scoped assignment: name={task.name}, id={task.task_id}")
        result = self._process_task(task, fl_ctx)
        self.fire_event(EventType.BEFORE_SEND_TASK_RESULT, fl_ctx)
        self._require_eager_result(result)
        return self._submit_result(result, task.task_id, fl_ctx)

    @staticmethod
    def _require_eager_result(result):
        if result.get_header(ReservedHeaderKey.PASS_THROUGH) or contains_lazy_download_ref(result):
            raise RuntimeError(
                "task-scoped execution requires eager task results; materialize lazy download references "
                "inside the executor before returning the Shareable"
            )

    def _require_task_session(self, task_ssid):
        if not isinstance(task_ssid, str) or not task_ssid or task_ssid != self.engine.client.ssid:
            raise RuntimeError("task-scoped task session is missing or does not match the current client session")

    def _write_task_artifact(self, kind, task, data, task_ssid):
        write_artifact(
            self._task_scope_directory,
            self._task_scope_attempt,
            self.job_id,
            kind,
            task.name,
            task.task_id,
            data,
            task_ssid=task_ssid,
        )

    def _read_task_artifact(self, kind, fl_ctx):
        artifact = read_artifact(self._task_scope_directory, self._task_scope_attempt, self.job_id, kind)
        task_ssid = artifact["task_ssid"]
        self._require_task_session(task_ssid)
        # Normal pull_task sets this private, nonsticky property. Fresh compute
        # and push CJs must restore it without rebinding old work to a new session.
        fl_ctx.set_prop(FLContextKey.SSID, task_ssid, private=True, sticky=False)
        return TaskAssignment(artifact["task_name"], artifact["task_id"], artifact["data"]), task_ssid

    def _compute_task(self, fl_ctx):
        task, task_ssid = self._read_task_artifact("input", fl_ctx)
        peer_props = task.data.get_peer_props()
        if not isinstance(peer_props, dict):
            raise RuntimeError("task-scoped input is missing the server's public context")
        peer_ctx = FLContext()
        peer_ctx.set_public_props(peer_props)
        if peer_ctx.get_job_id() != self.job_id:
            raise RuntimeError("task-scoped input server context does not match the job")
        fl_ctx.set_peer_context(peer_ctx)
        result = self._process_task(task, fl_ctx)
        # Task data/result filters and execution events run here. Send events
        # belong to the later CPU push incarnation and must be CPU-compatible.
        self._require_eager_result(result)
        self._require_success(result, task.task_id)
        self._write_task_artifact("result", task, result, task_ssid)
        self.log_info(fl_ctx, f"task-scope result committed: attempt={self._task_scope_attempt}, task={task.task_id}")
        return {"status": RESULT_READY, "task_id": task.task_id}

    def _push_result(self, fl_ctx):
        task, _ = self._read_task_artifact("result", fl_ctx)
        result = task.data
        fl_ctx.set_prop(FLContextKey.TASK_NAME, task.name, private=True, sticky=False)
        fl_ctx.set_prop(FLContextKey.TASK_ID, task.task_id, private=True, sticky=False)
        fl_ctx.set_prop(FLContextKey.TASK_RESULT, result, private=True, sticky=False)
        self.fire_event(EventType.BEFORE_SEND_TASK_RESULT, fl_ctx)
        self._require_eager_result(result)
        return self._submit_result(result, task.task_id, fl_ctx)

    def _try_send_result_once(self, result, task_id, fl_ctx):
        if self._task_scope_phase == PUSH:
            # A permanent session error must not become False in the run manager
            # and enter the ordinary transient-send retry loop.
            self._require_task_session(fl_ctx.get_prop(FLContextKey.SSID))
        return super()._try_send_result_once(result, task_id, fl_ctx)

    def _submit_result(self, result, task_id, fl_ctx):
        submitted = self._send_task_result(result, task_id, fl_ctx)
        fl_ctx.set_prop(PUBLICATION_ACK_PROP, submitted is True, private=True, sticky=False)
        self.fire_event(EventType.AFTER_SEND_TASK_RESULT, fl_ctx)
        if not submitted:
            raise RuntimeError(f"task {task_id} did not receive a successful result-submission ACK")
        self._require_success(result, task_id)
        return {"status": TASK_COMPLETE, "task_id": task_id}

    def _require_success(self, result, task_id):
        if result.get_return_code() != ReturnCode.OK:
            raise RuntimeError(f"task {task_id} returned failure code {result.get_return_code()}")
        if self._run_abort_requested:
            raise RuntimeError(f"task {task_id} was aborted")

    def _handle_do_task(self, topic, request, fl_ctx):
        # Aux tasks could execute concurrently with the ordinary task and defeat
        # the N=1 lifecycle boundary. Only the standard task-pull path is supported.
        self.log_error(fl_ctx, "aux task execution is unsupported by experimental task-scoped execution")
        return make_reply(ReturnCode.TASK_UNKNOWN)


class TaskScopedClientAppRunner(ClientAppRunner):
    CLIENT_RUNNER_CLASS = TaskScopedClientRunner

    def create_configurator(self, workspace_obj, config_file_name, app_root, args, kv_list):
        phase = parse_vars(kv_list).get(PHASE_OPTION)
        if phase in (PULL, PUSH):
            return TaskScopeTransferConfigurator(
                workspace_obj=workspace_obj,
                config_file_name=config_file_name,
                app_root=app_root,
                args=args,
                kv_list=kv_list,
                include_publication_components=phase == PUSH,
            )
        return super().create_configurator(workspace_obj, config_file_name, app_root, args, kv_list)

    def create_privacy_manager(self, workspace, conf):
        if isinstance(conf, TaskScopeTransferConfigurator):
            # Scope names remain available for protocol checks, but site filters and
            # their dependencies are constructed only in compute.
            return create_privacy_manager(workspace, names_only=True)
        return super().create_privacy_manager(workspace, conf)
