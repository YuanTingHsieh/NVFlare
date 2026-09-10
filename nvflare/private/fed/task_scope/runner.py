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
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.shareable import ReservedHeaderKey, Shareable, make_reply
from nvflare.fuel.utils.fobs.decomposers.via_downloader import contains_lazy_download_ref
from nvflare.private.defs import SpecialTaskName
from nvflare.private.fed.client.client_app_runner import ClientAppRunner
from nvflare.private.fed.client.client_runner import ClientRunner
from nvflare.private.fed.task_scope.protocol import END_RUN, IDLE, TASK_COMPLETE


class TaskScopedClientRunner(ClientRunner):
    """Run one synchronous, eager task in a disposable CJ process.

    Every configured executor must declare ``supports_task_scoped_process = True``.
    This is an application contract: all executors, filters, and handlers must
    tolerate construction, START_RUN, and END_RUN on every incarnation, including
    an idle incarnation. State needed by a later task must be restored from the
    task or durable shared-workspace artifacts before execute() runs. END_RUN and
    workspace archival retain their existing per-process behavior; they do not
    represent completion of the logical federated job in this experiment.

    Aux tasks, asynchronous execution, lazy results, and components that rely on
    continuing process-local state are unsupported. A successful outcome records
    the existing result-submission ACK, not an additional durable server commit.
    """

    def run(self, app_root, args):
        args.task_scope_outcome = None
        self._task_scope_outcome = None
        self._task_scope_error = None
        for executor in self.task_router.task_table.values():
            if getattr(executor, "supports_task_scoped_process", False) is not True:
                raise RuntimeError(
                    f"{type(executor).__name__} must declare supports_task_scoped_process=True "
                    "and support per-incarnation START_RUN/END_RUN before using experimental task-scoped execution"
                )

        # Retain the standard initialization, END_RUN, and streaming cleanup.
        # ClientRunner.run logs and swallows _try_run exceptions, so the sentinel
        # below must prevent a failed incarnation from publishing a clean receipt.
        super().run(app_root, args)
        if self._task_scope_error is not None:
            raise RuntimeError("task-scoped client execution failed") from self._task_scope_error
        if self._run_abort_requested or self._task_scope_outcome is None:
            raise RuntimeError("task-scoped client aborted without a clean task outcome")
        args.task_scope_outcome = self._task_scope_outcome

    def _try_run(self):
        heartbeat_thread = threading.Thread(target=self._send_job_heartbeat, daemon=True)
        heartbeat_thread.start()
        try:
            if self.run_abort_signal.triggered:
                raise RuntimeError("task-scoped client was stopped before fetching a task")
            with self.engine.new_context() as fl_ctx:
                self._task_scope_outcome = self._run_one_task(fl_ctx)
        except BaseException as e:
            self._task_scope_error = e
            raise
        finally:
            # Only stop the run signal after the synchronous send/ACK has
            # completed. This also stops the inherited heartbeat loop on IDLE.
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

        self.log_info(fl_ctx, f"executing task-scoped assignment: name={task.name}, id={task.task_id}")
        result = self._process_task(task, fl_ctx)
        self.fire_event(EventType.BEFORE_SEND_TASK_RESULT, fl_ctx)
        if result.get_header(ReservedHeaderKey.PASS_THROUGH) or contains_lazy_download_ref(result):
            raise RuntimeError(
                "task-scoped execution requires eager task results; materialize lazy download references "
                "inside the executor before returning the Shareable"
            )

        submitted = self._send_task_result(result, task.task_id, fl_ctx)
        self.fire_event(EventType.AFTER_SEND_TASK_RESULT, fl_ctx)
        if not submitted:
            raise RuntimeError(f"task {task.task_id} did not receive a successful result-submission ACK")
        if result.get_return_code() != ReturnCode.OK:
            raise RuntimeError(f"task {task.task_id} returned failure code {result.get_return_code()}")
        if self._run_abort_requested:
            raise RuntimeError(f"task {task.task_id} was aborted")
        return {"status": TASK_COMPLETE, "task_id": task.task_id}

    def _handle_do_task(self, topic, request, fl_ctx):
        # Aux tasks could execute concurrently with the ordinary task and defeat
        # the N=1 lifecycle boundary. Only the standard task-pull path is supported.
        self.log_error(fl_ctx, "aux task execution is unsupported by experimental task-scoped execution")
        return make_reply(ReturnCode.TASK_UNKNOWN)


class TaskScopedClientAppRunner(ClientAppRunner):
    CLIENT_RUNNER_CLASS = TaskScopedClientRunner
