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
"""Opt-in server probe for the experimental one-task client lifecycle."""

import threading
import time

from nvflare.apis.controller_spec import ClientTask
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_constant import FLContextKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.impl.bcast_manager import BcastTaskManager
from nvflare.apis.impl.send_manager import SendTaskManager
from nvflare.apis.impl.task_manager import TaskCheckStatus
from nvflare.apis.impl.wf_comm_server import _TASK_KEY_MANAGER, WFCommServer
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey
from nvflare.fuel.f3.cellnet.defs import ReturnCode as CellReturnCode
from nvflare.private.aux_runner import AuxMsgTarget
from nvflare.private.defs import CellChannel, CellMessageHeaderKeys, new_cell_message
from nvflare.private.fed.task_scope.protocol import (
    DONE,
    ERROR,
    PROBE_TOPIC,
    READY,
    REASON,
    STATUS,
    TASK_NAME,
    TASK_TOKEN,
    TERMINAL_TOPIC,
    WAIT,
)


def _reply(status, reason="", task_name="", task_token=""):
    reply = make_reply(ReturnCode.OK)
    reply[STATUS] = status
    reply[REASON] = reason
    reply[TASK_NAME] = task_name
    reply[TASK_TOKEN] = task_token
    return reply


def _available_task(communicator, client, fl_ctx):
    """Read scheduler eligibility without assigning tasks or running callbacks.

    Only the exact built-in broadcast/send managers have pure check_task_send
    implementations. Relay managers mutate their scheduling state while checking
    eligibility, and custom managers have no purity contract.
    """
    if type(communicator) is not WFCommServer:
        return _reply(ERROR, "task-scoped execution supports only the built-in WFCommServer")
    with communicator._controller_lock:
        with communicator._task_lock:
            for task in communicator._tasks:
                if task.completion_status is not None:
                    continue
                manager = task.props.get(_TASK_KEY_MANAGER)
                if type(manager) not in (BcastTaskManager, SendTaskManager):
                    return _reply(ERROR, "task-scoped execution supports ordinary broadcast/send tasks only")
                if task.timeout and task.schedule_time is not None and time.time() - task.schedule_time >= task.timeout:
                    continue
                previous = task.last_client_task_map.get(client.name)
                if previous is not None and previous.result_received_time is None:
                    return _reply(READY, task_name=task.name, task_token=task.msg_root_id)
                candidate = previous if previous is not None else ClientTask(client, task)
                status = manager.check_task_send(candidate, fl_ctx)
                if status == TaskCheckStatus.SEND:
                    return _reply(READY, task_name=task.name, task_token=task.msg_root_id)
                if status == TaskCheckStatus.BLOCK:
                    return _reply(WAIT)
    return _reply(WAIT)


class TaskScopedServer(FLComponent):
    """Advertise pending broadcast/send work to CPs without allocating a CJ.

    Add this component to the server job configuration only when the experimental
    task-scoped client launcher is used. Availability is a hint: task timeout or
    another recipient can invalidate it while the launcher queues a submission.
    """

    def __init__(self, terminal_timeout: float = 5.0):
        super().__init__()
        if (
            isinstance(terminal_timeout, bool)
            or not isinstance(terminal_timeout, (float, int))
            or terminal_timeout <= 0
        ):
            raise ValueError("terminal_timeout must be positive")
        self.terminal_timeout = terminal_timeout
        self._engine = None
        self._job_id = None
        self._ending = False
        self._clients = {}
        self._lock = threading.Lock()

    def handle_event(self, event_type, fl_ctx):
        if event_type == EventType.START_RUN:
            self._engine = fl_ctx.get_engine()
            self._job_id = fl_ctx.get_job_id()
            # The generic AUX dispatcher exposes caller-supplied peer properties,
            # but not the authenticated token. A topic-specific handler binds the
            # requested site to that token before inspecting its eligible tasks.
            self._engine.get_cell().register_request_cb(
                channel=CellChannel.AUX_COMMUNICATION, topic=PROBE_TOPIC, cb=self._handle_probe_message
            )
        elif event_type == EventType.ABOUT_TO_END_RUN:
            with self._lock:
                self._ending = True
        elif event_type == EventType.END_RUN:
            self._notify_parents(fl_ctx)

    def _handle_probe_message(self, request):
        with self._engine.new_context() as fl_ctx:
            server = self._engine.server
            error = server.authentication_check(request, server.server_state.aux_communicate(fl_ctx))
            token = request.get_header(CellMessageHeaderKeys.TOKEN)
            client = next((c for c in self._engine.get_clients() if token and c.token == token), None)
            if error or client is None:
                return new_cell_message(
                    {MessageHeaderKey.RETURN_CODE: CellReturnCode.AUTHENTICATION_ERROR},
                    _reply(ERROR, "unauthenticated task availability probe"),
                )
            data = request.payload
            props = data.get_peer_props() if isinstance(data, Shareable) else None
            peer = FLContext()
            if isinstance(props, dict):
                peer.set_public_props(props)
            if peer.get_identity_name() != client.name or peer.get_job_id() != self._job_id:
                reply = _reply(ERROR, "probe identity or job does not match the authenticated participant")
            else:
                fl_ctx.set_peer_context(peer)
                with self._lock:
                    self._clients[client.name] = client
                reply = self._probe(client, fl_ctx)
            return new_cell_message({MessageHeaderKey.RETURN_CODE: CellReturnCode.OK}, reply)

    def _probe(self, client, fl_ctx):
        with self._lock:
            if self._ending:
                return _reply(DONE)
        runner = fl_ctx.get_prop(FLContextKey.RUNNER)
        if runner is None:
            return _reply(WAIT)
        with runner.wf_lock:
            if runner.status == "done":
                return _reply(DONE)
            if runner.status != "started" or runner.current_wf is None:
                return _reply(WAIT)
            communicator = runner.current_wf.controller.communicator
            return _available_task(communicator, client, fl_ctx)

    def _notify_parents(self, fl_ctx):
        # A terminal message must reach the persistent CP: the ordinary END_RUN
        # targets only client job cells, which are absent during the idle gap.
        with self._lock:
            self._ending = True
            clients = list(self._clients.values())
        if not clients:
            return
        terminal = _reply(DONE)
        terminal["job_id"] = self._job_id
        replies = self._engine.run_manager.aux_runner.send_aux_request(
            targets=[AuxMsgTarget(c.name, c.get_fqcn(), job_scoped=False) for c in clients],
            topic=TERMINAL_TOPIC,
            request=terminal,
            timeout=self.terminal_timeout,
            fl_ctx=fl_ctx,
            optional=True,
        )
        for client in clients:
            reply = replies.get(client.name)
            if not isinstance(reply, Shareable) or reply.get_return_code() != ReturnCode.OK:
                self.log_error(fl_ctx, f"task-scoped terminal notice was not acknowledged by {client.name}")
