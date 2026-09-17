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
"""Federation adapter for deployment-owned task acquisition/publication."""

import threading
import time
import uuid

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_constant import FLContextKey, JobConstants, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.impl.bcast_manager import BcastTaskManager
from nvflare.apis.impl.send_manager import SendTaskManager
from nvflare.apis.impl.wf_comm_server import _TASK_KEY_MANAGER, WFCommServer
from nvflare.apis.job_launcher_spec import JobHandleSpec, JobReturnCode
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey
from nvflare.fuel.f3.cellnet.defs import ReturnCode as CellReturnCode
from nvflare.private.aux_runner import AuxMsgTarget
from nvflare.private.defs import CellChannel, CellMessageHeaderKeys, SpecialTaskName, new_cell_message
from nvflare.private.fed.deployment_supervisor.config import load_worker_routes
from nvflare.private.fed.deployment_supervisor.protocol import (
    ACK,
    ACQUIRE_TOPIC,
    ATTEMPT,
    DATA,
    END_RUN,
    ERROR,
    PUBLISH_TOPIC,
    REASON,
    STATUS,
    TASK,
    TASK_ID,
    TASK_NAME,
    WAIT,
)
from nvflare.private.fed.deployment_supervisor.service import TaskAssignment
from nvflare.private.fed.deployment_supervisor.supervisor import DeploymentTaskSupervisor


def _reply(status, **values):
    result = make_reply(ReturnCode.OK)
    result[STATUS] = status
    result.update(values)
    return result


class DeploymentTaskService(FLComponent):
    """SJ adapter: existing Controllers remain planners; this service is A's task owner."""

    def __init__(self):
        super().__init__()
        self._engine = None
        self._job_id = None
        self._attempts = {}
        self._lock = threading.RLock()

    def handle_event(self, event_type, fl_ctx):
        if event_type == EventType.START_RUN:
            self.start(fl_ctx.get_engine(), fl_ctx.get_job_id())

    def start(self, engine, job_id):
        """Bind the service before a workflow can make work available."""
        with self._lock:
            if self._engine is not None:
                if self._engine is engine and self._job_id == job_id:
                    return
                raise RuntimeError("deployment task service is already bound to another run")
            self._engine = engine
            self._job_id = job_id
            cell = engine.get_cell()
            # Cell.__getattr__ selects the streaming registration path from the
            # channel keyword. Positional arguments bypass that selection and
            # register only on CoreCell, while AUX requests arrive through the
            # stream/blob path and fall through to AuxRunner's wildcard.
            cell.register_request_cb(
                channel=CellChannel.AUX_COMMUNICATION,
                topic=ACQUIRE_TOPIC,
                cb=self._handle_acquire,
            )
            cell.register_request_cb(
                channel=CellChannel.AUX_COMMUNICATION,
                topic=PUBLISH_TOPIC,
                cb=self._handle_publish,
            )

    def _authenticated(self, request):
        with self._engine.new_context() as fl_ctx:
            server = self._engine.server
            error = server.authentication_check(request, server.server_state.aux_communicate(fl_ctx))
            token = request.get_header(CellMessageHeaderKeys.TOKEN)
            client = next((c for c in self._engine.get_clients() if token and c.token == token), None)
            if error or client is None:
                return None, None, _reply(ERROR, reason="unauthenticated deployment task request")
            payload = request.payload
            props = payload.get_peer_props() if isinstance(payload, Shareable) else None
            peer = FLContext()
            if isinstance(props, dict):
                peer.set_public_props(props)
            if peer.get_identity_name() != client.name or peer.get_job_id() != self._job_id:
                return None, None, _reply(ERROR, reason="deployment task identity or job mismatch")
            fl_ctx.set_peer_context(peer)
            return client, fl_ctx.clone(), None

    @staticmethod
    def _message(reply, return_code=CellReturnCode.OK):
        return new_cell_message({MessageHeaderKey.RETURN_CODE: return_code}, reply)

    @staticmethod
    def _communicator(runner):
        with runner.wf_lock:
            if runner.status != "started" or runner.current_wf is None:
                return None
            communicator = runner.current_wf.controller.communicator
            if type(communicator) is not WFCommServer:
                raise RuntimeError("deployment task service supports only built-in WFCommServer")
            with communicator._task_lock:
                for task in communicator._tasks:
                    manager = task.props.get(_TASK_KEY_MANAGER)
                    if task.completion_status is None and type(manager) not in (BcastTaskManager, SendTaskManager):
                        raise RuntimeError("deployment task service supports ordinary broadcast/send tasks only")
            return communicator

    def _handle_acquire(self, request):
        client, fl_ctx, error = self._authenticated(request)
        if error is not None:
            return self._message(error, CellReturnCode.AUTHENTICATION_ERROR)
        runner = fl_ctx.get_prop(FLContextKey.RUNNER)
        if runner is None:
            return self._message(_reply(WAIT))
        try:
            self._communicator(runner)
            task_name, task_id, data = runner.process_task_request(client, fl_ctx)
            if task_name == SpecialTaskName.END_RUN:
                return self._message(_reply(END_RUN))
            if not task_name or task_name == SpecialTaskName.TRY_AGAIN:
                return self._message(_reply(WAIT))
            if not isinstance(task_id, str) or not task_id or not isinstance(data, Shareable):
                return self._message(_reply(ERROR, reason="server produced an invalid task assignment"))
            with self._lock:
                key = (client.name, task_id)
                attempt = self._attempts.setdefault(key, uuid.uuid4().hex)
            return self._message(_reply(TASK, **{TASK_NAME: task_name, TASK_ID: task_id, ATTEMPT: attempt, DATA: data}))
        except Exception as e:
            self.log_exception(fl_ctx, f"deployment task acquisition failed: {e}")
            return self._message(_reply(ERROR, reason=str(e)))

    def _handle_publish(self, request):
        client, fl_ctx, error = self._authenticated(request)
        if error is not None:
            return self._message(error, CellReturnCode.AUTHENTICATION_ERROR)
        payload = request.payload
        task_name = payload.get(TASK_NAME) if isinstance(payload, Shareable) else None
        task_id = payload.get(TASK_ID) if isinstance(payload, Shareable) else None
        attempt = payload.get(ATTEMPT) if isinstance(payload, Shareable) else None
        result = payload.get(DATA) if isinstance(payload, Shareable) else None
        with self._lock:
            expected = self._attempts.get((client.name, task_id))
        if not expected or attempt != expected or not isinstance(result, Shareable):
            return self._message(_reply(ERROR, reason="stale or invalid deployment task publication"))
        runner = fl_ctx.get_prop(FLContextKey.RUNNER)
        try:
            communicator = self._communicator(runner)
            with communicator._task_lock:
                client_task = communicator._client_task_map.get(task_id)
                if (
                    client_task is None
                    or client_task.client.name != client.name
                    or client_task.task.name != task_name
                    or client_task.result_received_time is not None
                ):
                    return self._message(_reply(ERROR, reason="task publication is not currently admissible"))
            runner.process_submission(client, task_name, task_id, result, fl_ctx)
            with communicator._task_lock:
                client_task = communicator._client_task_map.get(task_id)
                accepted = client_task is not None and client_task.result_received_time is not None
            if not accepted:
                return self._message(_reply(ERROR, reason="server did not acknowledge task publication"))
            with self._lock:
                self._attempts.pop((client.name, task_id), None)
            return self._message(_reply(ACK))
        except Exception as e:
            self.log_exception(fl_ctx, f"deployment task publication failed: {e}")
            return self._message(_reply(ERROR, reason=str(e)))


class FederatedTaskServiceClient:
    """CP-owned authenticated client for one server deployment task service."""

    def __init__(self, engine, job_id, client_name, request_timeout=10.0, publication_delay=0.0):
        self.engine = engine
        self.job_id = job_id
        self.client_name = client_name
        self.request_timeout = request_timeout
        self.publication_delay = publication_delay
        self.terminal = False
        self.last_error = None

    def _request(self, topic, request):
        with self.engine.new_context() as fl_ctx:
            fl_ctx.put(FLContextKey.CURRENT_RUN, self.job_id, private=False, sticky=False)
            fl_ctx.put(FLContextKey.CURRENT_JOB_ID, self.job_id, private=False, sticky=False)
            replies = self.engine.aux_runner.send_aux_request(
                targets=[AuxMsgTarget("server", f"server.{self.job_id}", job_scoped=False)],
                topic=topic,
                request=request,
                timeout=self.request_timeout,
                fl_ctx=fl_ctx,
                optional=True,
            )
        reply = replies.get("server") if isinstance(replies, dict) else None
        if not isinstance(reply, Shareable) or reply.get_return_code() != ReturnCode.OK:
            self.last_error = "deployment task service unavailable"
            return None
        self.last_error = None
        return reply

    def acquire(self, client_name):
        if client_name != self.client_name:
            raise ValueError("deployment task client identity changed")
        reply = self._request(ACQUIRE_TOPIC, Shareable())
        if reply is None:
            return None
        status = reply.get(STATUS)
        if status == END_RUN:
            self.terminal = True
            return None
        if status == WAIT:
            return None
        if status == ERROR:
            raise RuntimeError(reply.get(REASON, "deployment task acquisition rejected"))
        if status != TASK:
            raise RuntimeError("invalid deployment task acquisition response")
        data = reply.get(DATA)
        if not isinstance(data, Shareable):
            raise RuntimeError("deployment task acquisition omitted task data")
        return TaskAssignment(
            job_id=self.job_id,
            task_id=reply.get(TASK_ID),
            task_name=reply.get(TASK_NAME),
            client_name=self.client_name,
            attempt=reply.get(ATTEMPT),
            data=data,
        )

    @staticmethod
    def _publication(assignment, result):
        request = Shareable()
        request[TASK_NAME] = assignment.task_name
        request[TASK_ID] = assignment.task_id
        request[ATTEMPT] = assignment.attempt
        request[DATA] = result
        return request

    def publish(self, assignment, result):
        if self.publication_delay:
            time.sleep(self.publication_delay)
        reply = self._request(PUBLISH_TOPIC, self._publication(assignment, result))
        if reply is None or reply.get(STATUS) != ACK:
            reason = reply.get(REASON) if isinstance(reply, Shareable) else "missing publication ACK"
            raise RuntimeError(reason)

    def fail(self, assignment, reason):
        failure = make_reply(ReturnCode.EXECUTION_EXCEPTION)
        failure[REASON] = str(reason)
        reply = self._request(PUBLISH_TOPIC, self._publication(assignment, failure))
        if reply is None or reply.get(STATUS) != ACK:
            self.last_error = reply.get(REASON) if isinstance(reply, Shareable) else "missing failure ACK"


class DeploymentFederationJobHandle(JobHandleSpec):
    def __init__(self, supervisor, service, client_name, poll_interval, communication_timeout):
        self.supervisor = supervisor
        self.service = service
        self.client_name = client_name
        self.poll_interval = poll_interval
        self.communication_timeout = communication_timeout
        self.result = JobReturnCode.UNKNOWN
        self._cancel = threading.Event()
        self._wait_lock = threading.Lock()
        self.failure_reason = None

    def request_abort(self):
        self._cancel.set()
        self.supervisor.cancel()

    def terminate(self):
        self.request_abort()

    def poll(self):
        return self.result

    def wait(self):
        with self._wait_lock:
            if self.result != JobReturnCode.UNKNOWN:
                return self.result
            last_response = time.monotonic()
            try:
                while not self._cancel.is_set():
                    worked = self.supervisor.run_once(self.client_name)
                    if self.service.terminal:
                        self.result = JobReturnCode.SUCCESS
                        return self.result
                    if worked:
                        last_response = time.monotonic()
                    elif self.service.last_error:
                        if time.monotonic() - last_response >= self.communication_timeout:
                            raise RuntimeError(self.service.last_error)
                    else:
                        last_response = time.monotonic()
                    self._cancel.wait(self.poll_interval)
                self.result = JobReturnCode.ABORTED
            except Exception as e:
                self.supervisor.cancel()
                self.failure_reason = str(e)
                self.result = JobReturnCode.EXECUTION_ERROR
            return self.result


class DeploymentSupervisorLauncherMixin:
    """Opt-in logical job owned by CP while physical workers use the launcher."""

    def __init__(
        self,
        *,
        deployment_supervised=False,
        deployment_poll_interval=1.0,
        deployment_request_timeout=10.0,
        deployment_communication_timeout=120.0,
        deployment_worker_timeout=3600.0,
        deployment_publication_delay=0.0,
        **kwargs,
    ):
        if deployment_publication_delay < 0:
            raise ValueError("deployment_publication_delay must be nonnegative")
        self.deployment_supervised = deployment_supervised
        self.deployment_poll_interval = deployment_poll_interval
        self.deployment_request_timeout = deployment_request_timeout
        self.deployment_communication_timeout = deployment_communication_timeout
        self.deployment_worker_timeout = deployment_worker_timeout
        self.deployment_publication_delay = deployment_publication_delay
        super().__init__(**kwargs)

    def launch_job(self, job_meta, fl_ctx):
        if not self.deployment_supervised:
            return super().launch_job(job_meta, fl_ctx)
        from nvflare.private.fed.deployment_supervisor.slurm import SlurmTaskWorkerLauncher

        job_id = job_meta.get(JobConstants.JOB_ID)
        workspace = fl_ctx.get_prop(FLContextKey.WORKSPACE_OBJECT)
        engine = fl_ctx.get_engine()
        client_name = fl_ctx.get_identity_name()
        routes = load_worker_routes(workspace, job_id)
        service = FederatedTaskServiceClient(
            engine,
            job_id,
            client_name,
            request_timeout=self.deployment_request_timeout,
            publication_delay=self.deployment_publication_delay,
        )
        physical_launcher = SlurmTaskWorkerLauncher(self, job_meta, fl_ctx)
        supervisor = DeploymentTaskSupervisor(
            service=service,
            workspace=f"{workspace.get_run_dir(job_id)}/.deployment_supervisor",
            routes=routes,
            launcher=physical_launcher,
            worker_timeout=self.deployment_worker_timeout,
            poll_interval=min(self.deployment_poll_interval, 0.2),
        )
        return DeploymentFederationJobHandle(
            supervisor,
            service,
            client_name,
            self.deployment_poll_interval,
            self.deployment_communication_timeout,
        )
