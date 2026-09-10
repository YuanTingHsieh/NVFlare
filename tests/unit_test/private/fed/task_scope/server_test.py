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

import copy
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nvflare.apis.client import Client
from nvflare.apis.controller_spec import ClientTask, SendOrder, Task, TaskCompletionStatus
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey, ReservedKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.impl.bcast_manager import BcastForeverTaskManager, BcastTaskManager
from nvflare.apis.impl.send_manager import SendTaskManager
from nvflare.apis.impl.seq_relay_manager import SequentialRelayTaskManager
from nvflare.apis.impl.wf_comm_server import _TASK_KEY_MANAGER, WFCommServer
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey
from nvflare.fuel.f3.cellnet.defs import ReturnCode as CellReturnCode
from nvflare.private.defs import CellChannel, CellMessageHeaderKeys, new_cell_message
from nvflare.private.fed.task_scope.protocol import (
    DONE,
    ERROR,
    PROBE_TOPIC,
    READY,
    STATUS,
    TASK_NAME,
    TASK_TOKEN,
    TERMINAL_TOPIC,
    WAIT,
)
from nvflare.private.fed.task_scope.server import TaskScopedServer, _available_task


def _task(name="train", targets=None):
    task = Task(name, Shareable({"model": [1, 2]}), before_task_sent_cb=Mock(), after_task_sent_cb=Mock())
    task.targets = targets if targets is not None else ["site-1"]
    task.props[_TASK_KEY_MANAGER] = BcastTaskManager(task)
    task.schedule_time = time.time()
    return task


def _probe(tasks, site="site-1"):
    communicator = WFCommServer()
    communicator._tasks = tasks
    return _available_task(communicator, Client(site, "token"), FLContext()), communicator


def test_repeated_broadcast_probes_do_not_assign_copy_filter_or_invoke_callbacks():
    task = _task()
    original_data = copy.deepcopy(task.data)
    original_props = task.props.copy()
    for _ in range(3):
        reply, communicator = _probe([task])
        assert reply[STATUS] == READY
        assert reply[TASK_NAME] == "train"
        assert reply[TASK_TOKEN] == task.msg_root_id
        assert communicator._client_task_map == {}
    assert task.client_tasks == []
    assert task.last_client_task_map == {}
    assert task.data == original_data
    assert task.props == original_props
    assert not hasattr(task, "_broadcast_data")
    task.before_task_sent_cb.assert_not_called()
    task.after_task_sent_cb.assert_not_called()


def test_probe_waits_for_broadcast_to_other_site_and_finished_task():
    task = _task(targets=["site-2"])
    assert _probe([task])[0][STATUS] == WAIT
    task.targets.append("site-1")
    task.completion_status = TaskCompletionStatus.OK
    assert _probe([task])[0][STATUS] == WAIT


def test_accepted_result_is_not_advertised_again_but_next_task_is():
    task = _task()
    previous = ClientTask(Client("site-1", "token"), task)
    previous.result_received_time = time.time()
    task.last_client_task_map["site-1"] = previous
    task.client_tasks.append(previous)
    assert _probe([task])[0][STATUS] == WAIT
    reply, _ = _probe([task, _task("validate")])
    assert (reply[STATUS], reply[TASK_NAME]) == (READY, "validate")
    assert reply[TASK_TOKEN] != task.msg_root_id


def test_unreturned_task_remains_available_without_incrementing_send_count():
    task = _task()
    previous = ClientTask(Client("site-1", "token"), task)
    previous.task_sent_time = time.time()
    previous.task_send_count = 1
    task.last_client_task_map["site-1"] = previous
    task.client_tasks.append(previous)
    assert _probe([task])[0][STATUS] == READY
    assert previous.task_send_count == 1
    assert task.client_tasks == [previous]


def test_sequential_send_blocks_later_task_until_site_is_eligible():
    task = _task(targets=["site-2", "site-1"])
    task.props[_TASK_KEY_MANAGER] = SendTaskManager(task, SendOrder.SEQUENTIAL, 30)
    assert _probe([task, _task("validate")])[0][STATUS] == WAIT
    task.create_time -= 31
    reply, _ = _probe([task, _task("validate")])
    assert (reply[STATUS], reply[TASK_NAME]) == (READY, "train")
    assert not task.client_tasks


def test_send_already_assigned_elsewhere_does_not_advertise_for_this_site():
    task = _task(targets=["site-1", "site-2"])
    task.props[_TASK_KEY_MANAGER] = SendTaskManager(task, SendOrder.ANY, 0)
    other = ClientTask(Client("site-2", "other"), task)
    other.task_sent_time = time.time()
    task.client_tasks.append(other)
    task.last_client_task_map["site-2"] = other
    assert _probe([task])[0][STATUS] == WAIT


def test_expired_task_does_not_allocate_worker_before_monitor_removes_it():
    task = _task()
    task.timeout = 1
    task.schedule_time -= 2
    assert _probe([task])[0][STATUS] == WAIT


@pytest.mark.parametrize("kind", ["forever", "relay", "custom"])
def test_unsupported_managers_are_rejected_without_mutating_state(kind):
    task = _task()
    if kind == "forever":
        manager = BcastForeverTaskManager()
    elif kind == "relay":
        manager = SequentialRelayTaskManager(task, 1, 1, True)
    else:

        class CustomManager(BcastTaskManager):
            pass

        manager = CustomManager(task)
    manager.check_task_send = Mock(side_effect=AssertionError("must not check stateful or custom manager"))
    task.props[_TASK_KEY_MANAGER] = manager
    props_before = copy.copy(task.props)
    assert _probe([task])[0][STATUS] == ERROR
    assert task.props == props_before
    manager.check_task_send.assert_not_called()


def test_custom_communicator_has_no_assumed_probe_contract():
    assert _available_task(Mock(), Client("site-1", "token"), FLContext())[STATUS] == ERROR


def _server():
    clients = [Client("site-1", "token-1"), Client("site-2", "token-2")]
    engine = Mock()
    engine.get_clients.return_value = clients
    engine.server.authentication_check.return_value = None

    def context():
        fl_ctx = FLContext()
        fl_ctx.set_prop(ReservedKey.ENGINE, engine, private=True, sticky=False)
        fl_ctx.set_prop(ReservedKey.RUN_NUM, "job-1", private=False, sticky=False)
        return fl_ctx

    engine.new_context.side_effect = context
    fl_ctx = context()
    server = TaskScopedServer()
    server.handle_event(EventType.START_RUN, fl_ctx)
    return server, engine, fl_ctx, clients


def _request(token="token-1", site="site-1", job="job-1"):
    peer = FLContext()
    peer.set_prop(ReservedKey.RUN_NUM, job, private=False, sticky=False)
    peer.set_prop(ReservedKey.IDENTITY_NAME, site, private=False, sticky=False)
    payload = Shareable()
    payload.set_peer_props(peer.get_all_public_props())
    return new_cell_message({CellMessageHeaderKeys.TOKEN: token}, payload)


def test_registration_uses_specific_cell_handler_for_authenticated_identity_binding():
    server, engine, _, _ = _server()
    engine.get_cell.return_value.register_request_cb.assert_called_once_with(
        channel=CellChannel.AUX_COMMUNICATION, topic=PROBE_TOPIC, cb=server._handle_probe_message
    )


@pytest.mark.parametrize("token", [None, "unknown"])
def test_unknown_authentication_token_cannot_probe(token):
    server, _, _, _ = _server()
    reply = server._handle_probe_message(_request(token=token))
    assert reply.get_header(MessageHeaderKey.RETURN_CODE) == CellReturnCode.AUTHENTICATION_ERROR
    assert server._clients == {}


@pytest.mark.parametrize("site,job", [("site-2", "job-1"), ("site-1", "another-job")])
def test_probe_cannot_choose_another_site_or_job_in_peer_properties(site, job):
    server, _, _, _ = _server()
    reply = server._handle_probe_message(_request(site=site, job=job))
    assert reply.payload[STATUS] == ERROR
    assert server._clients == {}


def test_server_authentication_failure_is_preserved():
    server, engine, _, _ = _server()
    engine.server.authentication_check.return_value = "wrong session"
    reply = server._handle_probe_message(_request())
    assert reply.get_header(MessageHeaderKey.RETURN_CODE) == CellReturnCode.AUTHENTICATION_ERROR
    assert server._clients == {}


def test_valid_participant_can_wait_during_startup_without_creating_task_assignment():
    server, _, _, clients = _server()
    reply = server._handle_probe_message(_request())
    assert reply.payload[STATUS] == WAIT
    assert server._clients == {"site-1": clients[0]}


def test_probe_recognizes_running_work_and_terminal_transition():
    server, _, fl_ctx, clients = _server()
    _, communicator = _probe([_task()])
    runner = SimpleNamespace(
        status="started",
        wf_lock=threading.RLock(),
        current_wf=SimpleNamespace(controller=SimpleNamespace(communicator=communicator)),
    )
    fl_ctx.set_prop(FLContextKey.RUNNER, runner, private=True, sticky=False)
    assert server._probe(clients[0], fl_ctx)[STATUS] == READY
    runner.current_wf = None
    assert server._probe(clients[0], fl_ctx)[STATUS] == WAIT
    server.handle_event(EventType.ABOUT_TO_END_RUN, fl_ctx)
    assert server._probe(clients[0], fl_ctx)[STATUS] == DONE


def test_terminal_notice_targets_persistent_parent_and_waits_for_ack():
    server, engine, fl_ctx, _ = _server()
    server._handle_probe_message(_request())
    send = engine.run_manager.aux_runner.send_aux_request
    send.return_value = {"site-1": make_reply(ReturnCode.OK)}
    server.handle_event(EventType.END_RUN, fl_ctx)
    kwargs = send.call_args.kwargs
    assert kwargs["topic"] == TERMINAL_TOPIC
    assert kwargs["timeout"] == 5.0
    assert kwargs["request"][STATUS] == DONE
    assert kwargs["request"]["job_id"] == "job-1"
    assert len(kwargs["targets"]) == 1
    assert kwargs["targets"][0].fqcn == "site-1"
    assert kwargs["targets"][0].job_scoped is False


def test_missing_terminal_ack_is_reported():
    server, engine, fl_ctx, _ = _server()
    server._handle_probe_message(_request())
    engine.run_manager.aux_runner.send_aux_request.return_value = {}
    server.log_error = Mock()
    server.handle_event(EventType.END_RUN, fl_ctx)
    server.log_error.assert_called_once()
