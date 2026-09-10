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
"""Real CP job-list/SP sync/SJ policy methods, without network or physical workers."""

import threading
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from nvflare.apis.client import Client
from nvflare.apis.fl_constant import RunProcessKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.impl.wf_comm_server import WFCommServer
from nvflare.apis.job_launcher_spec import JobReturnCode
from nvflare.apis.shareable import Shareable
from nvflare.private.defs import CellMessageHeaderKeys, new_cell_message
from nvflare.private.fed.client.client_engine import ClientEngine
from nvflare.private.fed.client.client_executor import JobExecutor
from nvflare.private.fed.client.client_status import ClientStatus
from nvflare.private.fed.server.fed_server import FederatedServer
from nvflare.private.fed.server.server_runner import ServerRunner
from nvflare.private.fed.task_scope.launcher import TaskScopedJobHandle


@pytest.fixture
def federation(tmp_path):
    clients = [Client(f"site-{i}", f"token-{i}") for i in (1, 2)]
    communicator = WFCommServer()
    communicator._engine = Mock()
    communicator._engine.get_clients.return_value = clients
    communicator._engine.new_context.side_effect = FLContext
    communicator.fire_event = Mock()

    runner = ServerRunner.__new__(ServerRunner)
    runner.wf_lock = threading.Lock()
    runner.current_wf = SimpleNamespace(controller=SimpleNamespace(communicator=communicator))
    server = FederatedServer.__new__(FederatedServer)
    server.engine = Mock()
    server.engine.run_processes = {"job-1": {RunProcessKey.PARTICIPANTS: {c.token: c for c in clients}}}
    server.engine.exception_run_processes = {}
    server.engine.job_runner.get_client_outcome_jobs.return_value = []
    server._job_reported_clients = {}
    server._job_reported_clients_lock = threading.Lock()
    server.logger = Mock()
    server.client_manager = SimpleNamespace(clients={c.token: c for c in clients})

    def report(job_id, client_name, reason):
        assert job_id == "job-1"
        runner.handle_dead_job(client_name, FLContext())

    server.engine.notify_dead_job.side_effect = report
    parents = []
    for client in clients:
        directory = tmp_path / client.name / "job-1"
        directory.mkdir(parents=True)
        handle = TaskScopedJobHandle(job_id="job-1", run_dir=str(directory), probe=Mock(), logger=Mock())
        handle._launch_attempt = Mock()
        executor = JobExecutor.__new__(JobExecutor)
        executor.lock = threading.Lock()
        executor.run_processes = {
            "job-1": {RunProcessKey.JOB_HANDLE: handle, RunProcessKey.STATUS: ClientStatus.STOPPED}
        }
        parent = ClientEngine.__new__(ClientEngine)
        parent.client_executor = executor
        parents.append(parent)
    return SimpleNamespace(server=server, clients=clients, parents=parents, communicator=communicator)


def _heartbeat_jobs(federation):
    # Communicator.send_heartbeat uses precisely engine.get_all_job_ids().
    for client, parent in zip(federation.clients, federation.parents):
        request = new_cell_message({CellMessageHeaderKeys.JOB_IDS: parent.get_all_job_ids()}, Shareable())
        assert federation.server._sync_client_jobs(request, client.token) == []


@pytest.mark.parametrize("require_previous_report", [True, False])
def test_all_cjs_absent_beyond_grace_do_not_make_logical_participants_dead(federation, require_previous_report):
    with patch(
        "nvflare.private.fed.server.fed_server.ConfigService.get_bool_var", return_value=require_previous_report
    ):
        for now in (1000.0, 2000.0, 10000.0):
            with patch("time.time", return_value=now):
                for parent in federation.parents:
                    handle = parent.client_executor.run_processes["job-1"][RunProcessKey.JOB_HANDLE]
                    assert handle.active is None  # All physical CJs are absent.
                    assert handle.poll() == JobReturnCode.UNKNOWN  # Logical job has not ended.
                    handle._launch_attempt.assert_not_called()
                _heartbeat_jobs(federation)
                federation.communicator._check_dead_clients()
                assert not federation.communicator._job_policy_violated()
    federation.server.engine.notify_dead_job.assert_not_called()
    assert federation.communicator._dead_clients == {}


@pytest.mark.parametrize("require_previous_report", [True, False])
def test_dropping_logical_jobs_still_causes_all_clients_dead(federation, require_previous_report):
    with patch(
        "nvflare.private.fed.server.fed_server.ConfigService.get_bool_var", return_value=require_previous_report
    ):
        with patch("time.time", return_value=1000.0):
            _heartbeat_jobs(federation)  # Establish a prior positive observation.
            for parent in federation.parents:
                parent.client_executor.run_processes.clear()
            _heartbeat_jobs(federation)
        with patch("time.time", return_value=2000.0):
            federation.communicator._check_dead_clients()
            assert federation.communicator._job_policy_violated()
    assert federation.server.engine.notify_dead_job.call_count == 2


def test_real_parent_loss_is_not_hidden_by_logical_job_handles(federation):
    with patch("time.time", return_value=1000.0):
        for client in federation.clients:
            federation.server.notify_dead_client(client)
    with patch("time.time", return_value=2000.0):
        federation.communicator._check_dead_clients()
        assert federation.communicator._job_policy_violated()
    assert federation.server.engine.notify_dead_job.call_count == 2
