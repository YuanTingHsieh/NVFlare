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

import socket
import threading
import time

import pytest

from nvflare.fuel.f3.communicator import Communicator
from nvflare.fuel.f3.drivers.connector_info import Mode
from nvflare.fuel.f3.endpoint import Endpoint, EndpointState
from nvflare.fuel.f3.message import Message
from nvflare.fuel.f3.sfm.constants import Types
from nvflare.fuel.f3.sfm.prefix import Prefix


def _wait_for(predicate, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


@pytest.mark.parametrize("scheme", ["tcp", "atcp", "http", "grpc", "agrpc"])
def test_silent_peer_is_closed_and_active_connector_reconnects(scheme):
    received = threading.Event()
    ignored_ping = threading.Event()
    drop_pings = threading.Event()
    original_readers = {id(t) for t in threading.enumerate() if t.name == "grpc_reader"}

    class Receiver:
        def process_message(self, endpoint, connection, app_id, message):
            if message.payload == b"after reconnect":
                received.set()

    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]

    server = Communicator(Endpoint("server"))
    client = Communicator(Endpoint("client"))
    server.register_message_receiver(532, Receiver())

    # Simulate a peer that keeps its connection open but stops answering PING.
    original_process_frame = client.conn_manager.process_frame

    def ignore_ping(sfm_conn, frame):
        if drop_pings.is_set() and Prefix.from_bytes(frame).type == Types.PING:
            ignored_ping.set()
            return
        return original_process_frame(sfm_conn, frame)

    client.conn_manager.process_frame = ignore_ping

    try:
        _, url, _ = server.start_listener(scheme, {"host": "127.0.0.1", "port": port})
        server.start()
        client.add_connector(url, Mode.ACTIVE)
        client.start()
        assert _wait_for(lambda: server.find_endpoint("client") and client.find_endpoint("server"))
        assert server.find_endpoint("client").state == EndpointState.READY
        assert client.find_endpoint("server").state == EndpointState.READY

        monitor = server.conn_manager.heartbeat_monitor
        sfm_conn = next(iter(server.conn_manager.sfm_conns.values()))
        last_activity = sfm_conn.last_activity
        monitor.curr_time = time.monotonic() + monitor.interval + 1
        monitor._check_heartbeat()
        assert _wait_for(lambda: sfm_conn.last_activity > last_activity)  # Healthy peer replies with PONG.
        assert server.find_endpoint("client").state == EndpointState.READY

        drop_pings.set()
        monitor.curr_time = time.monotonic() + monitor.interval + 1
        monitor._check_heartbeat()
        assert ignored_ping.wait(5)
        assert server.find_endpoint("client").state == EndpointState.READY

        monitor.curr_time = time.monotonic() + getattr(monitor, "timeout", 2 * monitor.interval) + monitor.interval + 1
        monitor._check_heartbeat()
        assert _wait_for(lambda: server.find_endpoint("client").state == EndpointState.DISCONNECTED)
        assert _wait_for(lambda: client.find_endpoint("server").state == EndpointState.DISCONNECTED)
        assert _wait_for(
            lambda: server.find_endpoint("client").state == client.find_endpoint("server").state == EndpointState.READY
        )
        if scheme == "grpc":
            # The old server-side request reader must exit after its RPC returns.
            assert _wait_for(
                lambda: sum(t.name == "grpc_reader" and id(t) not in original_readers for t in threading.enumerate())
                == 1
            )

        client.send(Endpoint("server"), 532, Message({}, b"after reconnect"))
        assert received.wait(5)
    finally:
        client.stop()
        server.stop()
