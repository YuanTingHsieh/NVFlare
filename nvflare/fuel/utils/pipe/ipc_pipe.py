# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

import logging
from abc import ABC
from queue import Empty, Full, Queue
from typing import Any, Union

from nvflare.fuel.f3.cellnet.cell import Cell
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey, ReturnCode
from nvflare.fuel.f3.cellnet.utils import make_reply
from nvflare.fuel.f3.message import Message as CellMessage
from nvflare.fuel.utils.constants import Mode
from nvflare.fuel.utils.fobs import fobs
from nvflare.fuel.utils.fobs.datum import DatumManager
from nvflare.fuel.utils.fobs.decomposer import Decomposer, Externalizer, Internalizer

from .pipe import Message, Pipe

SSL_ROOT_CERT = "rootCA.pem"
CHANNEL = "ipc_pipe"
TOPIC = "ipc_pipe"


class PipeMessageDecomposer(Decomposer):
    def supported_type(self):
        return Message

    def decompose(self, target: Message, manager: DatumManager = None) -> Any:
        externalizer = Externalizer(manager)
        return (target.msg_type, target.topic, externalizer.externalize(target.data), target.msg_id, target.req_id)

    def recompose(self, data: Any, manager: DatumManager = None) -> Message:
        msg_type, topic, data, msg_id, req_id = data
        internalizer = Internalizer(manager)
        return Message(
            msg_type=msg_type, topic=topic, data=internalizer.internalize(data), msg_id=msg_id, req_id=req_id
        )


class IPCPipe(Pipe, ABC):
    def __init__(
        self,
        mode: Mode = Mode.ACTIVE,
    ):
        super().__init__(mode)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.cell = None
        self.peer_fqcn = None
        self.agent_id = None
        self.site_name = None
        self.received_msgs = Queue()
        self.topic = None

        fobs.register(PipeMessageDecomposer)

    def set_agent_id(self, agent_id):
        self.agent_id = agent_id
        self.topic = f"{TOPIC}_{self.agent_id}"

    def set_cell(self, cell: Cell):
        self.cell = cell

    def register_cell_callback(self):
        if not self.cell or not self.topic:
            raise RuntimeError("self.cell and self.topic is not set")
        self.cell.register_request_cb(channel=CHANNEL, topic=self.topic, cb=self._receive_message)
        self.logger.debug(f"registered task CB for {CHANNEL} {self.topic}")

    def send(self, msg: Message, timeout=None) -> bool:
        if not self.peer_fqcn:
            # the A-side does not know its peer FQCN until a message is received from the peer (P-side).
            self.logger.warning("peer FQCN is not known yet")
            return False

        try:
            self.logger.info(f"sending msg: {msg} to {self.peer_fqcn}")
            cell_message = CellMessage(payload=msg)
            reply = self.cell.send_request(
                channel=CHANNEL, topic=self.topic, target=self.peer_fqcn, request=cell_message, timeout=timeout
            )
            if reply:
                rc = reply.get_header(MessageHeaderKey.RETURN_CODE)
                if rc == ReturnCode.OK:
                    return True
                else:
                    self.logger.error(f"return code from peer {self.peer_fqcn}: {rc}")
                    return False
            else:
                return False
        except Exception as e:
            self.logger.error(f"cell send exception: {e}")
            return False

    def _receive_message(self, request: CellMessage):
        sender = request.get_header(MessageHeaderKey.ORIGIN)
        if not self.peer_fqcn:
            # this is A-side
            self.peer_fqcn = sender
        elif self.peer_fqcn != sender:
            raise RuntimeError(f"peer FQCN mismatch: expect {self.peer_fqcn} but got {sender}")

        try:
            self.received_msgs.put_nowait(request.payload)
            return make_reply(ReturnCode.OK)
        except Full:
            self.logger.error("queue is full")
            return make_reply(ReturnCode.COMM_ERROR)

    def receive(self, timeout=None) -> Union[Message, None]:
        try:
            if timeout:
                return self.received_msgs.get(block=True, timeout=timeout)
            else:
                return self.received_msgs.get_nowait()
        except Empty:
            return None

    def clear(self):
        while not self.received_msgs.empty():
            self.received_msgs.get_nowait()

    @staticmethod
    def agent_cell_name(site_name, name):
        return f"{site_name}--{name}"
