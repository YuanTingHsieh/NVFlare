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

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_context import FLContext
from nvflare.fuel.utils.constants import Mode

from .ipc_pipe import IPCPipe


class IPCPPipe(IPCPipe, FLComponent):
    def __init__(self):
        """The PPipe (Passive Pipe) is used on FLARE client side."""
        super().__init__(Mode.PASSIVE)
        self.logger.info(f"IPCPPipe is created with {self.site_name=}")

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.START_RUN:
            engine = fl_ctx.get_engine()
            self.site_name = fl_ctx.get_identity_name()
            self.set_cell(engine.get_cell())
            if self.agent_id:
                self.peer_fqcn = self.agent_cell_name(self.site_name, self.agent_id)

    def open(self, name: str):
        self.set_agent_id(agent_id=name)
        self.register_cell_callback()
        if self.site_name:
            self.peer_fqcn = self.agent_cell_name(self.site_name, name)

    def close(self):
        # Passive pipe (on FLARE Client) doesn't need to close anything
        pass
