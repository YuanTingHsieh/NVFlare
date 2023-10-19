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

from nvflare.app_common.decomposers import common_decomposers
from nvflare.fuel.f3.cellnet.cell import Cell
from nvflare.fuel.f3.cellnet.net_agent import NetAgent
from nvflare.fuel.f3.drivers.driver_params import DriverParams
from nvflare.fuel.utils.config_service import ConfigService
from nvflare.fuel.utils.constants import Mode

from .ipc_pipe import IPCPipe

SSL_ROOT_CERT = "rootCA.pem"


class IPCAPipe(IPCPipe):
    """The APipe (Active Pipe) is used on the 3rd-party side"""

    def __init__(
        self,
        site_name: str,
        root_url: str = "",
        secure=True,
        workspace_dir: str = "",
    ):
        super().__init__(Mode.ACTIVE)
        self.site_name = site_name
        self.root_url = root_url
        self.secure = secure
        self.workspace_dir = workspace_dir
        self.net_agent = None
        common_decomposers.register()
        if secure:
            ConfigService.initialize(section_files={}, config_path=[workspace_dir])

    def _build_cell(self, name):
        cell_name = self.agent_cell_name(self.site_name, name)
        credentials = {}
        if self.secure:
            root_cert_path = ConfigService.find_file(SSL_ROOT_CERT)
            if not root_cert_path:
                raise ValueError(f"cannot find {SSL_ROOT_CERT} from config path {self.workspace_dir}")

            credentials = {
                DriverParams.CA_CERT.value: root_cert_path,
            }

        cell = Cell(
            fqcn=cell_name,
            root_url=self.root_url,
            secure=self.secure,
            credentials=credentials,
            create_internal_listener=False,
        )
        self.net_agent = NetAgent(cell)
        self.set_cell(cell)

    def open(self, name: str):
        self.set_agent_id(agent_id=name)
        self._build_cell(name)
        self.register_cell_callback()
        self.cell.start()

    def close(self):
        self.cell.stop()
        self.net_agent.close()
        self.cell.core_cell.ALL_CELLS.pop(self.cell.core_cell.my_info.fqcn)
