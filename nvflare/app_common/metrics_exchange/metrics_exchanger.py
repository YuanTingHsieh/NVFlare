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

from abc import ABC, abstractmethod
from typing import Any, Dict, Union

from nvflare.apis.analytix import AnalyticsDataType
from nvflare.app_common.abstract.metric_data import MetricData
from nvflare.client.config import ClientConfig, from_file
from nvflare.client.constants import CONFIG_METRICS_EXCHANGE
from nvflare.fuel.utils.pipe.file_pipe import FilePipe
from nvflare.fuel.utils.pipe.pipe import Message
from nvflare.fuel.utils.pipe.pipe_handler import PipeHandler


class MetricsExchanger(ABC):
    def __init__(
        self,
        topic: str = "metrics",
    ):
        self._topic = topic
        self._pipe_handler = None

    @abstractmethod
    def create_pipe_handler(self):
        pass

    def create_metric_message(self, key: str, value: Any, data_type: AnalyticsDataType, **kwargs):
        data = MetricData(key=key, value=value, data_type=data_type, additional_args=kwargs)
        req = Message.new_request(topic=self._topic, data=data)
        return req

    def log(self, key: str, value: Any, data_type: AnalyticsDataType, **kwargs):
        if self._pipe_handler is None:
            self.create_pipe_handler()
        req = self.create_metric_message(key, value, data_type, **kwargs)
        self._pipe_handler.send_to_peer(req)


class MemoryMetricsExchanger(MetricsExchanger):
    def __init__(
        self,
        pipe_handler: PipeHandler,
        topic: str = "metrics",
    ):
        super().__init__(topic)
        self._pipe_handler = pipe_handler

    def create_pipe_handler(self):
        pass


class FileMetricsExchanger(MetricsExchanger):
    def __init__(
        self,
        config: Union[str, Dict] = f"config/{CONFIG_METRICS_EXCHANGE}",
        topic: str = "metrics",
    ):
        super().__init__(topic)
        if isinstance(config, str):
            client_config = from_file(config_file=config)
        elif isinstance(config, dict):
            client_config = ClientConfig(config=config)
        else:
            raise ValueError("config should be either a string or dictionary.")
        self._client_config = client_config

    def create_pipe_handler(self):
        pipe_args = self._client_config.get_pipe_args()
        if self._client_config.get_pipe_class() == "FilePipe":
            pipe = FilePipe(**pipe_args)
        else:
            raise RuntimeError(f"Pipe class {self._client_config.get_pipe_class()} is not supported.")

        pipe.open(self._client_config.get_pipe_name())
        self._pipe_handler = PipeHandler(
            pipe,
            # read_interval=read_interval,
            # heartbeat_interval=heartbeat_interval,
            # heartbeat_timeout=heartbeat_timeout,
        )
