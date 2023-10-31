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

from queue import Queue

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.metrics_exchange.metrics_exchanger import MemoryMetricsExchanger
from nvflare.app_common.tracking.tracker_types import LogWriterName
from nvflare.app_common.widgets.streaming import ANALYTIC_EVENT_TYPE
from nvflare.fuel.utils.constants import Mode
from nvflare.fuel.utils.pipe.memory_pipe import MemoryPipe
from nvflare.fuel.utils.pipe.pipe_handler import PipeHandler

from .metrics_retriever import MetricsRetriever


class MemoryMetricsRetriever(MetricsRetriever):
    def __init__(
        self,
        metrics_exchanger_id: str,
        event_type=ANALYTIC_EVENT_TYPE,
        writer_name=LogWriterName.TORCH_TB,
        topic: str = "metrics",
        get_poll_interval: float = 0.5,
        read_interval: float = 0.1,
        heartbeat_interval: float = 5.0,
        heartbeat_timeout: float = 30.0,
    ):
        """Metrics retriever with memory pipe.

        Args:
            event_type (str): event type to fire (defaults to "analytix_log_stats").
            writer_name: the log writer for syntax information (defaults to LogWriterName.TORCH_TB)
        """
        super().__init__(
            event_type=event_type,
            writer_name=writer_name,
            topic=topic,
            get_poll_interval=get_poll_interval,
            read_interval=read_interval,
            heartbeat_interval=heartbeat_interval,
            heartbeat_timeout=heartbeat_timeout,
        )
        self.metrics_exchanger_id = metrics_exchanger_id

        self.x_queue = Queue()
        self.y_queue = Queue()

    def _init_pipe(self, fl_ctx: FLContext) -> None:
        self._pipe = MemoryPipe(x_queue=self.x_queue, y_queue=self.y_queue, mode=Mode.PASSIVE)

    def _create_metrics_exchanger(self):
        pipe = MemoryPipe(x_queue=self.x_queue, y_queue=self.y_queue, mode=Mode.ACTIVE)
        pipe.open(name=self._pipe_name)
        # init pipe handler
        pipe_handler = PipeHandler(
            pipe,
            read_interval=self._read_interval,
            heartbeat_interval=self._heartbeat_interval,
            heartbeat_timeout=self._heartbeat_timeout,
        )
        pipe_handler.start()
        metrics_exchanger = MemoryMetricsExchanger(pipe_handler=pipe_handler)
        return metrics_exchanger

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        super().handle_event(event_type, fl_ctx)
        if event_type == EventType.ABOUT_TO_START_RUN:
            engine = fl_ctx.get_engine()
            # inserts MetricsExchanger into engine components
            metrics_exchanger = self._create_metrics_exchanger()
            all_components = engine.get_all_components()
            all_components[self.metrics_exchanger_id] = metrics_exchanger

    def prepare_external_config(self, fl_ctx: FLContext):
        pass
