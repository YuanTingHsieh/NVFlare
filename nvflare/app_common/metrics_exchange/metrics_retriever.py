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

import os
import time
from threading import Event, Thread
from typing import Optional

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.metrics_exchange.metrics_exchanger import MetricData
from nvflare.app_common.tracking.tracker_types import LogWriterName
from nvflare.app_common.widgets.streaming import ANALYTIC_EVENT_TYPE, AnalyticsSender
from nvflare.client.config import ClientConfig, ConfigKey
from nvflare.client.constants import CONFIG_METRICS_EXCHANGE
from nvflare.client.utils import get_external_pipe_args, get_external_pipe_class
from nvflare.fuel.utils.pipe.pipe import Message, Pipe
from nvflare.fuel.utils.pipe.file_pipe import FilePipe
from nvflare.fuel.utils.pipe.ipc_ppipe import IPCPPipe
from nvflare.fuel.utils.constants import Mode
from nvflare.fuel.utils.pipe.pipe_handler import PipeHandler, Topic
from nvflare.fuel.utils.validation_utils import check_object_type


class MetricsRetriever(FLComponent):
    def __init__(
        self,
        pipe_id: Optional[str] = None,
        event_type=ANALYTIC_EVENT_TYPE,
        writer_name=LogWriterName.TORCH_TB,
        topic: str = "metrics",
        get_poll_interval: float = 0.5,
        read_interval: float = 0.1,
        heartbeat_interval: float = 5.0,
        heartbeat_timeout: float = 30.0,
    ):
        """Metrics retriever.

        Args:
            pipe_id (str): how to get the pipe_id.
            event_type (str): event type to fire (defaults to "analytix_log_stats").
            writer_name: the log writer for syntax information (defaults to LogWriterName.TORCH_TB)
        """
        super().__init__()
        self.analytic_sender = AnalyticsSender(event_type=event_type, writer_name=writer_name)

        self._read_interval = read_interval
        self._heartbeat_interval = heartbeat_interval
        self._heartbeat_timeout = heartbeat_timeout
        self._pipe = None
        self._pipe_name = "metrics"
        self._pipe_id = pipe_id
        self.pipe_handler = None

        self._topic = topic
        self._get_poll_interval = get_poll_interval
        self.stop = Event()
        self._receive_thread = Thread(target=self.receive_data)
        self.fl_ctx = None

    def _init_pipe_handler(self) -> None:
        if self._pipe is None:
            raise RuntimeError("Pipe is None")
        self._pipe.open(name=self._pipe_name)
        # init pipe handler
        self.pipe_handler = PipeHandler(
            self._pipe,
            read_interval=self._read_interval,
            heartbeat_interval=self._heartbeat_interval,
            heartbeat_timeout=self._heartbeat_timeout,
        )
        self.pipe_handler.start()

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.ABOUT_TO_START_RUN:
            self._init_pipe(fl_ctx)
            self.analytic_sender.handle_event(event_type, fl_ctx)
            self._init_pipe_handler()
            self.prepare_external_config(fl_ctx)
            self.fl_ctx = fl_ctx
            self._receive_thread.start()
        elif event_type == EventType.ABOUT_TO_END_RUN:
            self.stop.set()
            self._receive_thread.join()

    def prepare_external_config(self, fl_ctx: FLContext):
        workspace = fl_ctx.get_engine().get_workspace()
        app_dir = workspace.get_app_dir(fl_ctx.get_job_id())
        config_file = os.path.join(app_dir, workspace.config_folder, CONFIG_METRICS_EXCHANGE)

        # prepare config exchange for metrics exchanger
        client_config = ClientConfig()
        config_dict = client_config.config
        config_dict[ConfigKey.PIPE_NAME] = self._pipe_name
        config_dict[ConfigKey.PIPE_CLASS] = get_external_pipe_class(self._pipe)
        config_dict[ConfigKey.PIPE_ARGS] = get_external_pipe_args(self._pipe, fl_ctx)
        config_dict[ConfigKey.SITE_NAME] = fl_ctx.get_identity_name()
        config_dict[ConfigKey.JOB_ID] = fl_ctx.get_job_id()
        client_config.to_json(config_file)

    def _init_pipe(self, fl_ctx: FLContext) -> None:
        engine = fl_ctx.get_engine()
        if self._pipe_id:
            pipe_name = f"{fl_ctx.get_identity_name()}-{fl_ctx.get_job_id()}-metrics"

            # gets Pipe using _pipe_id
            pipe = engine.get_component(self._pipe_id)
            check_object_type(self._pipe_id, pipe, Pipe)

            if isinstance(pipe, FilePipe):
                if pipe.root_path is None or pipe.root_path == "":
                    app_dir = engine.get_workspace().get_app_dir(fl_ctx.get_job_id())
                    pipe.root_path = os.path.abspath(app_dir)
                pipe.mode = Mode.PASSIVE
            elif not isinstance(pipe, IPCPPipe):
                raise RuntimeError(f"Pipe ({self._pipe_id}) of type ({type(pipe)}) is not supported")

            self._pipe = pipe
            self._pipe_name = pipe_name

    def receive_data(self):
        """Receives data and sends with AnalyticsSender."""
        while True:
            if self.stop.is_set():
                break
            msg: Optional[Message] = self.pipe_handler.get_next()
            if msg is not None:
                if msg.topic == [Topic.END, Topic.PEER_GONE, Topic.ABORT]:
                    self.system_panic("abort task", self.fl_ctx)
                elif msg.topic != self._topic:
                    self.system_panic(f"ignored '{msg.topic}' when waiting for '{self._topic}'", self.fl_ctx)
                else:
                    data: MetricData = msg.data
                    self.analytic_sender.add(
                        tag=data.key, value=data.value, data_type=data.data_type, **data.additional_args
                    )
            time.sleep(self._get_poll_interval)
