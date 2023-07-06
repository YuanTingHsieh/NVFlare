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

import glob
import logging
import os
import threading
import time

import h5py


class NVDXI:
    def __init__(
        self,
        data_exchange_path: str,
        *,
        file_accessor_name: str = "PyTorchFileAccessor",
        heartbeat_period: float = 0.5,
        heartbeat_timeout: float = 10000,
    ):
        self.pipe_path = os.path.join(data_exchange_path, "pipe")
        for i in ["t", "x", "y"]:
            os.makedirs(os.path.join(self.pipe_path, i), exist_ok=True)

        self.logger = logging.getLogger(self.__class__.__name__)
        self.file_accessor_name = file_accessor_name
        # check heartbeat
        self._last_received_heartbeat_time = None

        # heartbeat
        self._heartbeat_thread = None
        self._stop_heartbeat = False
        self._heartbeat_period = heartbeat_period
        self.heartbeat_timeout = heartbeat_timeout

    def remove(self, name: str):
        endpoint_path = os.path.join(self.pipe_path, "y")
        os.remove(os.path.join(endpoint_path, f"REQ.{self.file_accessor_name}.{name}"))

    def get_data(self, name: str):
        endpoint_path = os.path.join(self.pipe_path, "y")
        input_file = os.path.join(endpoint_path, f"REQ.{self.file_accessor_name}.{name}")

        data = {}

        with h5py.File(input_file, "r") as file:
            for key in file.keys():
                data[key] = file[key][()]

        return data

    def put_data(self, name, output_weights):
        endpoint_path = os.path.join(self.pipe_path, "x")
        output_file = os.path.join(endpoint_path, f"REQ.{self.file_accessor_name}.{name}")
        with h5py.File(output_file, "w") as file:
            for key, value in output_weights.items():
                file.create_dataset(key, data=value)

    def check_heartbeat(self):
        endpoint_path = os.path.join(self.pipe_path, "y")
        heartbeat_files = glob.glob(os.path.join(endpoint_path, "REQ._HEARTBEAT_*"))

        now = time.time()
        if not self._last_received_heartbeat_time:
            self._last_received_heartbeat_time = now
        try:
            if len(heartbeat_files) != 0:
                self._last_received_heartbeat_time = now
                for file in heartbeat_files:
                    os.remove(file)
        except Exception as e:
            self.logger.exception(f"Error getting heartbeat time: {e}")

        if now - self._last_received_heartbeat_time > self.heartbeat_timeout:
            return False
        return True

    def start_heartbeat(self):
        if self._heartbeat_thread:
            return
        self._stop_heartbeat = False
        thread = threading.Thread(target=self._heartbeat)
        thread.start()
        self._heartbeat_thread = thread

    def stop_heartbeat(self):
        self._stop_heartbeat = True
        if self._heartbeat_thread:
            self._heartbeat_thread.join()
            self._heartbeat_thread = None

    def _heartbeat(self):
        endpoint_path = os.path.join(self.pipe_path, "x")
        count = 0
        while not self._stop_heartbeat:
            output_file = os.path.join(endpoint_path, f"REQ._HEARTBEAT_.{count}")
            with h5py.File(output_file, "w") as file:
                file.create_dataset("__default__", data="")
            time.sleep(self._heartbeat_period)
            count += 1
