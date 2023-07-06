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


from typing import Dict, Optional

import docker
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.launcher import Launcher


class DockerLauncher(Launcher):
    def __init__(self, script: str, image: str, app_root: str = "/app", options: Optional[Dict] = None):
        """DockerLauncher using docker python package.

        It will be launched as follows:
            client = docker.from_env()
            client.containers.run(image, script, **options)

        Args:
            script (str): Script to be launched inside docker container.
            image (str): Docker image to be launched.
            app_root (str): The path inside the docker container to find app custom folder.
            options (str): Option when launching docker.
        """
        super().__init__()

        self._image = image
        self._options = options if options else {}
        self._options["detach"] = True
        self._options["stdout"] = True
        self._options["stderr"] = True
        self._script = script
        self._app_root = app_root
        self._container = None

    def initialize(self, fl_ctx: FLContext):
        app_dir = self.get_app_dir(fl_ctx)
        volumes = self._options.get("volumes", [])
        volumes.append(f"{app_dir}:{self._app_root}")
        self._options["volumes"] = volumes
        self._options["working_dir"] = self._app_root

    def launch_task(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> bool:
        if self._container is None:
            client = docker.from_env()
            self._container = client.containers.run(self._image, self._script, **self._options)
            return True
        return False

    def stop_task(self, task_name: str, fl_ctx: FLContext) -> None:
        if self._container:
            self._container.stop()
            self._container.remove()
            self._container = None
