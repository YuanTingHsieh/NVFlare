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
"""Job API adapter with the current ScriptRunner input surface and B ownership."""

import os

from nvflare.client.config import ExchangeFormat, TransferType
from nvflare.fuel.utils.constants import FrameworkType
from nvflare.fuel.utils.secret_utils import split_command_preserving_secret_refs
from nvflare.private.fed.job_task_worker.executor import ClientAPIJobTaskWorkerExecutor


class JobTaskWorkerScriptRunner:
    def __init__(
        self,
        script: str,
        script_args="",
        launch_external_process: bool = False,
        command="python3 -u",
        framework: FrameworkType = FrameworkType.PYTORCH,
        server_expected_format: ExchangeFormat = ExchangeFormat.NUMPY,
        params_transfer_type: TransferType = TransferType.FULL,
        launch_once: bool = True,
        launch_timeout=300.0,
        shutdown_timeout: float = 0.0,
        memory_gc_rounds: int = 0,
        cuda_empty_cache: bool = False,
        execution_mode=None,
        launcher_id: str = "",
        worker_timeout: float = 3600.0,
        resources=None,
    ):
        if execution_mode is None:
            execution_mode = "external_process" if launch_external_process else "in_process"
        if execution_mode not in ("in_process", "external_process"):
            raise ValueError("B ScriptRunner supports in_process or external_process")
        self.script = script
        self.script_args = script_args
        self.command = command
        self.execution_mode = execution_mode
        self.framework = framework
        self.server_expected_format = server_expected_format
        self.params_transfer_type = params_transfer_type
        self.launch_once = launch_once
        self.launch_timeout = launch_timeout
        self.shutdown_timeout = shutdown_timeout
        self.memory_gc_rounds = memory_gc_rounds
        self.cuda_empty_cache = cuda_empty_cache
        self.launcher_id = launcher_id
        self.worker_timeout = worker_timeout
        self.resources = resources or []

    @staticmethod
    def _tokens(value):
        if isinstance(value, str):
            return split_command_preserving_secret_refs(value, posix=True)
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise ValueError("command and script_args must be a string or list of strings")
        return list(value)

    def add_to_fed_job(self, job, ctx, **kwargs):
        job.check_kwargs(args_to_check=kwargs, args_expected={"tasks": False})
        tasks = kwargs.get("tasks", ["*"])
        formats = {
            FrameworkType.PYTORCH: ExchangeFormat.PYTORCH,
            FrameworkType.TENSORFLOW: ExchangeFormat.KERAS_LAYER_WEIGHTS,
            FrameworkType.NUMPY: ExchangeFormat.NUMPY,
            FrameworkType.RAW: ExchangeFormat.RAW,
        }
        exchange_format = formats.get(self.framework)
        if exchange_format is None:
            raise ValueError(f"Framework {self.framework} unsupported")
        common = {
            "execution_mode": self.execution_mode,
            "launcher_id": self.launcher_id,
            "worker_timeout": self.worker_timeout,
            "script_resource": self.script,
            "resources": self.resources,
            "params_exchange_format": exchange_format,
            "server_expected_format": self.server_expected_format,
            "params_transfer_type": self.params_transfer_type,
            "launch_once": self.launch_once,
            "launch_timeout": self.launch_timeout,
            "shutdown_timeout": self.shutdown_timeout,
            "memory_gc_rounds": self.memory_gc_rounds,
            "cuda_empty_cache": self.cuda_empty_cache,
        }
        if self.execution_mode == "external_process":
            command = self._tokens(self.command)
            command.append(f"custom/{os.path.basename(self.script)}")
            command.extend(self._tokens(self.script_args))
            executor = ClientAPIJobTaskWorkerExecutor(command=command, **common)
        else:
            arguments = self.script_args if isinstance(self.script_args, str) else " ".join(self.script_args)
            executor = ClientAPIJobTaskWorkerExecutor(
                task_script_path=self.script,
                task_script_args=arguments,
                **common,
            )
        job.add_executor(executor, tasks=tasks, ctx=ctx)
        # Use an explicit file source so an absolute source underneath a
        # developer's sys.path is not reproduced as a repository-shaped path
        # in custom/.  The serialized executor names only the deployed file.
        job.add_file_source(src_path=self.script, dest_dir=None, app_folder_type="custom", ctx=ctx)
        if self.resources:
            job.add_resources(resources=self.resources, ctx=ctx)
        return {}
