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
"""Adapt the existing Slurm launcher to direct, Cell-free task workers."""

import os
from dataclasses import replace

from nvflare.apis.job_launcher_spec import JobHandleSpec, JobProcessEnv, JobReturnCode
from nvflare.app_opt.job_launcher.slurm.config import SlurmLauncherError


class SlurmTaskWorkerHandle(JobHandleSpec):
    """Normalize Slurm's UNKNOWN sentinel to the supervisor's pending value."""

    def __init__(self, handle):
        self.handle = handle

    def terminate(self):
        return self.handle.terminate()

    def poll(self):
        result = self.handle.poll()
        return None if result == JobReturnCode.UNKNOWN else result

    def wait(self):
        self.handle.wait()
        return self.poll()


class SlurmTaskWorkerLauncher:
    def __init__(self, job_launcher, job_meta, fl_ctx):
        self.job_launcher = job_launcher
        self.job_meta = job_meta
        self.fl_ctx = fl_ctx

    def launch(self, command, *, cwd, env):
        if self.job_launcher.manager is None:
            raise SlurmLauncherError("deployment task launch is unavailable inside a Slurm child process")
        if len(command) < 4 or command[-2] != "nvflare.private.fed.deployment_supervisor.worker":
            raise SlurmLauncherError("unexpected deployment task-worker command")
        attempt_dir = command[-1]
        if not os.path.isabs(attempt_dir):
            raise SlurmLauncherError("deployment task-worker directory must be absolute")
        plan = self.job_launcher._build_launch_plan(self.job_meta, self.fl_ctx)
        if plan.resources.nodes != 1 or plan.additional_node_command:
            raise SlurmLauncherError("deployment-supervised workers currently support one Slurm node")
        secret_env = dict(plan.study_secret_env)
        for name in (JobProcessEnv.AUTH_TOKEN, JobProcessEnv.TOKEN_SIGNATURE, JobProcessEnv.SSID):
            secret_env.pop(name, None)
        plan = replace(
            plan,
            exe_module="nvflare.private.fed.deployment_supervisor.worker",
            module_args=(attempt_dir,),
            study_secret_env=secret_env,
            additional_node_command=(),
            node_app_dir=None,
        )
        return SlurmTaskWorkerHandle(self.job_launcher.manager.launch(plan))
