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
"""Slurm allocation adapter for B's fresh application worker only."""

import os

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey, JobConstants
from nvflare.apis.fl_context import FLContext
from nvflare.apis.job_def import JobMetaKey
from nvflare.apis.job_launcher_spec import JobProcessEnv
from nvflare.apis.utils.format_check import check_job_id
from nvflare.apis.workspace import Workspace
from nvflare.app_opt.job_launcher.slurm.config import LaunchPlan, SlurmLauncherError, _require_string
from nvflare.app_opt.job_launcher.slurm.launcher import ClientSlurmJobLauncher, _resolve_resources, _validate_run_dir
from nvflare.private.fed.job_task_worker.executor import validate_worker_environment
from nvflare.private.fed.job_task_worker.protocol import WORKER_MODULE


class SlurmTaskWorkerLauncher(ClientSlurmJobLauncher):
    """Run only the application worker in a one-node Slurm allocation.

    The normal NVFlare job metadata is deliberately not used to select or size
    the CJ launcher: ``worker_resources`` sizes these nested task allocations,
    while the CPU CJ must be launched by the ordinary local job launcher.
    """

    def __init__(
        self,
        *,
        workspace_path,
        sandbox,
        python_path,
        executables,
        worker_resources=None,
        image=None,
        internal_port=8102,
        sbatch_directives=None,
        setup="",
        forward_env=None,
        parent_host=None,
        submit_timeout=30.0,
        query_timeout=10.0,
        cancel_timeout=10.0,
        poll_interval=10.0,
        pending_timeout=600.0,
        multi_node_port_range=None,
        task_environment=None,
    ):
        self._workspace_path = workspace_path
        self._sandbox = sandbox
        self._python_path = python_path
        self._executables = executables
        self.worker_resources = worker_resources or {"nodes": 1, "gpus_per_node": 1}
        self._image = image
        self._internal_port = internal_port
        self._sbatch_directives = sbatch_directives
        self._setup = setup
        self._forward_env = forward_env
        self._parent_host = parent_host
        self._submit_timeout = submit_timeout
        self._query_timeout = query_timeout
        self._cancel_timeout = cancel_timeout
        self._poll_interval = poll_interval
        self._pending_timeout = pending_timeout
        self._multi_node_port_range = multi_node_port_range
        self.task_environment = validate_worker_environment(task_environment)
        super().__init__(
            workspace_path=workspace_path,
            sandbox=sandbox,
            python_path=python_path,
            executables=executables,
            image=image,
            internal_port=internal_port,
            sbatch_directives=sbatch_directives,
            setup=setup,
            forward_env=forward_env,
            parent_host=parent_host,
            submit_timeout=submit_timeout,
            query_timeout=query_timeout,
            cancel_timeout=cancel_timeout,
            poll_interval=poll_interval,
            pending_timeout=pending_timeout,
            multi_node_port_range=multi_node_port_range,
        )

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.START_RUN:
            if self.manager is None:
                raise SlurmLauncherError(
                    "Architecture B requires the resident CPU CJ outside Slurm; nested launch is unavailable"
                )
            self.manager.initialize()
        elif event_type == EventType.END_RUN and self.manager is not None:
            self.manager.shutdown()

    def _build_task_plan(self, attempt_dir, fl_ctx):
        if self.manager is None:
            raise SlurmLauncherError(
                "Architecture B requires the resident CPU CJ outside Slurm; nested launch is unavailable"
            )
        job_meta = fl_ctx.get_prop(FLContextKey.JOB_META)
        if not isinstance(job_meta, dict):
            raise SlurmLauncherError("task-worker launch requires job metadata in the resident CJ")
        job_id = job_meta.get(JobConstants.JOB_ID)
        try:
            check_job_id(job_id)
        except ValueError as e:
            raise SlurmLauncherError("invalid job ID") from e
        site_name = _require_string(fl_ctx.get_identity_name(), "site identity")
        workspace = fl_ctx.get_prop(FLContextKey.WORKSPACE_OBJECT)
        if not isinstance(workspace, Workspace):
            raise SlurmLauncherError(f"missing {FLContextKey.WORKSPACE_OBJECT} in FLContext")
        if os.path.realpath(workspace.get_root_dir()) != self.config.workspace_path:
            raise SlurmLauncherError("FLContext workspace does not match configured Slurm workspace_path")
        run_dir = _validate_run_dir(self.config.workspace_path, workspace.get_run_dir(job_id))
        real_attempt = os.path.realpath(attempt_dir)
        if not os.path.isabs(attempt_dir) or os.path.commonpath((run_dir, real_attempt)) != run_dir:
            raise SlurmLauncherError("task-worker attempt directory must be inside the current run directory")

        study = job_meta.get(JobMetaKey.STUDY.value)
        if study is not None:
            study = _require_string(study, "study name")
        runtime = self._load_study_runtime(study)
        sandbox, image, python_path, setup, directives = self._effective_study_values(runtime, None)
        resources = _resolve_resources(
            {},
            site_name,
            sandbox,
            self.config.pending_timeout,
            spec=self.worker_resources,
        )
        if resources.nodes != 1:
            raise SlurmLauncherError("Architecture B prototype supports one Slurm node per task worker")
        study_env, secret_env = self._study_environment(runtime)
        duplicated = set(study_env) & set(self.task_environment)
        if duplicated:
            raise SlurmLauncherError(f"study env and task_environment duplicate name(s): {sorted(duplicated)}")
        study_env.update(self.task_environment)
        forbidden = {JobProcessEnv.AUTH_TOKEN, JobProcessEnv.TOKEN_SIGNATURE, JobProcessEnv.SSID}
        if forbidden & (set(study_env) | set(secret_env)):
            raise SlurmLauncherError("federation job credentials must not enter the application worker environment")
        mounts = tuple(self._study_mounts(runtime)) if sandbox != "none" else ()
        return LaunchPlan(
            job_id=job_id,
            site_name=site_name,
            run_dir=run_dir,
            exe_module=WORKER_MODULE,
            module_args=(real_attempt,),
            resources=resources,
            directives=directives,
            sandbox=sandbox,
            image=image,
            setup=setup,
            study_env=study_env,
            study_secret_env=secret_env,
            mounts=mounts,
            python_path=python_path,
            python_env=self._python_environment(workspace, job_id),
            forward_env=self.config.forward_env,
            additional_node_command=(),
            node_app_dir=None,
        )

    def launch(self, command, *, cwd, env, fl_ctx):
        del cwd
        if len(command) != 4 or command[1:3] != ["-m", WORKER_MODULE] or not isinstance(command[3], str):
            raise SlurmLauncherError("unexpected Architecture B task-worker command")
        for name in (JobProcessEnv.AUTH_TOKEN, JobProcessEnv.TOKEN_SIGNATURE, JobProcessEnv.SSID):
            if name in env:
                raise SlurmLauncherError("application worker environment contains a federation credential")
        self.manager.initialize()
        return self.manager.launch(self._build_task_plan(command[3], fl_ctx))
