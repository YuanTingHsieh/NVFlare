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
"""Run maintained NumPy and log-streaming recipes through the B worker adapter."""

import argparse
import os

from nvflare.app_common.np.recipes.fedavg import NumpyFedAvgRecipe
from nvflare.client.config import TransferType
from nvflare.private.fed.job_task_worker.script_runner import JobTaskWorkerScriptRunner
from nvflare.recipe import SimEnv, add_experiment_tracking


class ArchitectureBNumpyFedAvgRecipe(NumpyFedAvgRecipe):
    def _create_client_runner(self, site_config):
        return JobTaskWorkerScriptRunner(
            script=self._site_value(site_config, "train_script", self.train_script),
            script_args=self._site_value(site_config, "train_args", self.train_args),
            launch_external_process=self._site_value(
                site_config, "launch_external_process", self.launch_external_process
            ),
            command=self._site_value(site_config, "command", self.command),
            framework=self._site_value(site_config, "framework", self._client_runner_framework),
            server_expected_format=self._site_value(site_config, "server_expected_format", self.server_expected_format),
            params_transfer_type=self._site_value(site_config, "params_transfer_type", self.params_transfer_type),
            launch_once=self._site_value(site_config, "launch_once", self.launch_once),
            launch_timeout=site_config.get("launch_timeout", self.launch_timeout),
            shutdown_timeout=self._site_value(site_config, "shutdown_timeout", self.shutdown_timeout),
            memory_gc_rounds=self.client_memory_gc_rounds,
            cuda_empty_cache=self.cuda_empty_cache,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", choices=("hello-numpy", "hello-log-streaming"), required=True)
    parser.add_argument("--mode", choices=("in_process", "external_process"), required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()
    repository = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    source = os.path.join(repository, "examples", "hello-world", args.workload, "client.py")
    recipe = ArchitectureBNumpyFedAvgRecipe(
        name=f"architecture-b-{args.workload}-{args.mode}",
        min_clients=2,
        num_rounds=args.rounds,
        model=[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        train_script=source,
        train_args="--update_type full",
        launch_external_process=args.mode == "external_process",
        params_transfer_type=TransferType.FULL,
    )
    add_experiment_tracking(recipe, tracking_type="tensorboard")
    if args.workload == "hello-log-streaming":
        recipe.enable_log_streaming()
    run = recipe.execute(SimEnv(num_clients=2, workspace_root=args.workspace))
    print("result", run.get_result())
    print("status", run.get_status())


if __name__ == "__main__":
    main()
