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
"""Run: python examples/advanced/deployment-supervisor-a/demo.py --workspace /tmp/nvflare-a."""

import argparse
import json
import os

from nvflare.apis.shareable import Shareable
from nvflare.private.fed.deployment_supervisor import (
    ControllerTaskServiceAdapter,
    DeploymentTaskSupervisor,
    WorkerDefinition,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "config.json"))
    args = parser.parse_args()
    with open(args.config) as stream:
        config = json.load(stream)
    service = ControllerTaskServiceAdapter()
    task_ids = []
    for round_number in range(2):
        data = Shareable()
        data["value"] = round_number
        task_ids.extend(
            service.broadcast(
                job_id="architecture-a-demo",
                task_name="train",
                data=data,
                clients=["site-1", "site-2"],
                round_number=round_number,
            )
        )
    routes = {task: WorkerDefinition(**definition) for task, definition in config["routes"].items()}
    for site in ("site-1", "site-2"):
        supervisor = DeploymentTaskSupervisor(
            service=service,
            workspace=args.workspace,
            routes=routes,
        )
        supervisor.run_until_idle(site)
    print([dict(service.result(task_id)) for task_id in task_ids])


if __name__ == "__main__":
    main()
