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

import argparse

from components import CheckpointCounterExecutor, GapController

from nvflare.job_config.api import FedJob
from nvflare.private.fed.task_scope.server import TaskScopedServer


def main():
    parser = argparse.ArgumentParser(description="Export the experimental whole-CJ-per-task Slurm job")
    parser.add_argument("--output", required=True)
    parser.add_argument("--clients", nargs="+", default=["site-1", "site-2"])
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--gap-seconds", type=float, default=15.0)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--crash-round", type=int, default=-1)
    args = parser.parse_args()
    if args.rounds < 1 or args.gap_seconds < 0 or args.gpus < 0:
        parser.error("rounds must be positive; gap-seconds and gpus must be nonnegative")
    job = FedJob(name="slurm-task-scope", min_clients=len(args.clients), mandatory_clients=args.clients)
    job.to_server(TaskScopedServer(), id="task_scope")
    job.to_server(GapController(rounds=args.rounds, gap_seconds=args.gap_seconds))
    for site in args.clients:
        job.to(CheckpointCounterExecutor(crash_round=args.crash_round), site, tasks=["count"])
        job.job.add_resource_spec(site, {"num_of_gpus": args.gpus})
    job.export_job(args.output)


if __name__ == "__main__":
    main()
