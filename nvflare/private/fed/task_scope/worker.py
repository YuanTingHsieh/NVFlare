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
"""Experimental task-scoped entrypoint; the receipt follows normal worker cleanup."""

import os
import sys

from nvflare.fuel.f3.mpm import MainProcessMonitor
from nvflare.fuel.utils.argument_utils import parse_vars
from nvflare.private.fed.app.client.worker_process import main as run_worker
from nvflare.private.fed.app.client.worker_process import parse_arguments
from nvflare.private.fed.task_scope.protocol import (
    ATTEMPT_OPTION,
    COMPUTE,
    DIRECTORY_OPTION,
    PHASE_OPTION,
    PULL,
    PUSH,
    write_receipt,
)
from nvflare.private.fed.task_scope.runner import TaskScopedClientAppRunner


def main(args):
    options = parse_vars(args.set)
    attempt = options.get(ATTEMPT_OPTION)
    directory = options.get(DIRECTORY_OPTION)
    if not attempt or not directory:
        raise RuntimeError("experimental task-scoped worker requires an attempt ID and receipt directory")
    phase = options.get(PHASE_OPTION)
    if phase is not None and phase not in (PULL, COMPUTE, PUSH):
        raise ValueError(f"invalid task-scope phase: {phase}")

    args.task_scope_outcome = None
    if phase is None:
        run_worker(args, app_runner_class=TaskScopedClientAppRunner)
    else:
        # The shared filesystem carries pull/compute artifacts to the next CJ.
        # Only the CPU push allocation uploads the workspace on shutdown.
        run_worker(args, app_runner_class=TaskScopedClientAppRunner, upload_workspace_results=phase == PUSH)
    # The standard main returns only after its finally block has completed:
    # command/streaming shutdown, configured archive upload, cell/security/deployer cleanup,
    # client termination, and parent-monitor join. Never publish earlier.
    if args.task_scope_outcome is None:
        raise RuntimeError("task-scoped worker finished without a clean runner outcome")
    outcome = args.task_scope_outcome
    if phase is not None:
        directory = os.path.join(directory, phase)
        outcome = dict(outcome, phase=phase)
    write_receipt(directory, attempt, outcome)
    return 0


if __name__ == "__main__":
    args = parse_arguments()
    run_dir = os.path.join(args.workspace, args.job_id)
    sys.exit(MainProcessMonitor.run(main_func=main, run_dir=run_dir, args=args))
