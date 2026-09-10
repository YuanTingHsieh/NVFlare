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
from nvflare.private.fed.task_scope.protocol import ATTEMPT_OPTION, DIRECTORY_OPTION, write_receipt
from nvflare.private.fed.task_scope.runner import TaskScopedClientAppRunner


def main(args):
    options = parse_vars(args.set)
    attempt = options.get(ATTEMPT_OPTION)
    directory = options.get(DIRECTORY_OPTION)
    if not attempt or not directory:
        raise RuntimeError("experimental task-scoped worker requires an attempt ID and receipt directory")

    args.task_scope_outcome = None
    run_worker(args, app_runner_class=TaskScopedClientAppRunner)
    # The standard main returns only after its finally block has completed:
    # command/streaming shutdown, archive upload, cell/security/deployer cleanup,
    # client termination, and parent-monitor join. Never publish earlier.
    if args.task_scope_outcome is None:
        raise RuntimeError("task-scoped worker finished without a clean runner outcome")
    write_receipt(directory, attempt, args.task_scope_outcome)
    return 0


if __name__ == "__main__":
    args = parse_arguments()
    run_dir = os.path.join(args.workspace, args.job_id)
    sys.exit(MainProcessMonitor.run(main_func=main, run_dir=run_dir, args=args))
