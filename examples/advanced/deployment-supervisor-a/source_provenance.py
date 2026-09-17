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
"""Fail closed unless Architecture A's server-side modules load from one source root.

Run this through each server, CP, and worker entrypoint before a qualification
job is submitted.  The check deliberately verifies the server module because
the deployment task AUX callbacks live there, not in the disposable worker.
"""

import argparse
import hashlib
import inspect
import json
from pathlib import Path

from nvflare.private.fed.deployment_supervisor import federation
from nvflare.private.fed.server import server_runner


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inside(root: Path, path: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def check(source_root: Path) -> dict:
    modules = {"server_runner": Path(server_runner.__file__), "federation": Path(federation.__file__)}
    if not all(_inside(source_root, path) for path in modules.values()):
        raise RuntimeError(f"Architecture A modules did not load from {source_root}: {modules}")
    if "deployment_task_service" not in inspect.getsource(server_runner.ServerRunner.__init__):
        raise RuntimeError("loaded ServerRunner does not install the deployment task service")
    if federation.ACQUIRE_TOPIC != "deployment_task_acquire":
        raise RuntimeError("loaded deployment task protocol has an unexpected acquire topic")
    if "register_request_cb" not in inspect.getsource(federation.DeploymentTaskService.start):
        raise RuntimeError("loaded deployment task service does not register request callbacks")
    return {name: {"path": str(path.resolve()), "sha256": _digest(path)} for name, path in modules.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source_root", type=Path)
    args = parser.parse_args()
    print(json.dumps(check(args.source_root), sort_keys=True))


if __name__ == "__main__":
    main()
