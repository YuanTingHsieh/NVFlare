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
"""Example export and real Python-process checkpoint tests; not Slurm/CJ E2E."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable

ROOT = Path(__file__).resolve().parents[4]
EXAMPLE = ROOT / "examples" / "advanced" / "slurm-task-scope"
COUNTER_PROCESS = """
import importlib.util, json, sys
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.apis.workspace import Workspace
spec = importlib.util.spec_from_file_location('counter_components', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
ctx = FLContext()
ctx.set_prop(FLContextKey.CURRENT_RUN, 'job-1', private=False, sticky=False)
ctx.set_prop(FLContextKey.WORKSPACE_OBJECT, Workspace(sys.argv[2]), private=True, sticky=False)
executor = module.CheckpointCounterExecutor(compute_seconds=0, crash_round=int(sys.argv[4]))
result = executor.execute('count', Shareable({'round': int(sys.argv[3])}), ctx, Signal())
print(json.dumps(dict(result)))
"""


def _environment():
    return dict(os.environ, PYTHONPATH=str(ROOT))


def test_controller_records_result_data_without_runtime_peer_headers():
    spec = importlib.util.spec_from_file_location("counter_components", EXAMPLE / "components.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    controller = module.GapController()
    payload = {"round": 0, "value": 1, "pid": 123, "slurm_id": "42", "finished_at": 1.0}
    result = Shareable(payload)
    result.set_header("__peer_ctx__", FLContext())
    client_task = SimpleNamespace(
        result=result, task=SimpleNamespace(data={"round": 0}), client=SimpleNamespace(name="site-1")
    )
    controller._receive(client_task, FLContext())
    assert json.loads(json.dumps(controller.results)) == {"0": {"site-1": payload}}


def _count(workspace, round_number, crash_round=-1):
    (workspace / "startup").mkdir(exist_ok=True)
    (workspace / "local").mkdir(exist_ok=True)
    return subprocess.run(
        [
            sys.executable,
            "-c",
            COUNTER_PROCESS,
            str(EXAMPLE / "components.py"),
            str(workspace),
            str(round_number),
            str(crash_round),
        ],
        cwd=ROOT,
        env=_environment(),
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_counter_restores_state_in_three_fresh_python_processes(tmp_path):
    (tmp_path / "job-1").mkdir()
    results = []
    for number in range(3):
        process = _count(tmp_path, number)
        assert process.returncode == 0, process.stderr
        results.append(json.loads(process.stdout))
    assert [r["value"] for r in results] == [1, 2, 3]
    assert len({r["pid"] for r in results}) == 3
    assert json.loads((tmp_path / "job-1" / "task_scope_counter.json").read_text())["value"] == 3


def test_counter_replays_checkpoint_without_incrementing_twice(tmp_path):
    (tmp_path / "job-1").mkdir()
    first = _count(tmp_path, 0)
    replay = _count(tmp_path, 0)
    assert first.returncode == replay.returncode == 0
    assert json.loads(first.stdout) == json.loads(replay.stdout)


def test_hard_exit_does_not_advance_checkpoint(tmp_path):
    (tmp_path / "job-1").mkdir()
    assert _count(tmp_path, 0).returncode == 0
    assert _count(tmp_path, 1, crash_round=1).returncode == 1
    assert json.loads((tmp_path / "job-1" / "task_scope_counter.json").read_text())["round"] == 0


def test_export_preserves_nondefault_parameters_and_bundles_components(tmp_path):
    process = subprocess.run(
        [
            sys.executable,
            str(EXAMPLE / "job.py"),
            "--output",
            str(tmp_path),
            "--rounds",
            "4",
            "--gap-seconds",
            "20",
            "--gpus",
            "0",
            "--crash-round",
            "2",
        ],
        cwd=ROOT,
        env=_environment(),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert process.returncode == 0, process.stderr
    job = tmp_path / "slurm-task-scope"
    server = json.loads((job / "app_server/config/config_fed_server.json").read_text())
    assert server["workflows"][0]["args"] == {"rounds": 4, "gap_seconds": 20.0}
    assert any(c["path"] == "nvflare.private.fed.task_scope.server.TaskScopedServer" for c in server["components"])
    for site in ("site-1", "site-2"):
        app = job / f"app_{site}"
        assert (app / "custom/components.py").is_file()
        client = json.loads((app / "config/config_fed_client.json").read_text())
        assert client["executors"][0]["executor"]["args"]["crash_round"] == 2
