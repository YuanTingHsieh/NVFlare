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
"""Compare the final train, durable state, and submit-model tensor bytes."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from nvflare.app_common.utils.fl_model_utils import FLModelUtils
from nvflare.private.fed.deployment_supervisor.artifacts import read_artifact


def _digest(params):
    digest = hashlib.sha256()
    for name in sorted(params):
        value = params[name]
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().contiguous().numpy()
        value = np.ascontiguousarray(value)
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(value.tobytes())
    return digest.hexdigest()


def _result(attempt_dir):
    spec = json.loads((attempt_dir / "worker.json").read_text())
    _, shareable = read_artifact(
        attempt_dir,
        attempt=spec["attempt"],
        job_id=spec["job_id"],
        task_id=spec["task_id"],
        kind="result",
    )
    model = FLModelUtils.from_shareable(shareable)
    return spec, _digest(model.params) if model.params else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--supervisor-job-dir", required=True)
    parser.add_argument("--state-file", required=True)
    args = parser.parse_args()

    attempts = []
    for path in Path(args.supervisor_job_dir).iterdir():
        if (path / "worker.json").is_file() and (path / "result.json").is_file():
            spec, digest = _result(path)
            attempts.append(
                {
                    "attempt": spec["attempt"],
                    "task_id": spec["task_id"],
                    "task_name": spec["task_name"],
                    "digest": digest,
                    "mtime_ns": (path / "result.json").stat().st_mtime_ns,
                }
            )
    trains = sorted((item for item in attempts if item["task_name"] == "train"), key=lambda item: item["mtime_ns"])
    submits = [item for item in attempts if item["task_name"] == "submit_model"]
    if len(trains) != 3 or len(submits) != 1:
        raise RuntimeError(f"expected 3 train and 1 submit_model results, got {len(trains)} and {len(submits)}")
    state_digest = _digest(torch.load(args.state_file, map_location="cpu", weights_only=True))
    if not trains[-1]["digest"] or trains[-1]["digest"] != submits[0]["digest"] or state_digest != submits[0]["digest"]:
        raise RuntimeError("final train, durable state, and submit_model tensor digests differ")
    print(
        json.dumps(
            {
                "final_train": trains[-1],
                "submit_model": submits[0],
                "state_digest": state_digest,
                "all_attempts": sorted(attempts, key=lambda item: item["mtime_ns"]),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
