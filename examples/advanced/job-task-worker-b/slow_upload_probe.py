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
"""Observe the isolated G05 transfer while proving its Slurm worker is gone."""

import argparse
import glob
import json
import os
import subprocess
import time


def _events(path):
    try:
        with open(path) as stream:
            return [json.loads(line) for line in stream if line.strip()]
    except FileNotFoundError:
        return []


def _run(command):
    return subprocess.run(command, check=True, capture_output=True, text=True).stdout.strip()


def _audit(worker_pid, scheduler_id, squeue, nvidia_smi):
    try:
        os.kill(worker_pid, 0)
    except ProcessLookupError:
        worker_alive = False
    else:
        worker_alive = True
    slurm = _run([squeue, "-h", "-j", str(scheduler_id), "-o", "%A:%T"])
    gpu_pids = {
        int(line.strip())
        for line in _run([nvidia_smi, "--query-compute-apps=pid", "--format=csv,noheader,nounits"]).splitlines()
        if line.strip().isdigit()
    }
    if worker_alive or slurm or gpu_pids:
        raise RuntimeError(
            f"compute not settled during upload: worker_alive={worker_alive}, slurm={slurm!r}, gpu_pids={gpu_pids}"
        )
    return {"worker_alive": worker_alive, "slurm": slurm, "gpu_pids": sorted(gpu_pids)}


def _execution_identity(records, attempt):
    launched = next(
        record for record in records if record.get("event") == "worker_launched" and record.get("attempt") == attempt
    )
    loaded = next(
        record
        for record in records
        if record.get("event") == "durable_result_loaded" and record.get("attempt") == attempt
    )
    worker_pid = loaded.get("worker_pid", launched.get("worker_pid"))
    scheduler_id = launched.get("scheduler_id")
    if isinstance(worker_pid, bool) or not isinstance(worker_pid, int) or worker_pid <= 0:
        raise RuntimeError("G05 event ledger does not identify the completed worker PID")
    if not isinstance(scheduler_id, (str, int)) or not str(scheduler_id):
        raise RuntimeError("G05 event ledger does not identify the Slurm allocation")
    return worker_pid, scheduler_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--events-glob", required=True)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--min-active-seconds", type=float, default=3.0)
    parser.add_argument("--squeue", default="/usr/bin/squeue")
    parser.add_argument("--nvidia-smi", default="/usr/bin/nvidia-smi")
    args = parser.parse_args()

    deadline = time.monotonic() + args.timeout
    event_path = None
    while time.monotonic() < deadline:
        paths = glob.glob(args.events_glob)
        if len(paths) > 1:
            raise RuntimeError(f"events glob is ambiguous: {paths}")
        if paths:
            event_path = paths[0]
            break
        time.sleep(0.1)
    if not event_path:
        raise TimeoutError("timed out waiting for Architecture B event ledger")

    publication = None
    audits = []
    while time.monotonic() < deadline:
        records = _events(event_path)
        if publication is None:
            publication = next(
                (record for record in records if record.get("event") == "cpu_cj_publication_started"), None
            )
        if publication:
            attempt = publication.get("attempt")
            settled = next(
                record
                for record in records
                if record.get("event") == "allocation_settled" and record.get("attempt") == attempt
            )
            if not settled.get("settled") or settled["monotonic_ns"] >= publication["monotonic_ns"]:
                raise RuntimeError("publication started before the compute allocation settled")
            worker_pid, scheduler_id = _execution_identity(records, attempt)
            acknowledged = next(
                (
                    record
                    for record in records
                    if record.get("event") == "cpu_cj_publication_acknowledged" and record.get("attempt") == attempt
                ),
                None,
            )
            if not audits:
                if acknowledged:
                    raise RuntimeError("observer did not sample before the G05 upload was acknowledged")
                now = time.monotonic()
                audits.append(
                    {
                        "observed_monotonic": now,
                        **_audit(worker_pid, scheduler_id, args.squeue, args.nvidia_smi),
                    }
                )
            if acknowledged:
                active_seconds = (acknowledged["monotonic_ns"] - publication["monotonic_ns"]) / 1_000_000_000
                if active_seconds < args.min_active_seconds:
                    raise RuntimeError(
                        f"actual result transport lasted only {active_seconds:.3f}s; expected >= {args.min_active_seconds:.3f}s"
                    )
                print(
                    json.dumps(
                        {
                            "events": event_path,
                            "attempt": attempt,
                            "worker_pid": worker_pid,
                            "scheduler_id": scheduler_id,
                            "allocation_settled_ns": settled["monotonic_ns"],
                            "publication_started_ns": publication["monotonic_ns"],
                            "publication_acknowledged_ns": acknowledged["monotonic_ns"],
                            "transport_active_seconds": active_seconds,
                            "audits": audits,
                        },
                        sort_keys=True,
                    )
                )
                return
        time.sleep(0.05)
    raise TimeoutError("timed out waiting for acknowledged G05 publication")


if __name__ == "__main__":
    main()
