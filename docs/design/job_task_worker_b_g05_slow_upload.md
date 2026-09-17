# Architecture B bounded G05 slow-upload fixture

Status: reproducible bounded fixture. Private campaign artifacts and
machine-specific paths are not part of this repository.

This is one isolated observation row. It adds no executor phase or production
transport mechanism. A normal B task worker commits a 64 MiB test payload as
part of its ordinary result artifact. After the Slurm allocation settles, the
resident CPU CJ sends that same result through the normal NVFlare streamed
`SUBMIT_UPDATE` path. A server result filter verifies the received byte count
and SHA-256, removes only the test metadata, and leaves the real model update for
normal aggregation and ACK.

## Fixture identity

- Base source: `9759e2594dc6949ef80cedf739c244334fae5f70`.
- Runtime source: the branch commit used to export the job.
- Job name: `architecture-b-g05-slow-upload`.
- Topology: one server, one client, one train round, one fresh in-process-source
  task worker, `launch_once=false`, one four-CPU/16-GiB/one-GPU Slurm allocation.
- Slurm workspace: a shared, site-specific path supplied by the operator.
- Committed transport payload: 67,108,864 bytes; SHA-256
  `6fa5dbbc218aa0ca5e1280790474b4083d0d64576b6b76d08981db5a5fb552d8`.

The export command was:

```bash
python3 examples/advanced/job-task-worker-b/job.py \
  --mode in_process --clients 1 --rounds 1 --dataset synthetic \
  --train-size 32 --test-size 16 --epochs 1 --num-workers 0 \
  --train-only --transport-payload-mib 64 --slurm --gpus-per-node 1 \
  --cpus-per-node 4 --memory-gib 16 \
  --site-workspace '/shared/architecture-b-g05/{site}' \
  --export /tmp/architecture-b-g05-export
```

## Isolated transport control

Start the isolated B site processes with these environment values before their
NVFlare cells are constructed. They use existing NVFlare stream flow control:

```bash
export NVFLARE_STREAMING_CHUNK_SIZE=4096
export NVFLARE_STREAMING_WINDOW_SIZE=1
export NVFLARE_STREAMING_ACK_INTERVAL=1
export NVFLARE_STREAMING_SEND_TIMEOUT=600
export NVFLARE_STREAMING_ACK_PROGRESS_TIMEOUT=120
export NVFLARE_STREAMING_ACK_PROGRESS_CHECK_INTERVAL=1
```

The client parent and its CJ must inherit these values. Do not add them to the
Slurm task-worker environment: the intended controlled interval is the ordinary
CJ-to-server result stream after compute settlement. Use a fresh isolated site
startup or restart it before submission; setting these values after the Cell is
created does not reconfigure that Cell.

Before submitting the job, run the bundled observer on the client node:

```bash
python3 examples/advanced/job-task-worker-b/slow_upload_probe.py \
  --events-glob '/shared/architecture-b-g05/site-1/*/.job_task_worker_b/events.jsonl' \
  --min-active-seconds 3
```

Then submit the frozen exported job through the existing Colossus admin path.
The observer must start first so its audits occur while publication is active.

## Required assertions

The row passes only if all of these hold:

1. The source/export hashes above match, and the exported executor remains
   `ClientAPIJobTaskWorkerExecutor` in `in_process` source mode with
   `launch_once=false`.
2. Exactly one Slurm task allocation requests one GPU and reaches completed
   terminal state. The event ledger orders `allocation_settled` with
   `settled=true` before `durable_result_loaded`, `cpu_cj_publication_started`,
   and `cpu_cj_publication_acknowledged`.
3. The committed `result.fobs` exceeds 64 MiB, and its
   `durable_result_loaded.result_bytes` matches the same attempt sent by the CJ.
4. The observer reports at least three seconds from publication start to ACK.
   Every live audit in that interval must find the worker PID absent, its Slurm
   ID absent from `squeue`, and that worker PID absent from NVIDIA compute PIDs.
5. The client communicator log reports a `SubmitUpdate` larger than 64 MiB and
   a duration of at least three seconds. The server log reports verification of
   exactly 67,108,864 bytes with the expected digest, then an accepted model
   contribution. The ledger records a genuine positive ACK.
6. Immediate and +30-second checks find no B task worker, Slurm allocation, or
   GPU compute process. Standing isolated federation services are reported
   separately and then stopped by their normal cleanup path.

The 4 MiB local simulator check verified the full byte artifact, server digest
filter, accepted contribution, and positive ACK. It is not credited as the real
network observation.
