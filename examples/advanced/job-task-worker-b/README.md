# Job-lifetime CPU CJ with fresh application workers (Architecture B experiment)

This bounded experiment keeps the ordinary client job (CJ) alive for the NVFlare
job lifetime. That CPU CJ owns the federation Cell/session, task fetch, job and
site filters, worker supervision, cancellation, result publication retries, and
the server ACK. It stages each already-filtered task to an immutable artifact,
launches a fresh application worker, waits for the process or Slurm allocation to
settle, validates the durable result, and only then returns it to the normal CJ
publication path.

The application script or Executor runs only in the fresh worker. The worker has
no Cell and receives none of the CJ authentication token, token signature, or
session credential. This is not the existing ClientAPIExecutor plus a resident
external trainer: both accepted source configurations are translated to one
fresh, Cell-free task worker. Attach and resident application workers are
unsupported.

Run the two-client, three-round synthetic hello-pt matrix locally:

```bash
python job.py --mode in_process --simulate /tmp/architecture-b-inproc
python job.py --mode external_process --simulate /tmp/architecture-b-external
```

For common A/B/D qualification, pass the frozen workload directory explicitly:

```bash
python job.py \
  --workload-dir /path/to/architecture-a-b-d-common/workload \
  --mode in_process \
  --dataset synthetic \
  --export /tmp/architecture-b-common-q04
```

With `--workload-dir`, omitted workload values select the common synthetic
defaults (32/16 samples, batch size 8, one epoch, no data-loader workers) or the
common CIFAR-10 defaults (batch size 64, one epoch, two workers). Both use seed
202610 plus the site offset. Slurm defaults request one node, four CPUs, 16 GiB,
and one GPU for every disposable task. The worker interpreter receives
`PYTHONHASHSEED` before startup; the common script seeds Python, NumPy, PyTorch,
CUDA, and its data-loader generator.

Add `--publication-delay 5` only for supplementary CPU-filter-delay evidence;
it does not qualify the common slow-network-upload row. A Slurm run also
requires `--slurm` and a site workspace template such as
`/shared/architecture-b/{site}`. The site may launch the CPU CJ locally or in a
separate Slurm allocation. When the CJ is Slurm-scheduled,
`SlurmTaskWorkerLauncher` intentionally creates its own scheduler manager so
each application task still receives a distinct allocation. The shared
workspace and Slurm command set must be reachable from that CJ allocation.

Application state does not survive in memory between workers. The framework
provides `NVFLARE_TASK_STATE_DIR` and `job_task_worker_state_dir`; applications
must explicitly checkpoint the exact state they need. The hello-pt script uses
that directory only to preserve its last local model for `submit_model`.
Returning the latest trained model is a common application requirement across
the candidate architectures; the file and state directory are only this
example's Architecture B implementation.
