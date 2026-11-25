# Streamed full-weight Llama FedAvg

This proof of concept runs one full-weight SGD step per client per round and streams the complete BF16 model through disk-backed FedAvg.

## Tested 70B job

The [tested Slurm job](pilot/README.md) includes the exact training source and NVFlare wheel used for the completed Meta Llama 3 70B run. Two clients each took one full-weight BF16 SGD step in a single federated round. The [report](pilot-report.pdf) describes that job, its results, and its Slurm setup.

## Current-runtime example

The following instructions use `job.py`, the port to this branch's NVFlare runtime. It was validated with a three-round, two-client CPU simulation on a tiny BF16 Llama, including checkpoint restoration. Its Slurm export and site configuration were checked locally. To reproduce the completed 70B run, follow the tested job instructions above.

## Prepare and export the job

Install this branch and the PyTorch and Transformers dependencies. Stage an untied Llama safetensors checkpoint and tokenizer on shared storage. The 70B configuration targets the [Meta Llama 3 70B base checkpoint](https://huggingface.co/meta-llama/Meta-Llama-3-70B) at revision `c82494877ce7f6d7d317c56ec081328e382c72fe`.

```bash
# From the NVFlare repository root
python -m pip install -e . torch 'transformers>=5,<6'

example=examples/advanced/llm_hf/streamed_full_weight
model=/shared/models/Meta-Llama-3-70B
work=/shared/experiments/llama3-70b

mkdir -p "$work"
python "$example/job.py" prepare "$model" --workspace "$work/prepared"
python "$example/job.py" export "$model" \
  --index "$work/prepared/model.safetensors.index.json" \
  --workspace "$work/jobs" --device cuda --rounds 1
```

The preparation step splits large tensors into pieces of up to 32 MiB. It hard-links shards that still contain unsplit tensors, so the original checkpoint and prepared directory must be on the same filesystem. The model, prepared index and experiment paths must be visible at the same absolute paths from the relevant processes.

The exported job is `$work/jobs/llama-streamed-full-weight-smoke`. Submit that folder with the admin client from a provisioned NVFlare deployment:

```text
submit_job /shared/experiments/llama3-70b/jobs/llama-streamed-full-weight-smoke
```

## Deploy on Slurm

Provision one server and clients named `site-1` and `site-2`. Run the server parent (SP) and server job (SJ) together in a 64 GiB CPU Slurm allocation with the process server job launcher. Run each client parent (CP) in its own 16 GiB CPU Slurm allocation with scheduler access. `ClientProcessJobLauncher` starts the client job (CJ) as a process in that same allocation. The CJ requests GPU task workers through its CP, which submits them to Slurm. Each worker runs in a separate allocation.

Each worker requests one GPU, 8 GiB of host RAM and 45 minutes in Slurm. Set `--gpus-per-node` if your site's scheduler requires a larger allocation. The training script uses one visible GPU. Size the SP and CP allocations to cover the whole campaign, including scheduler waits and result uploads.

On the server, create a private, disk-backed temporary directory and set these environment variables before starting the parent. The server job inherits them.

```bash
export TMPDIR=/shared/experiments/llama3-70b/server-tensor-spool
export NVFLARE_DENSE_SERVER_DROP_FILE_CACHE=1
mkdir -p -m 700 "$TMPDIR"
```

Choose a disk-backed filesystem for `TMPDIR`. Two 70B BF16 updates occupy about 282 GB before aggregation. The global checkpoint and its archive each occupy about 141 GB. Allow additional space for the initial checkpoint, client artifacts and in-progress files.

Export `NVFLARE_DENSE_DISK_TASK_TENSORS=1` before starting each CP so its CJ keeps verified task tensors on disk. Merge these settings into each client's `local/resources.json`. Set the client workspace, the compute-reachable host of the CP/CJ allocation, and the Slurm account and GPU partition for that site. See the [Slurm task deployment guide](../../../../docs/user_guide/admin_guide/deployment/slurm_task_launcher.rst#process-cj-on-a-login-node-or-submit-host).

```json
{
  "components": [
    {
      "id": "job_launcher",
      "path": "nvflare.app_common.job_launcher.client_process_launcher.ClientProcessJobLauncher"
    }
  ],
  "task_execution": {
    "artifact_cleanup": "receipt",
    "worker_timeout": 2400,
    "settlement_timeout": 180,
    "finalization_timeout": 180,
    "launcher": {
      "path": "nvflare.app_opt.job_launcher.slurm.task_launcher.SlurmTaskLauncher",
      "args": {
        "workspace_path": "/shared/flare/site-1",
        "worker_listener_host": "client-cpu-node.example.org",
        "python_path": "/shared/venv/bin/python",
        "shared_filesystem": true,
        "cgroup_enforced": true,
        "cpu_cores": 4,
        "memory_mb": 8192,
        "gpu_count": 1,
        "pending_timeout": 1800,
        "sbatch_directives": {
          "partition": "batch",
          "account": "my-slurm-account"
        },
        "environment": {
          "NVFLARE_DENSE_DISK_TASK_TENSORS": "1"
        }
      }
    }
  }
}
```

The client site settings above allow 40 minutes for the GPU worker. The job configures a 45-minute result submission timeout.

## Verify and restore the output

Check that the workflow reports `Aggregated 2/2 results` and the job reaches `FINISHED:COMPLETED`. Extract `FL_global_model.safetensors` from the job's output archive. It stores the federated weights under NVFlare's tiled keys. Restore the original Llama tensor names and shapes before loading it with Transformers:

```bash
python "$example/job.py" restore "$model" \
  --checkpoint "$work/output/FL_global_model.safetensors" \
  --workspace "$work/restored-model"
```

The restored directory contains `model.safetensors`, the model config and the tokenizer. Restoration validates tile coverage, shapes and dtypes against the base checkpoint. It needs room on disk for a second copy of the weights and uses host memory for one full parameter plus one tile at a time.

Local validation used a tiny BF16 Llama checkpoint for three rounds with two simulated clients. Every saved tensor matched the mean of the two client updates, both transformer blocks changed, and Transformers loaded the restored model and produced finite logits. The Slurm job export and site configuration were checked separately.

Run the CPU integration check from the repository root. It generates its own checkpoint and tokenizer.

```bash
python -m pip install -e . pytest torch 'transformers>=5,<6'
python -m pytest -q tests/integration_test/slow/llama_streamed_full_weight_test.py
```

## Runtime changes

NVFlare already provides task execution, disk-backed FedAvg, and disk-backed tensor streaming. The pilot changes carried in this branch avoid read-ahead during checkpoint header scans, keep verified task artifacts as lazy references, advise the kernel to drop cached streaming pages, and release settled stream task handles. The two environment variables are process-level pilot switches. The page-cache switch is read when the streaming module loads, so set it only on the server for this example. The example uses internal lazy-tensor types and is tied to this branch.

`job.py` ports the [tested training script](pilot/streamed_full_weight_smoke.py.txt) and uses the Process coordinator support from [PR #5771](https://github.com/NVIDIA/NVFlare/pull/5771).
