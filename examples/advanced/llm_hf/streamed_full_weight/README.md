# Streamed full-weight Llama FedAvg

This proof of concept runs one full-weight SGD step per client and streams the complete BF16 model through disk-backed FedAvg. The [pilot report](pilot-report.pdf) records the two-client Llama 3 70B run, including process placement, timings and measured memory.

## Prepare and export the job

Install this branch and the PyTorch and Transformers dependencies. Stage an untied Llama safetensors checkpoint and tokenizer on shared storage. The model used in the pilot was the [Meta Llama 3 70B base checkpoint](https://huggingface.co/meta-llama/Meta-Llama-3-70B) at revision `c82494877ce7f6d7d317c56ec081328e382c72fe`.

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
  --workspace "$work/jobs" --device cuda --rounds 1 --gpus-per-node 4
```

The preparation step splits large tensors into pieces of up to 32 MiB. It hard-links shards that still contain unsplit tensors, so the original checkpoint and prepared directory must be on the same filesystem. The model, prepared index and experiment paths must be visible at the same absolute paths from the relevant processes.

The exported job is `$work/jobs/llama-streamed-full-weight-smoke`. Submit that folder with the admin client from a provisioned NVFlare deployment:

```text
submit_job /shared/experiments/llama3-70b/jobs/llama-streamed-full-weight-smoke
```

## Deploy on Slurm

Provision one server and clients named `site-1` and `site-2`. Run the server parent and server job together in a 64 GiB CPU Slurm allocation with the process server job launcher. Start each client parent (CP) on a CPU host with scheduler access. On this branch, `ClientSlurmJobLauncher` submits a separate CPU client job (CJ) allocation, and `SlurmTaskLauncher` runs each `train` task in a separate GPU allocation. In the recorded pilot, each CP and its CJ shared a 16 GiB CPU allocation.

Use the [Slurm task deployment guide](../../../../docs/user_guide/admin_guide/deployment/slurm_task_launcher.rst) to prepare each client's workspace. In the generated `ClientSlurmJobLauncher` component, set `cj_memory_mb` to `16384`, use the CPU partition and include `export NVFLARE_DENSE_DISK_TASK_TENSORS=1` in the launcher's `setup` so the CJ keeps verified task tensors on disk. Each GPU worker requests 8 GiB of host RAM and uses one CUDA device. The exported job requests four GPUs per worker by default. Set `--gpus-per-node` to match your cluster. The exported Slurm time limit defaults to two hours for both the CJ and task allocations. Increase it with `--time` for longer campaigns.

On the server, create a private, disk-backed temporary directory and set these environment variables before starting the parent. The server job inherits them.

```bash
export TMPDIR=/shared/experiments/llama3-70b/server-tensor-spool
export NVFLARE_DENSE_SERVER_DROP_FILE_CACHE=1
mkdir -p -m 700 "$TMPDIR"
```

Choose a disk-backed filesystem for `TMPDIR`. Two 70B BF16 updates occupy about 282 GB before aggregation. The global checkpoint and its archive each occupy about 141 GB. Allow additional space for the initial checkpoint, client artifacts and in-progress files. The pilot used a 1 TB free-space check for campaign scratch.

Add a `task_execution` section to each client's local resources configuration. Use the same workspace as the client job launcher and set the Slurm account and GPU partition for that site:

```json
{
  "task_execution": {
    "artifact_cleanup": "receipt",
    "worker_timeout": 2400,
    "settlement_timeout": 180,
    "finalization_timeout": 180,
    "launcher": {
      "path": "nvflare.app_opt.job_launcher.slurm.task_launcher.SlurmTaskLauncher",
      "args": {
        "workspace_path": "/shared/flare/site-1",
        "python_path": "/shared/venv/bin/python",
        "shared_filesystem": true,
        "cgroup_enforced": true,
        "cpu_cores": 4,
        "memory_mb": 8192,
        "gpu_count": 4,
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

The client site settings above allow 40 minutes for the GPU worker. The job configures a 45-minute result submission timeout. In the pilot, each 141 GB upload completed in about 14 minutes on its first attempt.

## Verify the output

Check that the workflow reports `Aggregated 2/2 results` and the job reaches `FINISHED:COMPLETED`. The job's output archive contains `FL_global_model.safetensors`. The pilot counted 70,553,706,496 parameters in the saved checkpoint and confirmed that a 32-row slice of the output projection changed from the base model. The test used one synthetic prompt per client and measured execution and memory usage.
