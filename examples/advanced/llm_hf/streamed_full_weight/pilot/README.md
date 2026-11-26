# Tested Llama 3 70B Slurm job

The [one-round run report](https://github.com/YuanTingHsieh/NVFlare/blob/bdf729a42219efb48bce4ed34a5107b947e83620/examples/advanced/llm_hf/streamed_full_weight/pilot-report.pdf) covers this job. Two sites each ran one full-weight BF16 SGD step during one federated round. The server accepted both 141 GB updates, ran disk-backed FedAvg and archived the global model.

This directory contains the exact [training source](streamed_full_weight_smoke.py.txt) and the [NVFlare runtime wheel](nvflare-2.10.0.dev261119-py3-none-any.whl) from the run. The source has a `.txt` suffix to preserve it byte-for-byte. Copy it to a `.py` file before exporting. The wheel includes the tested Slurm and disk-streaming patches. The scheduler account, partitions, paths and service startup are site configuration.

## Prepare and export

Stage the [Meta Llama 3 70B base checkpoint](https://huggingface.co/meta-llama/Meta-Llama-3-70B) and tokenizer at revision `c82494877ce7f6d7d317c56ec081328e382c72fe`. Place the model and the campaign workspace on the same shared filesystem so checkpoint preparation can hard-link shards. Use a shared Linux Python environment with a CUDA-enabled PyTorch build on the GPU nodes.

```bash
# From the NVFlare repository root
pilot="$PWD/examples/advanced/llm_hf/streamed_full_weight/pilot"
wheel="$pilot/nvflare-2.10.0.dev261119-py3-none-any.whl"
model=/shared/models/Meta-Llama-3-70B
work=/shared/experiments/llama3-70b-tested

mkdir -p "$work"
printf '%s  %s\n' \
  689cef2be4814e42a8c2cc916e651eec344ecfd9033839acdbc9eba5ec90c267 "$wheel" \
  | sha256sum -c -
python3 -m venv "$work/runtime"
python="$work/runtime/bin/python"
"$python" -m pip install "$wheel" torch 'transformers>=5,<6'

cp "$pilot/streamed_full_weight_smoke.py.txt" "$work/streamed_full_weight_smoke.py"
"$python" "$work/streamed_full_weight_smoke.py" prepare "$model" --workspace "$work/prepared"
"$python" "$work/streamed_full_weight_smoke.py" export "$model" \
  --workspace "$work/jobs" --index "$work/prepared/model.safetensors.index.json" \
  --device cuda --rounds 1
```

The job folder is `$work/jobs/llama-streamed-full-weight-smoke`. Its metadata requests two clients, task-lifetime training and, per worker, one node with four GPUs, 8 GiB of host RAM and a 45-minute time limit.

## Place the services and workers

Provision a server and clients named `site-1` and `site-2` using the bundled runtime. Use the same environment in all five allocations. Set site-specific workspace paths, scheduler accounts and partitions as in the [deployment instructions](../README.md#deploy-on-slurm), with these settings for the tested job:

| Process | Allocation and launcher |
| --- | --- |
| Server parent and server job | One CPU allocation, 2 CPUs and 64 GiB RAM. The parent starts the job with `ServerProcessJobLauncher`. Use the server's disk-backed `TMPDIR` and `NVFLARE_DENSE_SERVER_DROP_FILE_CACHE=1`. |
| Client parent and client job | One CPU allocation per site, 2 CPUs and 16 GiB RAM. Each parent starts its job with `ClientProcessJobLauncher`. Set `NVFLARE_DENSE_DISK_TASK_TENSORS=1` before starting each parent. |
| GPU task worker | A separate Slurm allocation per site. Use `SlurmTaskLauncher` with 4 CPUs and the compute-reachable client-job hostname. Export `NVFLARE_DENSE_DISK_TASK_TENSORS=1` and set `CUDA_VISIBLE_DEVICES=0` in the worker setup. The job reserves four GPUs and the training script uses one. |

The launcher and admin monitor ran on the login node. The training and federation processes ran in the allocations above. The service allocations had a two-hour time limit. The client resources used `max_cj_restart_failures: 0` and `task_execution.poll_interval: 1`. Use shared NFS for the server spool and task artifacts. The run checked for approximately 1 TB of free campaign scratch and used a 45-minute result submission timeout.

As an NVFlare admin, submit the exported folder after both clients connect:

```text
submit_job /shared/experiments/llama3-70b-tested/jobs/llama-streamed-full-weight-smoke
```

The run reached `FINISHED:COMPLETED`. Its output archive contains `FL_global_model.safetensors`, which stores the aggregated weights as 4,653 pieces. The [restore command](../README.md#verify-and-restore-the-output) in the current-runtime example reassembles these pieces into Llama tensor names for Transformers.
