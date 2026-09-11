# Experimental phased-D execution with Slurm

This prototype runs ordinary synchronous tasks as three sequential CJs:

```text
CP: availability probe → supervise each phase → wait for next task
                         │
                  CPU pull CJ
                         │ input.fobs + input.json
                  GPU compute CJ
                         │ result.fobs + result.json
                  compute allocation settles
                         │
                  CPU push CJ → existing task-result submission / ACK
```

CP handles readiness and physical allocation supervision. Task payloads,
Executors, filters, Cell communication and publication remain in CJs. Each phase
builds the full CJ stack, with its own process and Slurm allocation. This retains
D's application/runtime trust boundary; it does not isolate application code
from Cell credentials as proposed in A/B.

All phases use the existing `nvflare.private.fed.app.client.worker_process`
entrypoint. Its `--set` options select the phase runner and receipt policy;
there is no separate task-scope worker executable.

The compute allocation must be terminal before CP submits the CPU push phase.
Saving a local result does not mean the server has received it or completed the
job. The logical client handle remains active through upload/ACK and idle gaps.

## Configure production sites

Follow the existing [Slurm launcher setup](../../../docs/user_guide/slurm_job_launcher.rst).
Provision one server and two client startup kits. CP must run without a reserved
GPU. Each site's workspace must survive worker exit and be mounted at the same
absolute path on CP and all CPU/GPU nodes. Every process uses the same revision.

Keep the existing `nvflare.app_opt.job_launcher.slurm.ClientSlurmJobLauncher` in
each CP's `local/resources.json`, preserving scheduler commands, account,
partition, Python path, mounts, and resource-manager settings. Add:

```json
{
  "task_scoped": true,
  "task_phased": true,
  "task_probe_interval": 2.0,
  "task_probe_timeout": 5.0,
  "task_communication_timeout": 120.0
}
```

`task_phased` requires `task_scoped=true`. Pull/push use the job's CPU/memory
request with no GPU GRES and empty CUDA/ROCm device visibility. Compute uses the
original resource request. The configured partition must accept CPU allocations;
this prototype does not select a different partition per phase. Keep the
existing server launcher configuration.

For comparison, `task_scoped=true, task_phased=false` selects the earlier whole-CJ
per-task baseline, which retains its GPU through upload and cleanup.
`task_scoped=false` retains the standard job-lifetime CJ.

## Export and submit the example

```bash
python job.py --output /absolute/test-job-exports --clients site-1 site-2 --rounds 3 --gap-seconds 30 --gpus 1
```

Submit the exported `slurm-task-scope` directory through the normal admin job
workflow. The server app includes
`nvflare.private.fed.task_scope.server.TaskScopedServer`. Its authenticated CP
probe advertises ordinary broadcast/send work without assigning or pulling the
payload. Only the pull CJ makes the normal task request. A stale readiness hint
can yield TRY_AGAIN, in which case no compute/push allocation is submitted.

The example counter restores explicitly checkpointed state in each compute CJ,
returns values 1, 2, 3, and records its PID/Slurm ID. The server waits for all
clients and deliberately leaves a gap after every round. It requests a GPU but
does not run CUDA kernels; use hello-pt for actual training validation. Increase
task deadlines to cover all three queue/startup periods plus transfer/compute.
Use `--gpus 0` only for CPU smoke tests. For a crash test, export a new job with
`--crash-round 1`; the compute CJ exits 1 before checkpointing that round.

## Handoff and lifecycle contract

| Phase | Work | Successful local receipt |
|---|---|---|
| Pull / CPU | Fetch assignment; persist eager input and server public context | INPUT_READY, or IDLE/END_RUN control response |
| Compute / GPU | Restore input; run normal data filters, Executor, result filters; persist eager result | RESULT_READY |
| Push / CPU | Restore result; run send events and existing task-check/send/ACK path; workspace publication | TASK_COMPLETE |

Artifacts are in `<workspace>/<job_id>/.task_scope/<attempt>/`. Each input/result
uses a native FOBS stream plus a manifest containing version, job/attempt/task
identity, originating task session ID, byte count and SHA256. The manifest is
published last after file sync.
The next phase validates it before decoding. Partial, corrupt and stale artifacts
cannot advance the pipeline. No model payload is decoded in CP. Shared storage
must support hard links and directory fsync for the exclusive publication.

Compute and push validate the originating session against their current client
session and restore the task-local context that ordinary task pull initializes.
Missing or stale sessions fail the phase instead of retrying result submission.
Version-1 artifacts from the earlier prototype lack this session binding and
cannot be resumed. The existing communicator session check remains enabled;
transient network retries retain their existing behavior.

Each phase writes its own `<phase>/receipt.json` after worker cleanup. CP checks
the physical exit code and receipt after the Slurm handle settles. A saved
result cannot override a failed allocation. TASK_COMPLETE means the existing
result-submission ACK, not a new durable server commit protocol. Input/result
artifacts remain in the job workspace for diagnosis and are not automatically
retried or deleted after ACK in this prototype.

All components must tolerate construction and START_RUN/END_RUN per phase,
including CPU phases that never execute a task. Every Executor must declare
`supports_task_scoped_process = True`; this is an author assertion, not a proof.
Cross-task state belongs in the task or durable workspace. Data/result filters
and execution events run in compute. BEFORE_SEND/AFTER_SEND run in push; handlers
there cannot require GPUs. No in-memory FLContext or component state is carried
between phases. Pull and compute retain process cleanup but defer workspace
upload to push. Per-process logs/events are not a new final-log completeness
protocol.

The CP's logical handle continues to appear in its job list while CJs are absent.
SP/SJ therefore retain participation through phase queues and idle gaps. Actual
CP loss still invokes existing dead-client policy. Parent restart/adoption,
retries after a lost ACK, and durable coordinator recovery are not provided.

## Required production evidence

`events.jsonl` under `.task_scope/` records submission, returned scheduler ID,
allocation settlement and receipt, with attempt and `task_phase` values.
`allocated` means sbatch returned an ID, not that the allocation is RUNNING.
Correlate these records with `squeue`, `sacct`, `scontrol`, worker logs and GPU
sampling. The decisive ordering is:

```text
result manifest committed
  < compute CJ exit
  < compute Slurm allocation terminal
  < push sbatch submission
  < result network send
  < submission ACK
```

| Scenario | Acceptance |
|---|---|
| Two clients, three rounds | 18 distinct phase allocations; expected values/model result; successful final server status |
| Pull and push | No GPU GRES in actual allocation accounting |
| Slow result upload | CPU push remains active while compute allocation and GPU reservation are absent |
| Faster client | Its GPU is released while the slower client's compute and federation remain active |
| Next round | Fresh GPU allocation resumes correctly from task/file state |
| Idle gap | All CJs may be absent; logical job remains alive and resumes |
| Compute crash / SIGKILL | No push phase; bounded failure, no success from saved files |
| Partial artifact / disk failure | No dependent phase; preserve evidence |
| Push fails or ACK missing | No invented success and no automatic recomputation |
| Abort idle/pending/running | No later phase; owned allocation cancellation settles |
| Default launcher control | `task_scoped=false` retains standard behavior |

## Scope and validation

The shared lifecycle lives in `nvflare/private/fed/task_scope/`; the existing
Slurm launcher supplies physical handles and GPU-free transfer resource plans.
No new scheduler implementation is introduced. Single-node allocations and
eager, synchronous ordinary broadcast/send tasks are supported. Multi-node/DDP,
lazy/pass-through results, Attach, CCWF/aux tasks and unchanged stateful legacy
Executors are outside this prototype. Full CJs still initialize on CPU nodes;
applications that allocate GPUs unconditionally at initialization must adapt.

Tests under `tests/unit_test/private/fed/task_scope/` and
`tests/unit_test/app_opt/job_launcher/` check handoffs, phase ordering, rejection,
resource plans and default-mode regressions. Unit tests do not prove scheduler
release or production networking. The production Colossus phased-D result must
be reported separately with its exact commit and artifacts.

Earlier production baseline `bfe583a9e` demonstrated one site's allocation
release after result ACK while the other site ran, but failed after round 0 due
to example JSON serialization of runtime headers. The example now records only
its application fields. That baseline does not establish phased-D success.

The first phased-D production hello-pt run at `d47f985c5` demonstrated GPU
allocation release before CPU push submission, but failed because the fresh
push CJ lacked the task-local session context. No round completed; the run was
aborted for cleanup. The session handoff above addresses that defect; production
qualification of the corrected revision must still be reported separately.
