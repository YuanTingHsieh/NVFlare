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

CP handles readiness and physical allocation supervision. Task payloads and Cell
communication remain in CJs, but framework bootstrap selects the component graph
before constructing it. Pull builds only the transfer runtime. Compute builds the
application graph containing Executors, Learners and filters. Push builds the
transfer runtime plus explicitly registered publication components. Each phase
still has its own process and Slurm allocation. This retains D's application/runtime
trust boundary; it does not isolate compute application code from Cell credentials
as proposed in A/B.

All phases use the existing `nvflare.private.fed.app.client.worker_process`
entrypoint. Its `--set` options select the phase runner and receipt policy;
there is no separate task-scope worker executable.

The compute allocation must be terminal before CP submits the CPU push phase.
Saving a local result does not mean the server has received it or completed the
job. The logical client handle remains active through upload/ACK and idle gaps.

## Configure production sites

Follow the existing [Slurm launcher setup](../../../docs/user_guide/admin_guide/deployment/slurm_job_launcher.rst).
Provision one server and two client startup kits. CP must run without a reserved
GPU. For this prototype, each site must provide a persistent POSIX workspace
shared by its CP and every Slurm CPU/GPU execution node, mounted at the same
absolute path. Pull, compute and push run in separate allocations and exchange
their artifacts through this directory, so node-local or allocation-ephemeral
storage cannot be used. Every process uses the same revision.

This is a Slurm-specific requirement of the prototype, not a general NVFlare
workspace guarantee. The current Kubernetes launcher stages a workspace into a
pod-local `emptyDir`, and the Docker launcher maps the host job directory to a
different container path. Consequently, `task_phased=true` does not currently
support those workspace topologies. Supporting them requires an explicit phase
artifact handoff through CP or an artifact service, a shared persistent volume,
or path translation; enabling this flag alone is insufficient.

Keep the existing `nvflare.app_opt.job_launcher.slurm.ClientSlurmJobLauncher` in
each CP's `local/resources.json`, preserving scheduler commands, account,
partition, Python path, mounts, and resource-manager settings. Add:

```json
{
  "task_phased": true,
  "task_probe_interval": 2.0,
  "task_probe_timeout": 5.0,
  "task_communication_timeout": 120.0,
  "task_transfer_timeout": 600.0
}
```

`task_phased` is the only lifecycle switch:

| Setting | Client lifecycle |
|---|---|
| `task_phased=false` (default, or omitted) | Original job-lifetime CJ |
| `task_phased=true` | Phased D: CPU pull → GPU compute → release compute allocation → CPU push |

The framework—not an Executor—selects these phases. An Executor must not import
the task-scope protocol or branch on pull/push. Pull and push do not construct the
configured Executor or its Learner dependencies. Compute uses the ordinary client
configuration and filtering pipeline. The framework-injected `JobLogStreamer` is
retained in transfer processes so their logs follow the existing streaming path.

An application component that must observe publication can be registered explicitly
in `config_fed_client.json`:

```json
{
  "task_scope_publication": {
    "components": [
      {
        "id": "publication_audit",
        "path": "custom.PublicationAudit",
        "args": {}
      }
    ]
  }
}
```

The component must be an `FLComponent` and declare
`supports_task_scope_publication = True`. It and nested dependencies are constructed
only in push, receive START_RUN/END_RUN there, and observe the real
BEFORE_SEND_TASK_RESULT, network submission/ACK and AFTER_SEND_TASK_RESULT ordering.
At AFTER_SEND_TASK_RESULT, the private FLContext property
`__task_scope_publication_ack` records whether the server acknowledged submission.
Publication components must be CPU-safe and recover all required state from the
persisted result or other durable storage. Merely listening for send events in the
ordinary compute component graph does not migrate a legacy handler. A handler that
requires both transient trainer memory and the later ACK must be split or given an
explicit durable state handoff; otherwise the configuration is unsupported.

Pull/push use the job's CPU/memory
request with no GPU GRES and empty CUDA/ROCm device visibility. Compute uses the
original resource request. The configured partition must accept CPU allocations;
this prototype does not select a different partition per phase. Keep the
existing server launcher configuration.

The CP continues authenticated server probes while a physical phase is active.
Sustained server loss for `task_communication_timeout` cancels a still-executing
phase. Slurm's existing `pending_timeout` remains the only queue deadline. After
Slurm reports that a pull or push allocation has started,
`task_transfer_timeout` bounds its execution and transfer; healthy compute has
no task-scope wall limit. Once push execution finishes, server completion or
disappearance does not turn delayed Slurm accounting into an execution failure.
The CP retains ownership until the physical launcher settles, then validates the
launcher return code and phase receipt. Without an authoritative server terminal
outcome, the logical job still cannot claim success.

The earlier whole-CJ-per-task baseline is no longer selectable. Remove the old
`task_scoped` launcher argument from prototype configurations; it is not accepted.
These are site-level startup settings, not per-job or live toggles. Finish active
jobs before changing the configuration and restarting CP; parent restart/adoption
is unsupported. The SJ task-availability endpoint is built into the framework;
enabling phased D still requires the shared workspace and compatible executors
described below.

## Export and submit the example

```bash
python job.py --output /absolute/test-job-exports --clients site-1 site-2 --rounds 3 --gap-seconds 30 --gpus 1
```

Submit the exported `slurm-task-scope` directory through the normal admin job
workflow. Every SJ automatically installs the framework-owned task-scope server
endpoint; the job does not configure it as an application component. Its
authenticated CP probe advertises ordinary broadcast/send work without assigning
or pulling the payload. Only the pull CJ makes the normal task request. A stale
readiness hint can yield TRY_AGAIN, in which case no compute/push allocation is
submitted.

The example counter restores explicitly checkpointed state in each compute CJ,
returns values 1, 2, 3, and records its PID/Slurm ID. The server waits for all
clients and deliberately leaves a gap after every round. It requests a GPU but
does not run CUDA kernels; use hello-pt for actual training validation. Increase
task deadlines to cover all three queue/startup periods plus transfer/compute.
Use `--gpus 0` only for CPU smoke tests. For a crash test, export a new job with
`--crash-round 1`; the compute CJ exits 1 before checkpointing that round.

## Identity and serialization invariants

`job_id` remains the logical NVFlare run ID. It identifies the client site's
participation for the whole run, not one physical CJ or Slurm allocation. The CP
holds one logical task-scope job handle under this ID while any number of
sequential task attempts and phase allocations come and go.

The prototype uses the following identity hierarchy:

| Scope | Identity |
|---|---|
| Logical federation run | `job_id` |
| Site participating in the run | `(site_name, job_id)` |
| One local task attempt | `(site_name, job_id, attempt)` |
| One physical phase CJ | `(site_name, job_id, attempt, phase)` |
| Assigned server task | `task_id` |
| Scheduler allocation | Slurm job ID |

Pull, compute and push CJs for the same site and run deliberately reuse the
existing job Cell FQCN, `site_name.<job_id>`. Attempt, phase and Slurm allocation
identity are recorded in local artifacts, receipts and diagnostics; they are not
part of the Cell address and are not visible to the existing task protocol.

Correctness therefore depends on all of these invariants:

1. At most one CJ for a given `(site_name, job_id)` may be active or connecting
   at a time. Phase allocations and task attempts must never overlap.
2. CP must observe the current allocation as terminal and fully settled before
   submitting the next phase that reuses the same Cell FQCN. Allocation exit is
   assumed to close the old Cell route before that address is reused.
3. The logical handle, not an individual CJ, owns the site's job lifetime. An
   expected gap with no CJ must not be interpreted as completion or client-job
   death.
4. No workflow may require unsolicited CJ-addressed communication during an
   idle or inter-phase gap. The built-in SJ endpoint communicates with the
   persistent CP for readiness and terminal state instead.
5. Every phase must use the same authenticated job session. Persisted artifacts
   bind the logical job, task, attempt and session; receipts bind the attempt and
   phase. A mismatch or stale handoff fails rather than advancing the pipeline.
6. The shared attempt directory must remain durable and visible at the same
   absolute path until pull, compute and push have settled.

Consequently, this prototype does not support concurrent task workers for the
same site and job, overlapping phase startup/teardown, or precise on-wire
attribution of a delayed message to a physical incarnation. Supporting those
semantics requires either unique worker Cell identities or communication owned
by a persistent CP/helper rather than the transient workers.

This restriction is scoped to one `(site_name, job_id)` pair, not to the whole
site. Different jobs have different job IDs and Cell FQCNs, so their logical
handles and phase workers may run concurrently subject to the site's normal
resource-management and scheduler policy.

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

Every Executor must declare `supports_task_scoped_process = True`; this is an
author assertion, not a proof. Cross-task state belongs in the task or durable
workspace. Data/result filters and execution events run in compute.
BEFORE_SEND/AFTER_SEND run in push with the explicitly registered publication
graph. No in-memory FLContext or component state is carried between compute and
publication. Pull and compute retain process cleanup but defer workspace upload
to push. Per-process logs/events are not a new final-log completeness protocol.

`ClientAPIExecutor` supports task-scoped compute in `in_process` mode and in
`external_process` mode with `launch_once=False`. The latter launches and tears
down its managed trainer inside each compute process. Attach and external
`launch_once=True` retain process/session state and are rejected. The Client API
executor remains unaware of pull and push; the compute runner's generic local
result-consumer contract materializes external results before artifact commit.

The CP's logical handle continues to appear in its job list while CJs are absent.
SP/SJ therefore retain participation through phase queues and idle gaps. Actual
CP loss still invokes existing dead-client policy. Parent restart/adoption,
retries after a lost ACK, and durable coordinator recovery are not provided.
An abort closes phase admission immediately. The existing bounded CJ teardown
grace remains in place, after which any still-active allocation is terminated.

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
| Default launcher control | `task_phased=false` or omitted retains standard behavior |

## Scope and validation

The shared lifecycle lives in `nvflare/private/fed/task_scope/`; the existing
Slurm launcher supplies physical handles and GPU-free transfer resource plans.
No new scheduler implementation is introduced. Single-node allocations and
eager, synchronous ordinary broadcast/send tasks are supported. Multi-node/DDP,
unresolved lazy/pass-through results, Attach, CCWF/aux tasks and unchanged
stateful legacy Executors are outside this prototype. CPU transfer CJs initialize
only framework transport/logging components and explicitly registered publication
components; the application graph is confined to compute.

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
