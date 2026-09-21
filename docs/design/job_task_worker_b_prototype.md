# Job-lifetime CPU CJ with fresh task workers (Architecture B)

Status: bounded prototype and qualification record, not a merge proposal.

## Decision summary

Architecture B keeps the ordinary NVFlare client job (CJ) alive on CPU for the
job lifetime and moves only application execution into a fresh process or Slurm
allocation for each task. The resident CJ continues to own the federation Cell,
task acquisition, normal task and result filters, cancellation, publication,
retry, and server acknowledgement. The task worker is deliberately Cell-free.

This prototype establishes that the split is feasible for eager `Shareable`
payloads and applications that either are task-stateless or explicitly
checkpoint their required state. It is not yet a general replacement for the
current executor model. In particular, resident application sessions, auxiliary
messaging, lazy/pass-through payloads, distributed workers, crash adoption, and
exactly-once application state transitions are outside the boundary.

The principal benefit is an unambiguous GPU lifetime: the task allocation has
settled and the result is durably present in the site workspace before the CPU
CJ starts result filtering and federation publication. Releasing compute before
server acceptance necessarily leaves an acceptance-knowledge interval in A, B,
and D. B's specific cost is that one resident CJ is the sole post-compute owner
and its pending-publication record is memory-only; this prototype cannot adopt
the durable attempt after that CJ/site fails.

## Ownership and topology

| Concern | Resident CPU CJ | Fresh task worker |
|---|---|---|
| Lifetime | NVFlare job lifetime | One task attempt |
| Federation Cell/session | Owns | None |
| Task acquisition and task filters | Owns | Receives one already-filtered task |
| Application code | Does not run it | Runs it |
| Slurm allocation | May own a job-long CPU allocation | Owns a distinct one-node task allocation |
| Cancellation | Requests and verifies settlement | Is terminated with its allocation/process group |
| Result durability | Validates committed worker artifact | Writes payload, manifest, and receipt |
| Result filters/publication/retry/ACK | Owns through the normal `ClientRunner` path | None |

`JobTaskWorkerExecutor` is infrastructure, even though it implements the
`Executor` interface needed by `ClientRunner`. It is not the user's application
executor. `SlurmTaskWorkerLauncher` accepts only the task-worker module and
supports one node per task. It is the bounded exception to the general Slurm
launcher's non-nesting rule: when the CPU CJ is itself Slurm-launched, the
task-worker launcher creates a scheduler manager for the distinct nested task
allocations.

The worker environment removes `AUTH_TOKEN`, `TOKEN_SIGNATURE`, and `SSID`, and
the task-local engine returns no Cell. Its FL context does not contain the
federation SSID. A random CJ-incarnation identifier binds the durable handoff
without disclosing that session identifier.

This is process isolation, not a security sandbox. With `sandbox=none`, the
worker runs as the same operating-system user and can traverse paths that user
can read, including other parts of the site workspace. Preventing a malicious
application from reading deployment credentials requires a separately confined
UID/container and an allowlisted mount view; environment stripping alone does
not provide that guarantee.

## Task state machine

For each task, the resident CJ executes the following sequence:

1. Reject admission if an abort/end-run latch is set or another task is active.
2. Serialize the already-filtered eager input to a new attempt directory.
3. Commit the input manifest after the payload and directory have been synced.
4. Write an immutable worker specification and launch one worker.
5. Wait for the process group or Slurm allocation to settle completely.
6. Require a zero return code, a committed result manifest, matching identity,
   matching SHA-256/size, and a matching completion receipt.
7. Return the result to `ClientRunner`.
8. Let the normal result-filter path complete.
9. Let the normal federation publication path send the result and record whether
   the server acknowledged it.

The event order is therefore:

```text
input committed
  -> worker launched
  -> allocation settled
  -> durable result loaded
  -> result returned to CPU CJ
  -> result filters complete
  -> publication started
  -> server ACK observed
```

Each attempt has a fresh random identity. Input and result manifests bind the CJ
incarnation, job, task, task name, and attempt. The payload is committed first
and the manifest last using exclusive links, `fsync`, size, and SHA-256 checks.
FOBS data must be eager; lazy download references and pass-through payloads are
rejected rather than silently producing an incomplete handoff.

## Application state

No Python object or application session survives from one task worker to the
next. The framework exposes `NVFLARE_TASK_STATE_DIR` and the private FL-context
property `job_task_worker_state_dir`. Applications must explicitly persist the
minimum state needed by a later task.

The maintained hello-pt adaptation uses this directory only for the most recent
local model required by a later `submit_model` task. Returning that latest model
is a common application contract that A, B, and D must all satisfy. Saving it in
this particular file and state directory is the hello-pt-on-B implementation
choice; the requirement itself is neither optional nor B-specific.
`flare.receive()` and `flare.send()` already transfer the current task's model;
the extra file is needed only because a fresh `submit_model` worker cannot see
the prior trainer process's `last_params` Python local. Framework-owned retention
of the previously sent full model could service this workflow instead, but this
prototype does not implement or prove that alternative. Neither approach
preserves arbitrary Python locals or a general application session. When the
environment variable is absent, the adapted script retains its existing
`./cifar_net.pth` behavior.

State mutation and result publication are not transactional. The worker may
commit state and its result, followed by a CJ crash or missing ACK. Re-execution
can then apply the task twice unless the application supplies idempotence. A
production design needs a durable CJ journal and an adoption/replay protocol, or
must make at-least-once semantics explicit.

## Cancellation and failure semantics

| Condition | Prototype behavior | Qualification boundary |
|---|---|---|
| Worker exits nonzero | No result is loaded or published | Unit-tested |
| Result/manifest tampering | Digest or identity validation fails | Unit-tested |
| Task abort/end run | Latch admission, TERM then KILL process group/allocation, wait for settlement | Unit-tested locally |
| Worker leader exits with descendants | Kill descendants and report failure; do not advance | Unit-tested locally |
| Worker timeout | Terminate once, wait for settlement, report failure | Implemented; no real-Slurm fault row yet |
| Slurm pending timeout/nonzero | Inherited manager cancellation/terminal result | Implemented; no real-Slurm fault row yet |
| Missing server ACK | Recorded as not acknowledged, never as success | Unit-tested |
| CJ/site crash after worker commit | Durable files remain but are not adopted | Unsupported |
| Completion/abort race | Abort latch is checked before admission and while polling | Not exhaustively stress-tested |

Local process cancellation uses a new process group and does not declare
settlement until the group disappears. Slurm settlement is delegated to the
existing launcher manager and handle.

## Client API mapping

`ClientAPIJobTaskWorkerExecutor` accepts current `in_process` and
`external_process` configurations, but normalizes both into a single disposable
task-worker contract. The requested source mode is retained in the worker
specification as evidence. `external_process` does not create a resident trainer
and is not the legacy CJ-plus-external-trainer architecture. Attach, reconnect,
and launch-once application sessions are rejected or intentionally not
preserved.

The process-local Client API adapter publishes one global model to the script,
captures exactly one local result, captures log records, and then closes. A
minimal task-local engine supports component lookup, workspace/context creation,
and application lifecycle events. Federation auxiliary messaging fails
explicitly.

## Compatibility assessment

### Refined T2/T4/T7 disclosure

This inventory separates source-configuration compatibility from the runtime
topology that actually executed.

| Row | B evidence and exact boundary |
|---|---|
| T2a Client API `in_process` | Historical P01/P03 rows supplied `task_script_path=client.py`, but the exported executor was `ClientAPIJobTaskWorkerExecutor`, not the standard `ClientAPIExecutor`. The B worker created a process-local `InProcessClientAPI` and ran the script with `runpy`; each application worker still ended after one task (`executor.py:497-610`, `worker.py:150-210`). |
| T2b Client API `external_process`, `launch_once=false` | Historical P02/P04 and the current export preserve `execution_mode=external_process`, the source command, and `launch_once=false`. The adapter extracts only the Python script and arguments from that command and normalizes them to the same process-local worker path used by T2a. Therefore these are real source-config mapping rows, not an external backend/trainer/session pass (`executor.py:571-609`; `job.py:171-225`). |
| T2c Attach | Unsupported by the B disposable task-worker adapter and rejected explicitly (`executor.py:580-581`; `job_task_worker_test.py:536-538`). No Attach runtime evidence exists. |
| T2d external `launch_once=true` | No pass. A job-long trainer contradicts the one-worker-per-task contract. The adapter accepts and records the constructor field but does not preserve a resident trainer; the B recipe pins `launch_once=false`. Treat this as unsupported semantics, not as covered by T2b (`executor.py:500-503,541-545`; `job.py:171-184`). |
| T4 job workspace/state continuity across tasks | `JobTaskWorkerExecutor` creates `.job_task_worker_b/state/{state_id}` once under the job run directory and passes the same path to each fresh worker (`executor.py:359-365,381-401`). As an application-level implementation choice, the adapted hello-pt client writes its latest local weights there after training and a later fresh `submit_model` worker loads them (`client.py:37-50,122-147,188-191`). The historical ledgers show fresh PIDs across the per-site sequence `train, train, train, submit_model, validate, validate, validate, validate`. This proves continuity of that declared model file within one job workspace; it is not an inherent B checkpoint requirement, CJ restart/rejoin, active-attempt adoption, transactional restoration, arbitrary Python-local preservation, or generic application-state compatibility. Framework retention of a previously sent full model is a possible alternative, not implemented evidence. |
| T7 exact legacy evidence | The only completed legacy row is a **local two-client, one-round synthetic hello-pt simulation** using the standard `nvflare.app_common.executors.client_api_executor.ClientAPIExecutor` in `in_process` mode, with tasks `train`, `submit_model`, and `validate`; it reused the resident trainer session and shut down normally. Its export contains no B launcher. The portable successor changes only the task script path to deployed `client.py` and passed a packaged-resolution check. The real deployed legacy row remains pending. No legacy external trainer, Attach, `launch_once=true`, XGBoost variant, Flower/TIE, or CCWF/swarm row was run. Those legacy families are untested here, not all rejected by this single hello-pt result. |

The application delta is narrow but nonzero. Against base
`9759e2594dc6949ef80cedf739c244334fae5f70`, tracked hello-pt `client.py` was
changed to accept a configurable data root, persist/load the latest local model
through `NVFLARE_TASK_STATE_DIR`, and optionally create the isolated G05 payload.
`model.py` and the original hello-pt `job.py` are unchanged. There is no separate
trainer file: `client.py` is the Client API training program. B adds a separate
`examples/advanced/job-task-worker-b/job.py` rather than rewriting the original
recipe. That builder makes cross-site evaluation part of the normal candidate
and selects three rounds, producing eight logical tasks per site (16 total):
three train, one submit-model, and four validation tasks. Failure and G05
fixtures intentionally use smaller isolated schedules and must not be confused
with that normal row (`job.py:82-139,226-227,238-275`).

The generated normal client JSON replaces the standard resident
`ClientAPIExecutor` with `ClientAPIJobTaskWorkerExecutor` and adds
`SlurmTaskWorkerLauncher`; server JSON still names the standard
`ScatterAndGather`, `CrossSiteModelEval`, persistor, aggregator, shareable
generator, model selector, locator, and validation writer. The B adapter owns
artifact handoff and fresh-worker execution; the existing resident
`ClientRunner` still owns task filters, result filters, publication retry, and
ACK. The only change to that existing runner exposes its actual send boolean as
`PUBLICATION_ACK_PROP` for lifecycle evidence (`client_runner.py:581-587`).

Launcher settings are topology/resource settings only: workspace, sandbox/image
and interpreter, Slurm executable paths/directives, one-node worker resources,
time/poll/pending limits, and task environment (`job.py:58-79,263-274`;
`slurm.py:31-37,108-174`). They size and launch the disposable application
worker. They do not create an external Client API backend, change
`launch_once`, or decide the resident CPU CJ's placement. The site's ordinary
job launcher controls whether that CJ runs locally or in its own Slurm
allocation.

`SUPPORTED` below means exercised by this bounded adapter, not that every
configuration of the example is supported. `UNQUALIFIED` means dependencies or
data were not available in the frozen environment; it must not be read as a
product rejection.

| Hello-world workload | Classification | Reason |
|---|---|---|
| hello-pt | SUPPORTED | Synthetic in-process/external local and real Slurm; explicit model checkpoint for later submit-model |
| hello-numpy | SUPPORTED | Both source modes, two clients, three rounds locally |
| hello-log-streaming | SUPPORTED | Both source modes and server-side streamed logs locally |
| hello-collab | UNSUPPORTED | Requires a resident collaboration/session model |
| hello-flower | UNSUPPORTED | Flower owns a resident client session |
| hello-cyclic | UNQUALIFIED | Client API shape is adaptable; dependency/runtime row not completed |
| hello-dp | UNQUALIFIED | Client API shape is adaptable; Opacus/data row not completed |
| hello-huggingface | UNQUALIFIED | Client API shape is adaptable; model/data/dependency row not completed |
| hello-jax | UNQUALIFIED | Script shape is adaptable; JAX/data row not completed |
| hello-lightning / lightning-eval | UNQUALIFIED | Client API shape is adaptable; dependency row not completed |
| hello-lr | UNQUALIFIED | Requires its dependency/data setup and a recipe adapter |
| hello-numpy-cross-val | UNQUALIFIED | Needs a B recipe adapter and a maintained-data row |
| hello-tabular-stats | UNSUPPORTED | Uses a broader engine-dependent executor surface not present in the task-local registry |
| hello-tf | UNQUALIFIED | Client API shape is adaptable; TensorFlow row not completed |

Distributed/torchrun jobs, multi-node task workers, peer-to-peer workflows,
CCWF/swarm controllers, TIE, and executors that depend on live auxiliary
messaging remain unsupported by this prototype.

## Qualification record

Base source: `9759e2594dc6949ef80cedf739c244334fae5f70` in an isolated, uncommitted
checkout. The saved `main` checkout was not modified.

Local verification completed:

- A broader B/client-runner/Slurm-launcher/manager set passed 250 tests.
- Scoped black, isort, and flake8 checks pass for the current Python changes.
- The full repository Python style gate (`./runtest.sh --skip-install -s`)
  passed earlier for the historical Slurm candidate: black, isort, flake8, and
  agent-skill lint. The first invocation tried
  to install already-present dependencies and was blocked by the sandboxed uv
  cache; the documented skip-install form exercised the actual checks.
- hello-pt synthetic in-process and external-process simulations each completed
  two clients, three train rounds, validation, and submit-model behavior. Each
  produced 16 task attempts, 16 distinct worker PIDs, and 16 acknowledged
  publications.
- hello-numpy and hello-log-streaming each completed in both source modes with
  two clients and three rounds. Log streaming reached the server-side store.
- A delayed-publication row showed the worker settled about one second before
  the intentionally delayed CPU-CJ publication path.
- Targeted exception and exit-code-7 simulations produced no committed result
  artifact. The CPU CJ published only framework diagnostic error replies, the
  server rejected them as `EXECUTION_RESULT_ERROR`, and both jobs aborted in
  round zero rather than reporting success.
- A separately configured standard `ClientAPIExecutor` resident simulation
  completed one round, accepted both client contributions, reused the same
  trainer session for submit-model/validation tasks, and shut down normally.
- The isolated G05 fixture locally committed and transported a 4 MiB byte
  payload, verified its digest at the server, accepted the model contribution,
  and recorded a positive ACK. This validates the fixture but is not credited
  as the real slow-network observation.
- The corrected real G05 row completed successfully with a verified 64 MiB
  payload and a 44.421742-second `SubmitUpdate`. Slurm allocation 676 settled
  successfully before CPU-CJ publication began, and the server acknowledged the
  contribution. The passive probe crashed on a fixture schema assumption before
  collecting in-transfer `nvidia-smi` samples. A separately identified
  observer-only supplement preserved the product/runtime/export identity and
  sampled exactly once during a 44.503580-second publication window: the
  completed worker was absent, Slurm allocation 677 was absent, and the GPU
  compute-process list was empty. Its positive ACK and immediate/+30-second
  cleanup audits also passed; the authoritative product row remains unchanged.

The initial real-Slurm setup attempt failed before allocation because the copied
site workspace granted group/world permission; the inherited launcher correctly
rejected it. The setup defect was preserved, both B-owned site workspaces were
tightened, and the job was resubmitted once as allowed by the comparison
contract.

The second node's transient user service did not survive its SSH setup session,
so its persistent client parent was run through a held SSH session for the
campaign. During final cleanup, the site stop script left that client and SSH
session alive; terminating the exact two B-owned PIDs completed cleanup. This is
a harness/service-management defect, not a task-worker settlement failure. The
final controller and worker audits found no B federation process, task worker,
Slurm job, or CUDA process.

### Bounded G01--G09 evidence map

The active acceptance contract is the shared bounded G01--G09 campaign contract.
The exact common 16-task schedule is a future matched-performance prerequisite,
not a functional-feasibility requirement. Expanded restart/rejoin, lost-ACK,
upload-recovery, cancellation-race, and artifact-fault rows are retained in the
separate production-hardening roadmap and are not counted as prototype gaps.

| Gate | Status | Evidence and remaining bounded gap |
|---|---|---|
| G01 real federation | PASS (historical candidate) | P01-matched and both maintained-CIFAR rows ran one server and two clients for three train rounds. Their ledgers record the expected six accepted train contributions, later submit/validation tasks, positive ACKs, and successful terminal status. |
| G02 two source configs | READY; predecessor Slurm evidence | P01/P03 used the in-process source configuration and P02/P04 used the external-process configuration. Both map to the same fresh task worker, not a resident external trainer. The current local recipe sets `launch_once=false` explicitly for both modes; the historical in-process export relied on the adapter's disposable translation and must not be described as an exact-current-config run. |
| G03 maintained CUDA workload | PASS (historical candidate) | P03 and P04 used maintained CIFAR-10 with one A100 requested per task. Concurrent client Python CUDA processes were observed at 536 MiB on both sites, followed by clean GPU/allocation audits. |
| G04 explicit domain state | PASS (historical candidate) | Every task used a fresh worker PID. The adapted hello-pt client persisted only its most recent local model in the framework-provided state directory; later `submit_model` workers restored it. Generic input/result manifests and task phases remain framework-owned. |
| G05 settle before CPU publication | PASS | The first isolated Colossus product row (`12b0afa9-81b6-45e9-ab79-7f328baef098`, Slurm 669) exposed a job-local import bootstrap defect before training; the worker fix has an unmasked regression. The corrected authoritative row (`a67a9f93-183c-4ce9-8224-9f4d452341a3`, Slurm 676) completed, verified the exact 67,108,864-byte payload and digest, settled the allocation before publication, spent 44.421742 seconds in actual `SubmitUpdate`, and received a positive ACK. Its original probe defect is preserved. The separately sealed observer supplement (`13902bd1-9cb0-4984-ab6c-d61141335827`, Slurm 677) used the same product/runtime/export/training/stream identity and sampled exactly once inside a 44.503580-second upload: worker absent, `squeue` empty, and GPU compute PID list empty. ACK and immediate/+30-second cleanup passed. The supplement closes only the bounded observation gap and does not replace the authoritative product row. |
| G06 failed application cannot succeed | PENDING DEPLOYED E2E | Unit tests and two-client simulations are development evidence for a raised application exception and abrupt exit code 7. Neither leaves a committed result. The CPU CJ may publish a framework diagnostic error reply, as the gate permits; the server rejects it as `EXECUTION_RESULT_ERROR` and aborts instead of accepting a model or reporting job success. The two exported bounded real-Slurm fault variants remain required for terminal failure and cleanup evidence. |
| G07 legacy resident path | PENDING DEPLOYED E2E | A separate standard `ClientAPIExecutor` in-process resident simulation completed one round, accepted both contributions, reused the resident trainer for submit-model/validation, and shut down normally. That is development evidence only and does not use the B adapter. Preserve one bounded deployed regression separately from B compatibility. |
| G08 clean shutdown | PARTIAL; TARGETED GAP | Immediate P03/P04 worker, Slurm, and CUDA audits were clean. Campaign cleanup later found one harness-owned persistent client/SSH pair; terminating those exact B-owned PIDs produced a clean node audit. A bounded normal/fault rerun should preserve an unassisted terminal cleanup audit rather than relabeling the manual harness cleanup as a pass. |
| G09 limits and migration | PASS (design evidence) | The compatibility table, ownership/topology table, failure boundary, unsupported features, and comparison/operational-cost sections below state migration work and remaining limitations without claiming production hardening. |

Only bounded remote confirmation remains: both exported G06 failure variants
plus their G08 cleanup audits, and the
exported G07 resident success plus its G08 cleanup audit. No other expanded
campaign row is required for the prototype gate.

Real two-node Slurm rows completed on 2026-09-16:

| Row | NVFlare run ID | Mode | Workload size | Terminal | Duration | Task allocations |
|---|---|---|---:|---|---:|---:|
| P01-smoke | `9ac571bc-82a5-4a96-aca2-edc80b29c8ed` | in-process | synthetic 32/16, one epoch | `FINISHED:COMPLETED` | 1m30s | 16 (Slurm 570--585) |
| P02-smoke | `2c4da223-9ab7-477a-aae5-f3c4dff46bf9` | external-process configuration mapping | synthetic 32/16, one epoch | `FINISHED:COMPLETED` | 1m33s | 16 (Slurm 586--601) |
| P01-matched | `bff32f81-caa7-4f91-bd6f-981ec7b91aab` | in-process | synthetic 50k/10k, two epochs | `FINISHED:COMPLETED` | 8m04s | 16 (Slurm 602--617) |
| P03-matched | `f412e703-88fa-4199-b25b-836107db506c` | in-process | maintained CIFAR-10, two epochs, one GPU | `FINISHED:COMPLETED` | 3m38s | 16 (Slurm 618--633) |
| P04-matched | `999e9dc1-e7c5-46b0-8ed7-c91be4d7be3a` | external-process configuration mapping | maintained CIFAR-10, two epochs, one GPU | `FINISHED:COMPLETED` | 3m58s | 16 (Slurm 644--659) |
| T01-like | `158a6c26-3e8f-4573-8c99-7fac5b88c825` | in-process | synthetic 32/16, one round, 5s result filter | `FINISHED:COMPLETED` | 1m24s | 10 (Slurm 634--643) |
| G05 slow upload | `a67a9f93-183c-4ce9-8224-9f4d452341a3` | in-process | synthetic 32/16, one round, 64 MiB result payload | `FINISHED:COMPLETED` | 59.91s | 1 (Slurm 676) |
| G05 observer supplement | `13902bd1-9cb0-4984-ab6c-d61141335827` | in-process | same frozen G05 product; one corrected observer sample | `FINISHED:COMPLETED` | 65.91s | 1 (Slurm 677) |

Each smoke row had two resident CPU CJs outside Slurm, 16 distinct worker PIDs,
16 distinct one-node/two-CPU/no-GPU Slurm jobs, and 16 positive server ACKs. All
32 Slurm jobs completed. Every task ledger ordered allocation settlement before
CPU-CJ result filters, publication, and ACK. No sampled task worker remained
after its allocation. These are topology/lifecycle results only; their reduced
workload is not used for performance comparison.

The frozen A/D comparison workload uses 50,000 synthetic training samples,
10,000 test samples, batch size 16, two local epochs, and two data-loader workers.
The B recipe defaults were corrected to those values after the smoke rows. The
matched synthetic row used two CPUs and 8 GiB per task; its 8m04s wall time is
resource-qualified because the frozen D summary does not preserve an identical
per-phase CPU allocation. Its 16 ledger sequences still had distinct worker
PIDs and Slurm IDs, positive ACKs, and the required B event order.

The matched CIFAR-10 row used the frozen maintained cache and requested one A100
per task. Both sites were observed concurrently with live Python CUDA processes
using 536 MiB each. Slurm accounting reported 16 completed one-node, two-CPU,
one-GPU jobs. The event ledgers contained 16 distinct worker PIDs and allocation
IDs, 16 positive ACKs, and ordered settlement-before-publication sequences.
Immediate post-run checks on both nodes found no worker PID, CUDA process, or
Slurm allocation. Its 3m38s duration is recorded as B evidence only. It is not a
performance-equivalence claim against D because the reports do not establish
identical logical task counts, phase structure, or allocation resources.

The external-process mapping repeated the maintained CIFAR-10 row in 3m58s.
Both sites were again observed on CUDA, and all 16 worker/allocation/ACK
sequences and cleanup checks passed. Together the six completed real-Slurm rows
created 90 fresh task allocations; all of their recorded Slurm jobs completed.
The exact full-size synthetic P02 row was exported but not submitted because the
smoke P02 plus matched external GPU row already exercised the mode mapping, and
the two-CPU matched synthetic row had shown an eight-minute runtime.

The real-Slurm delayed-publication row produced ten distinct worker PIDs and
Slurm allocations and ten positive ACKs. For every task, allocation settlement
preceded completion of the intentionally slow CPU-CJ result filter by at least
5.001s (median 5.006s), followed by publication and ACK. This directly proves
B's early allocation-release boundary. It is `T01-like`, not the frozen
contract's exact slow-network-upload T01: the measured delay is a CPU result
filter immediately before transport. It proves GPU/task-allocation release
before potentially long CPU-side publication, but does not emulate slow network
transport.

### Provenance and evidence index

The public candidate is based on
`9759e2594dc6949ef80cedf739c244334fae5f70`; the branch commit is the canonical
source identity. The example can reproduce in-process, external-process,
application-exception, nonzero-exit, and legacy-resident exports. B exports name
`ClientAPIJobTaskWorkerExecutor` plus `SlurmTaskWorkerLauncher`; the resident
export names the standard `ClientAPIExecutor` and no B launcher. All B
source-mode exports explicitly record `launch_once=false`.

Local campaign archives, deployment roots, and machine-specific evidence paths
are intentionally not published. In any deployment, the authoritative per-site
ledger is `{site_workspace}/{run_id}/.job_task_worker_b/events.jsonl`. Each
attempt's `input.fobs`, `input.json`, `worker.json`, `result.fobs`,
`result.json`, and `worker_receipt.json` are under the sibling
`attempts/{attempt}/` directory.

P03 and P04 task-worker, Slurm, and GPU checks were clean in the immediate
post-terminal audits. At campaign cleanup, the first delayed audit found a
persistent site-1 harness parent that the stop script had missed; terminating
the exact B-owned client and dedicated SSH-session PIDs fixed it, and a five-
second re-audit found no B federation, task-worker, Slurm, or GPU residue. This
must not be reported as a clean +30-second harness audit. The pre-allocation
workspace-permission setup failure used run ID
`53a10f65-11c2-4cf0-bd4c-445e82d72c74`; it is preserved as a failed setup
attempt and is not one of the six successful rows.

## Comparison with Architecture A and phased D

| Dimension | A: deployment supervisor | B: resident CPU CJ + task worker | Phased D: framework task scope |
|---|---|---|---|
| Federation owner | Deployment-owned supervisor plus a new authenticated server task service | Existing job-lifetime CJ and `ClientRunner` | Framework-owned task/phase launcher; a fresh CJ owns each phase |
| Application lifetime | Fresh worker under the supervisor | Fresh worker per task | Fresh compute CJ; separate fresh pull and push CJs |
| Persistent site cost | Deployment supervisor/CP process | One full CPU CJ per active job | Phase coordinator; no job-long application CJ |
| GPU release point | After worker settlement, before supervisor publication | After a site-durable worker result, before normal CJ result filters/publication | After compute settlement, before a separate CPU-only push CJ |
| Filter and ACK path | Configured filters are translated into the worker; the new server service accepts publication | Existing `ClientRunner` owns both filter chains, retry and ACK | Filters execute in compute; push owns normal send/ACK and explicit publication components |
| Durable handoff | Supervisor artifacts | Manifest-last FOBS plus receipt in the CJ workspace | Input/result artifacts plus a receipt for each pull/compute/push phase |
| Parent crash adoption | Not established | Not implemented | Explicitly not provided by the frozen launcher |
| Compatibility shape | Direct-script mapping for Client API modes; bounded server communicator support | Direct-script mapping plus an explicit ordinary-Executor opt-out path | Requires task-scope-capable executors and eager synchronous work |

This table is a topology comparison, not a ranking by pass count, source size,
or unmatched wall time. The exact common CSE v1.2 evidence says B passes the
16-task train/submit/validate contract, while A v8 fails it because its
translation omitted default Client API task names. A v9 restores those defaults
and is a new candidate. D completed all 16 application tasks, eight validations,
and 48 Slurm phases, but its authoritative terminal status is
`FINISHED:EXECUTION_EXCEPTION`: server finalization overtook the last push's
settlement and CP receipt consumption after the server had accepted the result.
The detailed B, A, and D run reports remain in the private campaign archive;
none of those facts settles the final architecture decision by itself.

Durable result ownership and compute-allocation lifetime are separate concerns.
The user-required boundary is to release the GPU after compute settles and
before a potentially long CPU-side upload. Both B and phased D do that: B hands
a site-durable result to the resident CPU CJ, while D moves from its compute
phase to a separate CPU-only push phase. Server acceptance occurs later in both
descriptions. B's narrower weakness is that this prototype does not adopt its
durable result after a CJ/site crash; it is not a failure to retain the GPU
through server acceptance.

## Source-cited decision assessment

### What B reuses and what it adds

**Source-backed ownership.** B inserts one infrastructure `Executor` under the
ordinary client task loop. `JobTaskWorkerExecutor` explicitly receives an
already-filtered `Shareable`, waits for the worker/allocation to settle, validates
the committed result, and returns it to the normal runner; it does not own
publication (`nvflare/private/fed/job_task_worker/executor.py:154-162,337-494`).
The existing `ClientRunner` path still performs input filters before invoking that
executor and result filters after it returns
(`nvflare/private/fed/client/client_runner.py:317-387,448-519`). It also retains
the task-check/retry/send loop and records the actual send boolean used by B's
ACK evidence (`client_runner.py:574-639`).

The resident CJ therefore continues to own CellNet, federation credentials,
task acquisition, cancellation and ACK semantics. B removes the federation
token, signature and SSID from the application environment
(`executor.py:301-310`; `nvflare/private/fed/job_task_worker/slurm.py:151-183`),
and the task-local engine returns no Cell and rejects auxiliary messaging
(`nvflare/private/fed/job_task_worker/worker.py:73-106`). Credentials are still
available to the resident CJ through its normal engine; they are not recreated
inside B.

B adds the following lifecycle below that boundary:

- an attempt directory and manifest-last eager FOBS handoff bound to the CJ
  incarnation, job, task and attempt; payload size and SHA-256 are revalidated
  on read (`nvflare/private/fed/job_task_worker/artifacts.py:53-65,85-150`);
- one fresh script or ordinary-Executor process per task, with task-local
  `START_RUN`/`END_RUN` for the ordinary-Executor form and a process-local Client
  API for scripts (`worker.py:132-210,227-293`);
- process-group or Slurm launch, timeout/abort termination, full settlement,
  receipt validation and only then return to the CJ
  (`executor.py:333-494`; `slurm.py:31-37,98-184`); and
- a stable job/site state directory. The hello-pt model file is one application
  implementation of the common later-`submit_model` contract, not a B-specific
  semantic requirement (`executor.py:359-401`).

**Failure scope.** A worker exception or nonzero exit prevents B from loading a
result and becomes an execution-error reply through the still-running CJ. It
does not crash or replace that CJ (`executor.py:451-479`;
`client_runner.py:382-444`). Conversely, a CJ/site failure after the worker has
committed and settled but before server ACK strands the result: `_pending_publication`
is memory-only, and no restart path scans/adopts the attempt directory
(`executor.py:217-218,268-290,480-494`). Application state may already have
changed, so blind re-execution can duplicate a state transition. The fundamental
server-acceptance/client-knowledge gap is common to A, B, and D. B's specific
liability is its memory-only CJ ownership and lack of durable adoption or
reconciliation for the otherwise retained attempt.

**Compatibility and fallback.** B preserves `in_process` and
`external_process` as source-configuration evidence but maps both to the same
single-task, process-local Client API worker. Attach and a resident
`launch_once=true` trainer are not preserved
(`executor.py:497-610`). There is no resident B worker pool: only the CPU CJ is
resident. The example's `--legacy-resident` option is an explicit opt-out that
constructs the original standard `ClientAPIExecutor` and forbids combining that
baseline with the B Slurm path; it is not an automatic fallback for an
unsupported B mode (`examples/advanced/job-task-worker-b/job.py:204-234,295-296`).

### Contrast with A and phased D

**Source-backed design consequence.** B's strongest concrete advantage over
both alternatives is preservation of the existing federation boundary: normal
ClientRunner filters, Cell/session ownership, retry and ACK remain authoritative.
A instead translates executor configuration and filters into worker routes
(`A v9 review candidate, nvflare/private/fed/deployment_supervisor/config.py:52-86,115-145`)
and adds a new authenticated server/CP task-acquire and publish service
(`A v9, deployment_supervisor/federation.py:60-97,136-196,199-280`). D keeps
standard CJ machinery but reconstructs it across pull, compute and push: filters
run in compute, while send events and ACK run later in push
(`D commit 3dd3b42d59006f810aa3454ad473ec68f6fa5cc7,
nvflare/private/fed/task_scope/runner.py:46-64,140-157,192-239`). B therefore
adds the least new federation semantics for an eager ordinary task; its new
protocol is local CJ-to-worker handoff.

That advantage is conditional. A can remove the full job-lifetime CJ from the
application path, while D gives transfer and publication their own framework
phases. D's launcher explicitly sequences pull/compute/push receipts and retains
phase liveness/cancellation until settlement
(`D task_scope/launcher.py:83-89,174-287,289-370`). B instead pays for a full
resident CJ and makes correctness depend on both that CJ and the disposable
worker. Calling B categorically "the most complicated" is not supported: B has
fewer federation/phase adapters and more direct reuse, A has a new server task
service plus route/filter translation, and D has three physical phase
lifecycles. B may still be the most operationally awkward at high job counts
because it combines a standing CJ with per-task scheduler churn. The current
evidence does not measure maintenance or operator effort well enough to rank
those different kinds of complexity.

### Final-result ownership and the D finalization race

The exact D failure sharpens, but does not by itself settle, the B comparison.
In B the disposable worker/allocation ends before result filters or publication.
The same resident CJ that received the task then owns the returned `Shareable`
through synchronous `ClientRunner._send_task_result`: it checks that the server
still owns the task, retries transport, and fires `AFTER_SEND_TASK_RESULT` only
after send success, task-gone, or abort (`client_runner.py:574-639`). There is no
separate post-ACK physical push handle whose scheduler settlement must be
converted into another CP receipt. Consequently, the particular D ordering--SJ
accepts, push writes `TASK_COMPLETE`, SJ exits, then CP cancels before consuming
that receipt--is not present in B's topology.

B still has a related unresolved ambiguity, not an atomic acceptance protocol.
If the server accepts a result but the reply is lost, a later task check can say
the task is gone, or normal/abnormal teardown can trigger the run abort signal.
`ClientRunner` then returns `False`; B records
`cpu_cj_publication_not_acknowledged`, removes only its in-memory pending entry,
and leaves the durable attempt without adoption or reconciliation
(`client_runner.py:592-639`; `executor.py:282-290,480-494`). Reusing
`ClientRunner` therefore removes D's extra push-to-parent receipt handoff, but
does not remove the common server-acceptance/client-knowledge gap.

The existing B CSE run proves the normal success ordering only. For its final
result, the server logged receipt at `23:59:32.601`, site 1 recorded positive
publication ACK at `23:59:32.608`, and the server began finalization at
`23:59:32.718`; the resident CJ did not enter `END_RUN` until
`23:59:34.615`. All 16 B attempts followed the same settled-worker, loaded
result, positive-ACK pattern and the job ended `FINISHED:COMPLETED`. This is
direct evidence that one stable publisher survived compute exit and completed
before finalization on that run. It is not evidence for ACK loss, CJ failure in
the commit-to-ACK window, or restart/adoption. Those are common A/B/D roadmap
risks and optional future decision evidence, not required rows in the current
campaign. Nor does one D implementation race prove B is categorically superior.

### Measurements and operational burden

These are observations from the recorded B runs, not general capacity claims:

- the two job-lifetime CJs in a maintained CIFAR-10 row used about 117 and
  118 MiB RSS and roughly 2% CPU each, excluding the deployment-level client
  parent;
- small synthetic tasks paid median launch-to-settlement costs of about 5.95s
  for in-process source mapping and 7.31s for external-process mapping, while
  median worker peak RSS was about 726 MiB; and
- the exact 64 MiB slow-upload row proved that the task allocation and GPU were
  gone during the later CPU-CJ publication window.

Operationally, every active job needs a persistent CPU CJ, either on the host
or in a job-long Slurm allocation, plus a shared durable workspace visible to
the task allocation, correct owner-only workspace permissions, the Slurm
command set/configuration, and cleanup of both the job-long CJ and short
allocations (`slurm.py:31-38,98-145`). The campaign
also exposed a harness burden: one client parent needed a held SSH session and
manual exact-PID cleanup. That incident is not a B product leak, but it shows
that the standing-CJ topology makes service supervision part of deployment
quality rather than just test plumbing.

### Recommendation and evidence that could change it

**Current recommendation (judgment):** keep B as an experimental bounded
adapter, not the default architecture. Its strongest advantage is semantic
conservation at the federation boundary while releasing the GPU before upload.
Its strongest B-specific liability is memory-only pending-publication ownership
in one resident CJ, with no adoption/reconciliation after that CJ fails, combined
with one such CJ per active job. The underlying server-acceptance/client-knowledge
ambiguity is common to A, B, and D. Short-task scheduler/import cost and
unsupported resident/Attach/lazy/auxiliary application behavior are important
secondary limits.

Optional future evidence that could move B toward selection is a restart-adoption
row proving a committed result is recovered without a duplicate application
state transition. ACK loss, restart/adoption, multi-node/DDP, and broad legacy
families are deferred equally as common A/B/D roadmap work, not current
acceptance gates. Current bounded evidence still includes deployed
exception/nonzero cleanup and the separate original-Executor regression.
Evidence that would move B away is an A or D matched common/fault/cleanup
campaign that preserves the required Controller/filter behavior with lower
measured standing-site cost, or a site-scale test showing the resident-CJ cost
is unacceptable. A decision also needs
an explicit product choice for lazy/pass-through data, auxiliary messaging and
resident application protocols. Those are proposals/decision inputs, not claims
that the current prototype already supplies them.
