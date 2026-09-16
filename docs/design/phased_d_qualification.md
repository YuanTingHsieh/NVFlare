# Phased D qualification

Status: in progress. This document evaluates the phased-CJ prototype against
the common Architecture A/B/D qualification contract. It is an evidence ledger,
not a production-readiness claim.

## Candidate provenance

| Field | Value |
|---|---|
| Branch | `feat/slurm-task-scoped-cj-prototype` |
| Candidate commit | `3dd3b42d59006f810aa3454ad473ec68f6fa5cc7` (not frozen) |
| Candidate tree | `11e419b142d06403a429c4b70135fdf9cdecd657` |
| Merge base | `9759e2594dc6949ef80cedf739c244334fae5f70` |
| Signature | Signed with the configured developer key |
| Remote state at inventory time | Branch matched `origin/feat/slurm-task-scoped-cj-prototype` |

The comparison workload is not yet frozen. Earlier phased-D runs used a
different logical task count from Architecture B, so none of them qualifies a
common E2E row by default. A final manifest must pin source, exported configs,
data digest, seeds, task sequence, resources, timeouts, fault triggers and
assertions before comparative runs begin.

## Runtime boundary being qualified

For each eligible ordinary client task, the CP owns one logical job handle and
launches three sequential CJs:

```text
CPU pull CJ -> committed input -> GPU compute CJ -> committed result
             -> compute allocation fully settles -> CPU push CJ -> server ACK
```

All three phases use the normal client `worker_process` entrypoint. Pull and
push construct a framework-only transfer graph; compute constructs the ordinary
application graph. The Executor does not select phases. The same logical
`job_id` and existing job Cell FQCN are reused, so only one CJ for a given
`(site, job)` may exist at a time.

## Common E2E matrix

Only retained evidence from the exact candidate and frozen common manifest may
change a row to `PASS`. Unit tests establish local mechanics but do not qualify
real federation, Slurm, GPU, transport or cleanup behavior.

| ID | Status | Current implementation/evidence | Work required for qualification |
|---|---|---|---|
| Q01 | NOT RUN | Ordinary eager tasks use the standard task pull and result ACK paths. | Run the frozen two-client hello-numpy in-process source config and retain task/acceptance evidence. |
| Q02 | NOT RUN | `ClientAPIExecutor(external_process, launch_once=False)` is accepted for task-scoped compute. | Run the frozen external-process config; retain exported config and process trees proving the trainer mapping. |
| Q03 | NOT RUN | Each phase builds the framework `JobLogStreamer`; no new server barrier waits for every stream EOF before workspace archival. | Run both source configs and audit every expected client log/analytic. A failure in final-log archival belongs to the shared live-log/server-finalization subsystem unless phased ordering introduces a distinct defect. |
| Q04 | NOT RUN | Input/result artifacts preserve task identity and session; application checkpointing is demonstrated only by the counter example and process tests. | Run the frozen synthetic train/validate/submit-model sequence with in-process Client API and assert restored model/domain state. |
| Q05 | NOT RUN | Same as Q04; external `launch_once=False` reconstructs its trainer for every compute CJ. | Run the identical synthetic workload through the frozen external-process source config. |
| Q06 | NOT RUN | Pull/push remove GPU requests; compute retains the declared Slurm resource request. | Run maintained CIFAR-10 CUDA on both clients and retain `sacct` plus GPU samples for the full task sequence. |
| Q07 | NOT RUN | The supported external topology is a fresh trainer owned by each fresh compute CJ. | Run the same maintained CUDA workload/config as Q06 with external process and document the extra trainer process. |
| Q08 | NOT RUN | Fresh-process checkpoint restoration and task events have focused unit/process coverage. The current example does not prove arbitrary component lookup. | Use the common ordinary Executor fixture with a declared component dependency, task-event assertions and durable state restored in a later worker. |
| Q09 | NOT RUN | The server readiness probe supports exact built-in `WFCommServer` broadcast/send managers. Client data/result filters run in compute; server filters remain on receipt; push uses the existing send/ACK path. No new publication-hook API is required or provided. | Run blocking/nonblocking send and broadcast with selected filters and framework instrumentation distinguishing send completion from actual ACK. Inventory arbitrary application send-event handlers separately. |
| Q10 | NOT RUN | The compute allocation must settle before push submission. Existing tests simulate phase ordering, not delayed network transport. | Delay transfer of committed model bytes at an isolated endpoint and retain manifest-commit, allocation-terminal, upload and ACK timestamps plus GPU samples. |
| Q11 | NOT RUN | `task_phased=false` or omission follows the original launcher path; a focused unit test verifies immediate resident launch. | Run the frozen maintained application through the unchanged resident-CJ configuration and record it separately from new-mode compatibility. |

## Common fault and restart matrix

| ID | Status | Current implementation/evidence | Work required for qualification |
|---|---|---|---|
| F01 | NOT RUN | A nonzero physical return code or missing valid receipt stops the pipeline; crash and receipt behavior have focused tests. | Inject application exception and hard nonzero exit in real allocations; prove bounded failure, no publication/retry and full cleanup. |
| F02 | NOT RUN | The logical handle waits for the physical launcher handle to settle before reading a receipt or starting the next phase. | Kill compute and separately leave a descendant alive; use Slurm/process evidence to prove the entire owned allocation tree settles first. |
| F03 | NOT RUN | Cancellation closes admission and terminates the active physical handle; race cases have focused tests. | Exercise idle, pending, compute and push cancellation in real Slurm and audit late task/result rejection and cleanup. |
| F04 | NOT RUN | FOBS artifacts are write-once, manifest-last and bound to job, attempt, task and session; malformed/stale/corrupt artifacts have focused tests. | Corrupt real attempt artifacts at each required point and retain bounded terminal and cleanup evidence. |
| F05 | NOT RUN | A committed result survives push failure. Existing send retries are ultimately bounded by transfer supervision; no training retry is implemented. | Interrupt and restore actual transport using the common timeout policy. Verify either successful bounded publication or explicit failure, unchanged artifact hash and no new compute allocation. |
| F06 | NOT RUN | `TASK_COMPLETE` is written only after the existing ACK. If the server accepted the result but the ACK is lost, a later task check is expected to make push fail rather than fabricate success; there is no reconciliation protocol. | Inject ACK loss after acceptance; prove one server contribution, no retraining, preserved artifact and an explicit ambiguous/failed client outcome agreed by all candidates. |
| F07 | NOT RUN | Unit tests cover handle settlement and cleanup ordering, not real residue. | Audit Slurm, process trees, GPU processes, listeners and job-owned files immediately, +30s and fault +75s; list standing CP services separately. |
| R01 | UNSUPPORTED | CP restart loses `ClientExecutor.run_processes`, `TaskScopedJobRegistry` and the logical handle. The server sends `START_JOB` only at initial launch and does not recreate running participation for a newly registered CP. | Define shared rejoin semantics: invalidate old ownership, idempotently reissue/resume an eligible running job, reconstruct the logical launcher handle from durable metadata, and prove one later-round contribution with restored state. Active-attempt adoption is out of scope. |

## Existing semantics that cannot remain unchanged

These behaviors conflict with fresh application execution rather than merely
being missing from the prototype.

| Existing behavior | Lifetime/API conflict | Explicit replacement | Scope |
|---|---|---|---|
| Executor/Learner state kept only in Python objects across tasks | Every compute CJ constructs a fresh application graph. | Declare and restore domain state through task input or durable application checkpoints. | Common to disposable execution |
| External Client API `launch_once=True` | The trainer session belongs to a compute CJ that ends after one task. | Use `launch_once=False`, or keep the legacy resident-CJ mode for a genuinely job-lived trainer. | D topology |
| Attach trainers and other independently resident trainer sessions | Their identity and liveness span multiple task workers. | Use resident mode or design a separately owned session service; do not silently restart it per task. | Common |
| Lazy/pass-through results whose producer must remain alive | Compute and trainer must be gone before CPU publication starts. | Materialize an eager committed result before compute settlement; large-result implementations may stream from durable storage afterward. | Common |
| Async work that outlives `Executor.execute()` | The compute allocation cannot settle while unowned application work still produces the result. | Join/commit within the task boundary, or model the long-lived interaction as an explicitly resident execution class. | Common |
| Aux `DO_TASK` execution or unsolicited messages to an idle CJ | No CJ exists during gaps, and concurrent aux execution defeats the one-attempt boundary. | Express ordinary work as scheduled tasks; retain resident mode for interactive protocols until a declared replacement exists. | D topology |
| Application handlers that require one job-lived object or must observe network send/ACK in the same graph as compute | Compute and push are different CJs; pull/push intentionally do not construct the application graph. | Split compute-local behavior from framework publication observations, persist required data, or use resident mode. | D topology |
| Custom task managers/communicators with side-effecting or unspecified availability checks | CP readiness must not assign, copy/filter or invoke callbacks before the pull CJ requests the task. | Add a pure readiness contract, or reject the manager in task-phased mode. | D topology |
| Concurrent workers for the same `(site, job)` reusing one Cell FQCN | Overlap makes routing and incarnation attribution ambiguous. | Keep sequential admission, assign unique worker identities, or move communication to a persistent owner. | D topology |
| `START_RUN`/`END_RUN` handlers interpreted only as logical job lifetime | A fresh compute application graph observes them per incarnation. | Define task-incarnation lifecycle semantics and move job-lifetime state to a persistent/durable owner. | Common, with D event mapping |

## Possible but unimplemented or unqualified

| Capability | Exact current status | Implementation needed | Tests/evidence needed |
|---|---|---|---|
| Shared CP restart/rejoin at a quiescent boundary | Unsupported; in-memory owner is lost and no start-existing-job command exists. | Common durable participation/ownership record and authenticated idempotent resume command. | R01 on all candidates with duplicate-contribution and state assertions. |
| Active-attempt or pending-publication adoption | Not implemented and not required by R01. | Durable phase state, ownership fencing and reconciliation. | Separate crash-at-each-transition campaign. |
| Publication retry/reconciliation/deduplication | Existing transient send retry only; no durable ACK reconciliation. | Common accepted-result identity/status query if stronger recovery is desired. | F05/F06 plus duplicate injection. |
| General task retry | No automatic compute retry; a failed phase ends the logical client job. | Controller policy, attempt identity/fencing and idempotent state contract. | Retry-safe and retry-unsafe Executor fixtures. |
| Very large committed results | Eager FOBS artifact is fully materialized on shared POSIX storage before push. | Bounded-memory file/object-store streaming with filters placed before or during committed publication. | Multi-GB memory, integrity, restart and slow-upload tests. |
| Multi-node Slurm/DDP | Explicitly rejected when `nodes != 1` or an additional-node command exists. | Preserve phase scoping while launching and settling the complete multi-node compute allocation; define rank ownership and artifact commit. | Two-node/multi-GPU Client API E2E and descendant cleanup faults. |
| Kubernetes and Docker task-phased execution | Not supported because current workspaces are pod/container-local or mounted at a different path. | Artifact transfer/service, persistent shared volume, or explicit path mapping owned by the framework. | Same Q/F rows on each launcher. |
| Non-shared storage | Prototype requires the same durable absolute POSIX path on CP and all Slurm nodes. | Framework-owned artifact store/transfer contract. | Node-local and remote object-store tests with corruption and cleanup faults. |
| Final live-log completeness | No barrier proves every phase's log stream reached EOF before server archival. | Shared live-log/server-readiness finalization fix, unless testing finds a D-specific ordering defect. | Q03 repeated stress runs with expected-file manifest. |
| Relay/custom task managers and arbitrary communicators | Explicitly rejected by the readiness endpoint. | Pure availability interface or an owner that can safely acquire without allocating compute. | Blocking/nonblocking routing and callback-order tests for each supported manager. |
| Unique Cell identity and overlapping tasks | Not implemented; sequential same-FQCN reuse is an invariant. | Unique phase/attempt routing or persistent communication owner. | Controlled overlap, delayed-message and stale-session tests. |
| Broad Executor compatibility | Only Executors that explicitly opt in are accepted; the declaration is not proof. | Audit each maintained Executor for state, event, aux and async dependencies; add adaptations where meaningful. | Inventory plus task-phased E2E for every claimed Executor/recipe. |
| Client API attach and external `launch_once=True` | Correctly rejected in task-phased mode. | A separately owned resident-session design if these are product requirements. | Session reconnect, abort and cleanup matrix. |

## Evidence manifest required for every remote row

Retain one machine-readable manifest and raw evidence directory containing:

- candidate commit, tree, imported module paths and hashes on server/CP/CJ/trainer;
- exported server/client configs, launcher/site settings and any adaptations;
- Python/package environment, data digest, seeds and exact command;
- per-site logical task sequence and expected/accepted contribution counts;
- job, task, attempt and Slurm allocation IDs;
- timestamps for artifact commit, compute exit, allocation settlement, push start,
  transport completion, ACK and terminal status;
- `squeue`/`sacct`, process-tree, listener and GPU samples;
- immediate/+30s cleanup audits and +75s audits for fault rows.

The historical expectation of 30 phase allocations apparently represented 10
logical tasks. That count must be reconciled with the Controller's actual task
IDs; allocation count is never evidence of a matched logical schedule.
