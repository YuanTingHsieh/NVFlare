# Architecture B production-hardening roadmap

Status: preserved roadmap, not an active prototype acceptance gate. The shared
bounded G01--G09 campaign contract supersedes the
expanded campaign described below. Restart/rejoin, active-attempt adoption,
lost-ACK reconciliation, interrupted-upload recovery, exhaustive cancellation
and artifact faults, and the common 16-task schedule remain useful product or
benchmark work, but they are not required to decide prototype feasibility.

## Candidate identity

- Owner: Architecture B, job-lifetime CPU CJ plus a disposable application
  worker per task.
- Base: `9759e2594dc6949ef80cedf739c244334fae5f70`, isolated uncommitted checkout.
- Source identity: the branch commit based on
  `9759e2594dc6949ef80cedf739c244334fae5f70`.
- Historical report and evidence index:
  `docs/design/job_task_worker_b_prototype.md`.
- This inventory does not grandfather historical rows onto a future frozen
  candidate. Every credited row must match the common manifest or document why
  its source and assertions are unaffected.

## Common E2E row inventory

`Historical/local evidence` is diagnostic only. It does not change the common
status until the shared source, workload, resources, triggers, and assertions
match.

| ID | Common status | Historical/local evidence | Work remaining for common PASS |
|---|---|---|---|
| Q01 | NOT RUN | hello-numpy in-process passed locally with two clients and three rounds | Export and run the frozen real-federation/Slurm fixture; retain contributions and task/allocation evidence |
| Q02 | NOT RUN | hello-numpy external source config passed locally; B translated it to the disposable task-worker topology | Run the same frozen fixture and retain source config plus process tree proving translation rather than a legacy trainer |
| Q03 | NOT RUN | hello-log-streaming passed locally in both source modes and logs reached the server-side store | Run both common real E2E rows and retain expected log/analytics assertions after worker exit |
| Q04 | BLOCKED | Historical real Slurm synthetic in-process row used three rounds and 16 logical task allocations | Freeze seeds, exact task sequence, dependencies/resources, data/config hashes, and assertions; rerun if any differ |
| Q05 | BLOCKED | Historical real Slurm external row used only the reduced synthetic data; the full-size job was exported but not run | Run the frozen full workload in the external source configuration and prove B's explicit topology translation |
| Q06 | BLOCKED | Historical maintained CIFAR-10 in-process row used both A100s and completed 16 logical tasks | Freeze and compare dataset digest, seeds, exact task sequence, resources, source bundle, configs, and acceptance assertions |
| Q07 | BLOCKED | Historical maintained CIFAR-10 external-source row used both A100s and completed 16 logical tasks | Same reconciliation as Q06; retain the external source config and translated process topology |
| Q08 | NOT RUN | Local unit test proves component lookup, fresh PIDs, explicit state checkpoint, no Cell, and next-task restoration | Build and run the shared ordinary-Executor real E2E fixture without putting phase selection into the Executor |
| Q09 | NOT RUN | B reuses normal `ClientRunner` filters/publication; local tests expose the actual ACK and ordering | Run shared built-in send/broadcast/nonblocking Controller cases with input/result/server filters and publication hooks |
| Q10 | NOT RUN | Historical `T01-like` row delayed a CPU result filter for five seconds after allocation settlement | Replace with the shared actual transport-delay fixture; prove sustained upload after settlement, no worker relaunch, and eventual acceptance |
| Q11 | NOT RUN | B changes are bounded away from the ordinary resident application executor, but no common maintained-app regression row exists | Run the unchanged resident CJ path with the exact common app/config and report it separately from disposable compatibility |
| F01 | NOT RUN | Local tests cover both a raised application exception and abrupt nonzero exit, with no durable result | Run both common faults; assert bounded failure, no publication/retry, and cleanup |
| F02 | NOT RUN | Local tests kill a process group and detect/kill a descendant left after leader exit | Run both real Slurm faults; retain time order proving settlement before publication or next allocation |
| F03 | NOT RUN | Local test covers cancellation while computing and rejects later admission | Run common idle, Slurm-pending, computing, and publishing cancellation races with explicit outcomes |
| F04 | NOT RUN | Local artifact tests cover missing/partial manifests, checksum tampering, wrong owner/attempt identity, and unsupported pass-through | Run the common real fixture and retain bounded diagnostics, no publication, and cleanup |
| F05 | NOT RUN | Existing `ClientRunner` unit coverage retries the same result object after a failed send; the B artifact remains on disk | Run shared interrupted-upload/restore fault; assert no worker relaunch and bounded retry or agreed pending/failure status |
| F06 | NOT RUN | Local B instrumentation records missing ACK as not acknowledged, never as success, while preserving the committed result | Run server-accepts/ACK-lost injection; assert one server contribution, no retraining, and agreed ambiguity record |
| F07 | NOT RUN | Historical task-worker/Slurm/GPU immediate audits were clean; final harness cleanup initially found one persistent parent, then a five-second re-audit was clean | Run immediate/+30s for successes and +75s for faults on the frozen candidate; separate standing services from job-owned residue |
| R01 | UNSUPPORTED | Declared application state is stored under the job run directory, but CP restart/rejoin was not exercised and durable results are scoped to one CJ incarnation | From a quiescent boundary after ACK and compute settlement, restart one CP/runtime while the server job stays alive; rejoin a later eligible round under the same site/job without duplicate, stale ownership, or state loss |

## Current logical schedule and configuration facts

The historical three-round hello-pt P03 ledgers record this exact per-site
application task sequence:

```text
train, train, train, submit_model, validate, validate, validate, validate
```

That is eight logical tasks per site and 16 total. It must not be equated with
D's historical 30 phase receipts or apparent 10 logical tasks. The common
manifest must freeze one Controller schedule before comparative reruns.

The two accepted source configurations are:

- `in_process`: `task_script_path=client.py`.
- `external_process`: source command `python3 -u custom/client.py ...`.

B deliberately translates both configurations into the same fresh, Cell-free
task worker. The second row is source-configuration compatibility, not the
legacy resident external trainer topology.

Historical matched synthetic tasks requested one node, two CPUs, and 8 GiB.
Historical CIFAR-10 tasks requested one node, two CPUs, and one A100. Current
builder defaults are a 600-second worker timeout, 300-second Slurm-pending
timeout, one-second Slurm polling, and a ten-minute allocation limit. The
historical source did not freeze an application seed or a common dataset digest;
those are blockers, not implied matches.

## Proposed common interpretations requiring owner agreement

- Q09 uses existing Controller `send`/`broadcast`, including nonblocking
  scheduling, and the normal framework filter and publication events. B does
  not add an architecture-specific application publication-hook API. Task-input
  filters run in the CPU CJ before input commit; task-result filters run after
  compute-result commit and before transport; server validation remains at the
  server. `AFTER_SEND_TASK_RESULT` observes the actual returned ACK boolean.
- F05 retries the same committed result through the normal CPU publisher without
  relaunching compute. On the agreed bound (cancel or server task gone), the
  artifact remains and the event journal records not acknowledged/pending rather
  than success.
- F06 treats server acceptance with a lost client ACK as an ambiguity unless the
  shared protocol adds reconciliation. The server counts one contribution; B
  neither fabricates an ACK nor retrains.
- R01 starts quiescent after the previous ACK and compute settlement: no pending
  attempt or publication, declared state durable, and the server job still
  alive. It restarts the CP/runtime under the same site/job and rejoins a later
  eligible round without duplicate contribution or stale ownership. It is not
  active-attempt adoption. B's state directory can preserve declared domain
  state, but the framework's rejoin/task-eligibility records must be common to
  A/B/D.

## R01 existing mechanisms and minimal common gap

The current tree already supplies several pieces of the bounded restart but not
the complete rejoin:

- The provisioned site workspace supplies parent-process startup configuration,
  security material, policy, and server connection metadata. The deployed job
  app and job metadata remain under the same site workspace and job ID.
- Client registration authenticates the same logical site, removes an older
  same-name token, and issues a new token. Normal server authorization therefore
  rejects the superseded parent token. The existing parent monitor also stops a
  client job when its parent disappears; the common test must prove both facts
  rather than assume that no old owner survives.
- A client parent can launch a job child from the deployed app, job metadata,
  current startup kit, current parent listener, and current credentials. Today
  that launch occurs only from the server's initial `START_JOB` command.
- The running server job retains Controller state and decides task eligibility.
  Its workflow communication layer keys task history by logical client name and
  can resend an unresolved task. R01 deliberately avoids that path by restarting
  only after the prior result was accepted and ACKed.
- B's declared domain state persists at
  `.job_task_worker_b/state/{state_id}` under the same job run directory. A new
  CJ incarnation can read it, while its new incarnation ID prevents automatic
  adoption or publication of old attempt artifacts.

The missing common behavior is a server-authorized way to recognize that a
currently running job's logical site has re-registered, re-establish any
resource reservation, and launch/sync a new job runtime from the already
deployed app using current credentials. The initial `START_JOB` path is not
automatically replayed on re-registration, and the code explicitly documents
that CJ restart is not currently supported. No server attempt ledger should be
assumed.

A minimal common durable record may contain job/site identity, last accepted
task identity, artifact identity/digest, compute-settled state, and publication
state. For this bounded row, rejoin is admissible only when that record is
unambiguously quiescent (`compute_settled` and `publication_acked`, no active or
pending attempt). The server remains the admission and task-acceptance
authority. On restart, B restores only declared domain state and resumes normal
task acquisition; it neither launches nor publishes any old attempt. This is a
proposal to review with A/D, not a B-private API and not a pass claim.

## Existing semantics that cannot remain unchanged

| Area | Original behavior | Disposable lifetime/API conflict | Explicit B replacement | Scope |
|---|---|---|---|---|
| Attach, launch-once, resident and interactive sessions | One application process/session survives multiple tasks | A fresh application instance is required per task | Reject Attach; translate both accepted source configs to a fresh worker; disclose loss of resident memory/session | Common disposable constraint; translation is B-specific |
| Stateful Learners/application objects | Python object fields may carry state across tasks | Worker objects die after every task | Application explicitly checkpoints the minimum domain state under `job_task_worker_state_dir` | Common constraint; B path/API is specific |
| Engine, Cell and auxiliary messaging | Application executor may use the live job engine/Cell | Worker is intentionally Cell-free and has no federation credential/session | Minimal task-local component registry/context; federation aux calls fail explicitly | B boundary; other candidates must declare theirs |
| Lifecycle handlers | Application may observe job-long events in one process | Per-task worker sees only task-local startup/execution/shutdown | Resident CJ keeps job lifecycle; worker fires bounded application lifecycle around one task | Common semantic decision, B implementation |
| Filter placement | Filters surround an in-CJ Executor invocation | Compute ends before CPU publication | Existing input filters precede handoff; existing result filters follow committed output; server filters stay server-side | Common semantics, placement differs |
| Large/lazy/pass-through results | Trainer-dependent references may outlive an execute call | Disposable worker cannot leave unresolved trainer-owned data | Require eager FOBS artifact and reject pass-through | Current B limitation; common large-result strategy needed |
| CP restart and adoption | Resident path relies on framework job lifecycle | New CJ gets a new handoff-scope identity and does not adopt pending attempts | Preserve declared domain state only; no result adoption yet | Shared R01 protocol needed; active adoption beyond R01 |
| Publication retry/deduplication | `ClientRunner` retries transport and checks whether the server task remains | Durable result ownership crosses compute/publication boundary | Reuse normal publisher; retain artifact and journal actual ACK/not-ACK | Framework behavior plus B durability gap |
| Task retry | Controller/framework policy may issue another task | Re-execution can duplicate non-idempotent state changes | No silent worker retry; application must be idempotent or protocol must add a transaction/idempotency key | Common correctness issue |
| Relay/custom managers | Executors may depend on richer workflow/engine interactions | Minimal worker engine omits them | Currently unqualified or explicitly rejected | Broad gap, not common bounded gate unless selected |
| DDP/multi-node | Application may own multiple ranks/nodes | B launcher currently enforces one node per task | Unsupported pending a rank-aware task-worker contract | Broad gap |
| Docker/Kubernetes | Other launchers may own isolation and lifecycle | Only local process and Slurm worker launch are implemented | Unimplemented launcher adapters and containment policy | Broad gap |

## Possible but unimplemented or unqualified

| Capability | Current B status | Implementation needed | Tests/evidence needed |
|---|---|---|---|
| Frozen common job fixture | Blocked on manifest | Consume exact task schedule, resources, seeds, dataset/config hashes and assertions | Q01--Q11 exported configs and evidence schema |
| Actual slow upload | Slow filter only; does not qualify | Shared isolated transport-delay/fault injection below the committed-output boundary | Q10 sustained interval, settled compute tree, no relaunch, eventual ACK |
| Publication interruption/recovery | Same-process standard retry exists; no durable publisher restart | Agree bound/status; add durable pending record only if common protocol requires it | F05 failed send, restore, same attempt/result digest, no compute |
| ACK-loss ambiguity | Instrumented not-ACK; real accepted-but-lost case unqualified | Shared ACK-loss trigger and optional common reconciliation | F06 server count one, client ambiguity, no retrain |
| CP restart between rounds | Unsupported/unqualified | Common rejoin/task eligibility plus restoration of declared state | R01 later eligible round, no duplicate/state loss |
| Cancellation matrix | Computing covered locally only | Fault controls for idle, pending, compute, publish; ensure publisher and allocation owners settle | F03 time-ordered events and cleanup |
| Artifact fault matrix | Digest/identity covered locally | Common attempt fixture for missing/partial/corrupt/wrong-attempt states | F04 bounded diagnostics and no server acceptance |
| Ordinary Executor/component E2E | Local unit only | Shared server workflow and exportable job | Q08 component/event/state assertions |
| Built-in Controller semantics | Inherited but not commonly qualified | Shared send/broadcast/nonblocking workflow and selected filters | Q09 routing/order/hooks/ACK evidence |
| Resident compatibility | Not modified but not rerun | Common unchanged resident job | Q11 full maintained-app evidence |
| Stateful Learners beyond hello-pt | Only minimal model checkpoint adapted | Per-application state contract and migration | Multi-task restoration and crash-boundary tests |
| Large results/streaming | Eager artifacts only | Chunked durable commit or framework streaming handoff independent of worker lifetime | Size, interruption, digest and cleanup rows |
| General CP/publisher adoption | Not implemented | Durable journal, owner lease, adoption and idempotency protocol | Restart at every boundary and duplicate-suppression tests |
| Relay/custom managers/aux | Unqualified or rejected | Expand task-local service surface without exposing federation authority | Feature-specific E2E and negative isolation tests |
| DDP/multi-node | Unsupported | Multi-rank launcher, collective cancellation and output ownership | Rank/descendant/allocation settlement tests |
| Docker/Kubernetes | Unimplemented | Launcher and containment adapters | Equivalent lifecycle/fault/security evidence |

## Superseded coordination note

The desktop safety gate rejected attempts to send the proposed contract and
resource/slot request to the Colossus and D task IDs, even though the comparison
task requested coordination. No shared-node run will be started without a
verified slot. Work that does not depend on the remote manifest continues
locally; the common manifest, slot, and owner agreements remain explicit
blockers.

## Latest local verification

- 250 B, `ClientRunner`, and Slurm launcher/manager tests pass locally.
- New gap-closing coverage verifies a raised application exception, missing and
  partial manifests, wrong-attempt identity, and retention of committed result
  bytes after a missing ACK.
- Targeted exception and nonzero-exit simulations abort without a committed
  model result; a separate standard resident `ClientAPIExecutor` simulation
  completes successfully.
- Black, isort, and flake8 pass for the current changed Python files.
- No remote work was started while the common manifest and slot remain pending.
