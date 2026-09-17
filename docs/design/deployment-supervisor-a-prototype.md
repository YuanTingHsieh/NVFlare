# Deployment-supervised task runtime: executable Architecture A prototype

**Status:** experiment, not a selected architecture or production feature
**Architecture A starting revision:** `9759e2594dc6949ef80cedf739c244334fae5f70`
**Phased-D reviewed reference:** `23faec2cfbc2f7359b065e4b521857f7e5fc3b3c`
**Phased-D prototype range used for size comparison:**
`83ec31a8e14bc41f788ec5a739327298173632b5..23faec2cfbc2f7359b065e4b521857f7e5fc3b3c`

## Result

The experiment is a deliberately narrow Architecture A candidate, not an
argument that A must win. An opt-in client launcher now keeps one logical job in
the CP, acquires ordinary built-in broadcast/send tasks from an SJ adapter,
stages immutable inputs, launches a fresh local or Slurm worker, waits for
allocation settlement, validates a durable result, and publishes it through the
CP with an explicit server ACK. The worker has no federation Cell/session and no
publication credentials.

This does **not** establish that A is generally compatible or cheaper than D.
The slice confirms A's clean worker/publication ownership for supported tasks,
while the audit identifies substantial adapters for Engine-dependent Executors,
Learners, resident sessions, and existing server Controllers.

## Executed topology

```text
server deployment task service (prototype Controller adapter)
                  ^ acquire / publish
                  |
site deployment supervisor (shared across jobs)
  | input filters -> durable input
  | launch/cancel/wait for settlement
  v
fresh task worker (Executor or flare.receive/send script)
  | no Cell, no task polling, no publication
  v
durable result -> result filters -> publication handlers -> publish/ACK
```

There is no per-job resident helper. A helper that lives only for one job and
owns this lifecycle would be Architecture B, even if named a deployment helper.

## Implemented guarantees

- Attempt, job, and task identity are checked at both handoff reads.
- FOBS payloads are written and fsynced before an immutable commit manifest.
- Result publication rejects lazy/pass-through data and never depends on a live
  trainer as a byte source.
- Worker launch strips the CJ authentication token, signature, and session ID;
  a test verifies that job credentials are absent from the worker environment.
- Server F1/F4 filters remain in `ServerRunner`. Job client filters are loaded
  from deployed job configuration and run inside the fresh worker. The earlier
  in-memory adapter proves site-before-job ordering, but site privacy-filter
  loading is not integrated and must not be inferred compatible.
- The `JobHandleSpec.wait()` boundary completes before result validation or
  publication. This is real local-process settlement, not a scheduler/GPU claim.
- Timeout and explicit cancellation terminate the worker process group with a
  bounded TERM-to-KILL grace. Failed attempts are not published.
- Publication handlers observe the actual before/after service publish boundary.
- An application can declare `state_id`; the framework supplies a durable state
  directory without making generic task staging the application's concern.

The prototype does not implement attempt retry. Phased D also does not provide
general durable attempt retry or CP restart/adoption, so neither is used as an
asymmetric acceptance gate here.

## Integrated boundary and present limits

`DeploymentTaskService` is installed in each SJ. It authenticates CP requests,
calls the existing `ServerRunner.process_task_request` and
`process_submission`, accepts only the exact built-in `WFCommServer` with
ordinary `BcastTaskManager`/`SendTaskManager`, and returns ACK only after the
existing communicator has synchronously marked the client result received.
This preserves Controller scheduling callbacks and server F1/F4 filters for
that bounded surface. Unit tests prove accepted versus dropped publication.
The source-bound v4 federation reached the intended SJ but exposed the stream
registration defect described below. The corrected v7 candidate subsequently
completed the bounded real-federation gate on Colossus; the exact run and the
remaining qualification gaps are recorded below.

The service registers exact acquire/publish callbacks on the SJ Cell's
streaming AUX channel. These are raw Cell callbacks, not entries in AuxRunner's
topic table, because authentication also consumes the Cell token/session
headers before reconstructing the peer context. `Cell.register_request_cb`
must therefore receive `channel`, `topic`, and `cb` by keyword: Cell's dynamic
dispatch uses the channel keyword to install both CoreCell and stream/blob
callbacks. Positional registration installs only the CoreCell callback and
lets streamed AUX requests fall through to AuxRunner's wildcard as unknown.

The CP-side launcher maps deployed Client API in-process and external-process
Python-script configurations to the same fresh script-worker contract and
reuses the existing Slurm launch plan and manager. It does not reconstruct a CJ
Engine/Cell. There is no component registry or Engine-service proxy beyond
importable Executor/filter factories plus constructor arguments. `START_RUN`,
`BEFORE/AFTER_TASK_EXECUTION`, and `END_RUN` are task-scoped. Cross-task state is
an explicit durable state directory. These are the bounded interface being
tested, not temporary implementation details to conceal.

## Decision hypotheses and experiments

| Assumption under test | Expected observable | Evidence required | Decision impact |
|---|---|---|---|
| A limited worker interface preserves representative task-local workloads | Existing script behavior and selected Executors survive with only declared state/config adapters; missing dependencies are identifiable | Actual deployed configs, train/validate/submit outputs, filter/callback traces, code audit of Engine/component/aux use | Additional Engine/Cell dependencies increase A's adapter and migration cost; widespread dependence weakens A's premise |
| Removing CJ reconstruction/session setup and nested trainer supervision matters operationally | At matched placement A has fewer process/session/allocation transitions and measurably lower task-boundary latency or host overhead | Matched A/D timestamps, process trees, Cell/session counts, Slurm accounting, CPU/RSS and transfer bytes where observable | Only material, attributable savings can justify A-specific API/service migrations; topology alone is insufficient |
| Failure behavior conforms to the ownership design | Worker/tree death settles before publication; missing ACK is not success; abort admits no later task | Injected leader/descendant death, compute failure, lost/dropped publication, idle/pending/running abort, event and scheduler receipts | Defects are fixed; missing adapters are estimated; repeated violations caused by the boundary require contract revision |

GPU release before slow upload and framework-owned staging are conformance
requirements already demonstrated by D. They are not by themselves benefits of
A. Clearing three credential variables shows reduced credential exposure, not a
secure isolation boundary.

## Common infrastructure versus architecture-specific work

| Surface | Common to A/B/D target | A-specific in this experiment | Phased-D-specific at reference commit |
|---|---|---|---|
| Durable data | Eager, manifest-last attempt artifacts; identity/digest validation | Supervisor is stable reader/publisher | PULL/COMPUTE/PUSH CJs exchange artifacts |
| Allocation | Launcher handle must settle before reuse/publication | One fresh worker allocation per task in the local profile | Three phase allocations per task in the prototype profile |
| Task authority | Attempt identity, stale/duplicate rejection, cancellation, terminal outcome | Deployment service owns acquisition/publication | SJ remains task handler; CP probes and sequences CJs |
| Application data | Framework stages/publishes; application handles only domain state | Worker receives one staged task; script adapter is process-local | Transfer CJs and compute CJ split the lifecycle |
| Filters | Preserve selected filters, ordering, bytes, and failures | Supervisor owns both client chains in this slice | Compute CJ reuses current chains |
| Client API pass-through removal | Durable output must not depend on trainer liveness | Direct script worker uses local `flare.receive/send` adapter | D still uses a CJ-hosted ClientAPIExecutor path; removing trainer pass-through is separate common work |
| Server changes | First-class task/attempt service is needed for target semantics | Service moves to SP/deployment helper and needs Controller adapter | Server probe/admission remains tied to SJ/WFCommServer |

## Existing semantics that cannot be preserved unchanged

These are lifetime/state incompatibilities shared with D's fresh-task mode
unless the A-specific conflict says otherwise. Keeping the unchanged legacy CJ
path is an equal fallback policy for either candidate; it is not compatibility
inside either disposable mode and does not imply a resident-A mode.

| Existing behavior | Conflict with a disposable task worker | Possible explicit contract | A-specific effect/evidence |
|---|---|---|---|
| Executor, Learner, or handler keeps required state only in Python objects between tasks | The object and process disappear after the task | Checkpoint declared state, let a framework service retain an explicitly published value, or select legacy resident CJ execution | Shared constraint. `DeclaredStateExecutor` and hello-pt's managed state file prove this prototype's explicit state-directory option; they do not make file persistence an inherent A requirement or reconstruct undeclared Python locals. A framework-owned retained-model/standard submit-model alternative is possible but not implemented or qualified here. |
| Attach, `external_process(launch_once=True)`, Flower/TIE, interactive XGBoost, split learning, or another live trainer/session remains addressable between tasks | There is no live task worker between assignments | Model the whole session as one task, redesign as tasks, or retain legacy CJ execution | Shared constraint. A deliberately provides no resident worker. Its config adapter rejects Attach and cannot preserve arbitrary external commands/session identity. |
| A returned result contains lazy/pass-through references served by the worker | The later consumer cannot reach the source after worker exit | Materialize and commit before exit, or move ownership to a durable object service | Shared constraint. A rejects pass-through at the artifact boundary before launch/publication. |
| Background work or a result callback completes after `Executor.execute()` returns | Process exit cancels work not represented by the committed result | Keep the attempt alive until final commit, or expose an asynchronous-attempt protocol | Shared constraint. A treats worker exit plus committed result as the only terminal compute boundary. |
| `DO_TASK` arrives over aux at an already-running CJ, possibly concurrently | No CJ exists while idle; an untracked aux RPC defeats one-attempt ownership | Convert it to a scheduled task/lease, route through a persistent site service, or use legacy CJ execution | Shared lifetime conflict plus A API break: the worker intentionally has no Cell or aux proxy. |
| A job-lifetime `START_RUN`/`END_RUN` handler needs one object instance or transient compute state at publication ACK | Compute and publication have different owners and lifetimes | Define job/task/publication scopes and use explicit durable handoff | Shared scope migration. A additionally moves publication to CP, so a handler needing both worker objects and ACK cannot be preserved by a local registry alone. |

## Can be supported, but is not implemented or fully qualified here

These are qualification or adapter gaps, not claims of architectural
impossibility.

| Capability | A candidate status | Work needed |
|---|---|---|
| Existing non-Client-API Executors | A no-argument Executor and constructor args are supported; representative Engine-dependent families are not | Audit and classify each family; define a local component registry, explicitly selected Engine-service proxies, task-scoped lifecycle mapping, and durable state declarations. Do not recreate an implicit full CJ under another name. |
| Server-side nonblocking `send`/`broadcast` | The integrated adapter delegates request/submission to exact built-in `WFCommServer`; unit tests cover acceptance and ACK, not a real federation yet | Qualify callbacks, timeouts, result ordering, two clients/rounds, server filters, abort, and logical-job terminal behavior on Colossus. |
| Very large results | Eager manifest-last FOBS is written and read before network publish | Add a disk-backed stream/artifact source while preserving completed worker/filter/settlement and ACK ordering; measure duplicate materialization and transfer cost. |
| Multi-node Slurm and DDP | Explicitly rejected; one Slurm node is supported | Define rank authority, one commit, process-tree cleanup, rendezvous, multi-node settlement, and tests. |
| Kubernetes, Docker, and other launchers | Local process and Slurm adapters only | Implement the same settled-handle and shared artifact contract per launcher/storage model. |
| CP restart and attempt adoption | Not implemented | Persist scheduler IDs, attempts, receipts, and terminal decisions; reconcile before acquire/launch/publish. |
| Attempt retry | Attempt identity and stale publication checks exist; general retry and lost-ACK recovery do not | Add leases, idempotent commit/publication, deduplication, and retry policy. |
| Relay and custom task managers | Rejected by exact manager checks | Add a non-mutating scheduler admission API and qualify ordering rather than guessing from custom types. |
| Full lifecycle-component compatibility | Worker runs only Executor task events and configured job filters; CP owns publication | Classify every handler into deployment/job/task/publication scope and define configuration/dependency mapping. |
| Site privacy filters | Not loaded into the managed worker or CP supervisor | Define the trusted execution host, ordering with job filters, policy/component loading, failure behavior, and isolation implications. |
| Arbitrary external-process Client API commands | Adapter accepts a Python script token and maps it to the direct script runtime | Either define this narrower API migration or add a bounded executable protocol; shell wrappers and non-Python trainers are not silently equivalent. |

## Compatibility audit

The categories apply to the new managed-A path. The existing resident CJ path
is not modified and remains usable.

### T2/T4/T7 source and configuration disclosure

| Row | Application and job/export changes | Actual backend/topology and evidence | Result/limit |
|---|---|---|---|
| T2a Client API `in_process` | The repository's hello-pt `job.py` and recipe are unchanged. As this adapter's current continuity choice, `client.py` is changed to load/save `cifar_net.pth` under `NVFLARE_TASK_STATE_DIR` with an atomic replace (`examples/hello-world/hello-pt/client.py:36-49,119-124,187-188`). The real G01 export used a harness-selected train-only, two-client/three-round synthetic schedule, not the optional submit-model/cross-site-evaluation schedule from the full recipe. | The exported `ClientAPIExecutor` definition is consumed as configuration, but the class and original in-process CJ backend do not run. `config._script_worker` maps it to `WorkerDefinition(kind="script")`; the fresh worker embeds `InProcessClientAPI` for one task (`config.py:42-72`, `worker.py:125-178`). Real v7 G01 passed this mapping. Separately, the exact existing hello-numpy `client.py` runs unchanged in local managed-worker tests. | Pass for the named real train-only mapping; unchanged hello-numpy is local evidence only. Neither proves full-original-job compatibility. The file edit is not inherent to A: a framework service could retain the model passed through `flare.send` and answer a later standard submit-model task, but that alternative is not implemented or qualified. |
| T2b Client API `external_process`, `launch_once=false` | The same source config's command is parsed only to find a Python script and its arguments; no trainer source change is required (`config.py:47-56`). Exact remote export identity is retained by Colossus. | A again launches the direct task script through the same fresh worker and embedded in-process adapter. It does **not** launch the original external backend/process/session. | Configuration-to-script mapping is locally tested; the targeted real CUDA row is pending. Even a pass qualifies the mapping, not original external-backend/session compatibility. |
| T2c Attach | No application or recipe migration is supplied for managed A. | `load_worker_routes` has no Attach/session adapter; supported Client API modes are only `in_process` and `external_process`. | Unsupported in the new path. The unchanged legacy resident Attach path is a separate T7 question and is untested here. |
| T2d external `launch_once=true` | No session-preserving application/job adaptation exists. | `launch_once` is not represented in `WorkerDefinition`; every A worker has task lifetime. Accepting/parsing surrounding args cannot preserve a job-lived trainer. | Unsupported semantically in the new path; must not be labeled preserved. Legacy resident `launch_once=true` is untested here. |
| T4 workspace/state continuity | This implementation makes hello-pt explicitly save the latest local weights and reload them in a later fresh worker through the stable job/client state directory. The generic local fixtures persist a counter in the same contract. | The framework chooses `state_dir` under supervisor workspace and supplies it through `NVFLARE_TASK_STATE_DIR` or FLContext (`supervisor.py:241-269`, `worker.py:96-106,153-168`). A different framework adapter could retain the model already supplied through `flare.send` for later standard submit-model handling, but it is not implemented here. | Local continuity evidence passes for the file-backed choice. Train-only G01 did not include a later submit-model read, so it is not remote T4 evidence. File persistence neither reconstructs arbitrary Python locals nor defines a mandatory A architecture contract. |
| T7 legacy resident | No legacy recipe, exported JSON, or client/trainer file was changed or executed for a credited A row. | Managed A is disabled to select the unchanged resident CJ path, but retention of code is not execution evidence. | T7a in-process, T7b external false/true, T7c Attach, T7d XGBoost variants, T7e Flower/TIE, and T7f CCWF/swarm/aux are all unqualified in this campaign unless a separately identified run is supplied. |

Launcher-only integration is independent of these application disclosures:
`ClientSlurmJobLauncher` gains `DeploymentSupervisorLauncherMixin`; the opt-in
`deployment_supervised` settings select the CP-owned logical supervisor and
per-task Slurm worker while reusing the normal launch-plan/resource/storage
configuration. `ServerRunner` installs the job-scoped task service. These are
framework/deployment changes, not evidence that an original Client API backend
or session ran.

| Surface examined | Classification | Evidence / required action |
|---|---|---|
| Plain synchronous no-argument `Executor` | **Adapter required** | Signature is unchanged and `IncrementExecutor` runs in a fresh process. Production job-config construction, BYOC loading, Engine services, and constructor arguments still need a worker bootstrap adapter. |
| Ordinary broadcast/send, multiple clients and rounds | **Supported in slice** | Tests execute three rounds on two clients plus a directed validate task. Relay/custom task managers are not inferred compatible. |
| Existing hello-numpy `flare.receive/send` script | **Unchanged for managed task** | The exact existing script runs twice through the managed worker, representing scripts currently selected by both in-process and external-process ClientAPIExecutor configurations. It contains no A-specific phase or persistence checks. |
| Direct task script with `torchrun` | **Unqualified** | A can structurally launch `supervisor -> torchrun -> task-worker ranks`, but rank-scoped result ownership, group cancellation, DDP rendezvous, GPU resources, and multi-node settlement are not implemented or tested. The topology alone is not multi-GPU proof. |
| `LearnerExecutor` / `ModelLearnerExecutor` | **Adapter required** | They resolve Learners and components through `fl_ctx.get_engine()`. The prototype worker intentionally has no full Engine registry. A worker component-graph bootstrap or a new explicit learner contract is required. |
| Executor/Learner instance fields across train/validate/submit | **Explicit continuity contract needed** | Fresh workers discard object fields. `DeclaredStateExecutor` proves the file-backed option, and this adapter's hello-pt edit loads/atomically saves `last_params` when the framework provides `NVFLARE_TASK_STATE_DIR`; legacy behavior remains unchanged otherwise. A framework-owned retained-model path is another possible contract but is not implemented/qualified, and neither option restores arbitrary Python locals. |
| Pure Shareable filters | **Supported with intentional host change** | Site-before-job order is tested in both directions. Filters now run in the trusted supervisor, after/before durable worker handoff. |
| Engine/Cell/component-dependent filters | **Adapter required** | Full filter ordering is retained, but a filter expecting the CJ Engine, Cell, application components, or lazy live byte source cannot run unchanged in this supervisor. |
| Worker-local lifecycle hooks | **Adapter required** | Executor `START_RUN`, `BEFORE/AFTER_TASK_EXECUTION`, and `END_RUN` run per task. This is not equivalent to job-lifetime events or a complete handler graph. |
| Publication callbacks | **Adapter required, hook proven** | A supervisor-owned before/after publication hook observes the real publish boundary. Existing FLComponent handlers still need configuration/dependency mapping. |
| Job metrics and script analytics | **Adapter required** | Supervisor lifecycle is recorded in `events.jsonl`; Client API logs are consumed and counted. Existing JobStatsReporter/MetricsCollector event routing is not installed. |
| ClientAPIExecutor `launch_once` and in-process resident loops | **Unsupported in managed A slice** | Their lifetime intentionally spans tasks. Use the unchanged legacy CJ path or migrate the workload contract; no resident-A mode is proposed. |
| Attach / independently operated trainer | **Unsupported in managed A slice** | Current Attach owns a live Cell session and rendezvous. It needs an explicit deployment-supervisor session adapter, not silent conversion to task scope. |
| Aux/P2P Executor, TaskController, CCWF/swarm, split/interactive workflows | **Unsupported/unqualified** | Representative P2P code calls `engine.send_aux_request` and registers handlers. No relay exists in the worker contract. Lack of a test alone is not the reason; the required live service is absent. |
| Streaming and arbitrary application networking | **Unqualified** | Worker networking is not prohibited, but streaming lifecycle, policy enforcement, and cancellation semantics were not qualified. |
| Resident/session workflows | **Unchanged outside A; unsupported inside slice** | The branch does not remove or alter the resident CJ implementation. Preflight rejects `lifetime="resident"` with an explicit migration message. |
| GPU/Slurm allocation settlement | **Adapter implemented; real settlement qualified** | The adapter reuses the existing Slurm plan/manager and publishes only after its handle settles. Colossus job `000eeefc-c557-4e35-90d8-d99e6f3907ac` completed all six client task allocations with scheduler return code 0 before publication. A separate controlled slow-network-upload observation with the GPU already free is still required. |

## Implementation and migration cost versus phased D

Implementation size is intentionally excluded as architectural evidence. The A
candidate is narrower and still has missing adapters; the D candidate contains
its own phase-incarnation/session/allocation machinery. Diff size does not
measure either state machine's inherent maintainability or operational cost.

Phased D's implemented cost is concentrated in CP sequencing, SJ availability
probing, transfer-only configuration, canonical CJ reconstruction, task-session
preservation, and three allocation handoffs. Its compatibility advantage is
real: it reuses ClientRunner, job configuration, Engine/Cell services, the
Executor router, and current task filter/event machinery.

A removes the per-task reconstruction of the full CJ runtime and gives durable
publication a stable deployment owner. It also removes an intermediate
ClientAPIExecutor process/session layer for direct scripts. However, its missing
productization surfaces remain central: broader Controller/task-manager
adaptation, component-graph loading, selected Engine-service proxies,
job/publication lifecycle mapping, additional scheduler adapters, site privacy
policy, mixed-version negotiation, and resident/interactive workload migration.

For the tested task-scoped script profile, A has the cleaner ownership chain:
one deployment supervisor, one compute worker, durable output, then publication.
For broad existing Executor/Learner compatibility, D currently has the lower
migration cost because it retains the CJ runtime surface. Selection should turn
on which workload set is required and on matched Slurm/GPU, failure, and
server-integration tests—not line count.

## Operational and maintenance inventory

This inventory separates source-reviewed consequences from observations. It
does not use patch size, bug count, or the unmatched G01 duration as an
architecture score.

| Claim | Concrete evidence | Application change | Framework responsibility | Remaining uncertainty |
|---|---|---|---|---|
| Acquisition and publication retain existing Controller authority while A owns attempt identity | `DeploymentTaskService` authenticates the CP, calls `ServerRunner.process_task_request`, records `(client, task_id) -> attempt`, calls `process_submission`, and returns ACK only after `WFCommServer` records receipt (`federation.py:60-196`). G01 server logs show 2/2 accepted results in every round. | None for supported broadcast/send workflows | New SJ service and CP client; reuses ServerRunner, the built-in communicator, existing client identity, Cell transport, and server filters | Relay/custom managers and broader Controllers are rejected or unqualified; attempt records are memory-only |
| The CP supervisor owns staging, cancellation, settlement, and the publish decision | `DeploymentTaskSupervisor.run_once` records acquired/launched/settled/published events, terminates the active handle on cancellation, requires terminal RC0, validates the committed result, then publishes (`supervisor.py:195-348`). | Domain state must be explicitly durable; generic task I/O remains framework-owned | New supervisor and worker specification; reuses `JobHandleSpec` and the selected launcher | CP restart/adoption and general retry are deferred; event-file retention/rotation is prototype-only |
| Durable commit is worker-owned; publication never reads from a live trainer | Payload is fsynced and linked before the manifest is committed last; reads recheck identity, size, and digest; lazy/pass-through data is rejected (`artifacts.py:72-158`). G01 proves all six allocations settled before publication. | Lazy results must be materialized; declared state uses the supplied state directory | New artifact contract and FOBS handoff; reuses FOBS serialization | Large-result streaming, total disk I/O/copy count, retention policy, and controlled actual slow-upload timing remain unmeasured |
| Terminal job state remains a CP-owned logical handle | `DeploymentFederationJobHandle` reports success only after the server sends `END_RUN`; abort/error paths cancel the supervisor (`federation.py:283-329`). | None for the qualified task-scoped slice | New logical handle; reuses launcher handle semantics and server workflow terminal state | Lost-ACK reconciliation and CP restart are deferred |

### Compatibility and adapter cost

| Surface | Current A behavior and source evidence | Exact migration or limitation |
|---|---|---|
| Client API script | Existing in-process and external-process source configurations are translated to one fresh script worker; the worker embeds `InProcessClientAPI` only for that task (`config.py:42-72`, `worker.py:125-178`) | Representative Python scripts remain unchanged. External commands are narrowed to a Python script and `launch_once=false`; resident/Attach/session identity is unsupported inside managed A |
| Ordinary `Executor` | v8 loads the deployed `custom/` module inside the fresh worker, preserves the ordinary `execute()` signature, and runs task events (`config.py:101-134`, `worker.py:109-122,181-215`) | Constructor args work; object fields do not survive tasks. Job-lifetime event semantics require redesign or the legacy resident path |
| Component-dependent Learner/Executor | A supplies a deliberately small task-local component registry (`worker.py:65-106`) | Code requiring full Engine/Cell services, Learner lookup, aux, or a job-lifetime component graph needs an explicit adapter and remains unqualified |
| Filters | Configured job input/result filters are selected per task and loaded in the worker; supervisor-side filter hooks also exist (`config.py:75-98,124-130`, `worker.py:87-93`) | Pure Shareable filters are supported. Engine/Cell-dependent filters and site privacy-filter loading are not established; filter host/lifetime changes must be disclosed |
| Lifecycle/publication handlers | Executor START/BEFORE/AFTER/END events are task-scoped; A publication handlers surround the real service publish boundary (`worker.py:109-122`, `supervisor.py:318-330`) | Existing job-lifetime handlers and handlers needing both worker objects and ACK need scope/configuration mapping; the A hook is not general legacy-handler compatibility |

### New versus reused framework surfaces

- New A surfaces: deployment task service/client protocol, deployment
  supervisor, durable artifact contract, fresh worker bootstrap, deployed-config
  route translation, and the thin Slurm task-handle normalization.
- Reused surfaces: `ServerRunner.process_task_request`/`process_submission`,
  built-in `WFCommServer`, normal authentication/Cell transport, FOBS,
  `Executor.execute`, FLContext/Shareable/filter interfaces, existing Slurm
  launch-plan construction/manager, and `JobHandleSpec` termination/settlement.
- Reuse is bounded: the worker does not reconstruct a CJ Engine, Cell, full
  component graph, or job-lifetime event bus.

### Actual change-impact examples

1. Streamed AUX routing: changing `Cell.register_request_cb` to keyword
   `channel`/`topic`/`cb` in `federation.py` fixed the real unknown-topic
   failure without changing the task protocol. The regression crosses the
   real `Cell.__getattr__` dispatch boundary.
2. NFS and terminal status: bounded `EAGAIN` retry plus blocking normalization
   stayed in `artifacts.py`; Slurm running/terminal normalization stayed in
   `slurm.py`, while `supervisor.py` now treats `wait()` as a barrier and reads
   status from `poll()`. G01 showed both prior failures absent across six tasks.
3. Deployed ordinary modules: v8 changes only `config.py`, `supervisor.py`, and
   `worker.py` to carry the deployed `custom/` path into the fresh worker. A
   regression launches a custom-only Executor absent from ambient
   `PYTHONPATH`; the supervisor still does not import application code.

### Process and measurement inventory

- Persistent/control: the existing server job process hosts
  `DeploymentTaskService`; each site CP hosts the logical job handle and
  supervisor. Managed A adds no resident per-job CJ/helper process and creates
  no worker Cell/session.
- Transient: the tested placement launches one fresh application worker in one
  Slurm allocation per logical task. v7 G01 observed six client task
  allocations for six contributions, plus the common server allocation.
- Measured: v7 G01 completed in 44.762163 seconds and cleanly released all
  owned processes, allocations, and GPU processes immediately and at +30
  seconds. This interval includes queue/startup/computation/publication and is
  not a startup benchmark.
- Missing: no retained A task-boundary startup split, CP/supervisor CPU/RSS,
  worker peak RSS, serialization/copy volume, or idle baseline has yet been
  reported. The RTX/TITAN hardware and synthetic workload differ from B/D, so
  no comparative performance or resource-efficiency conclusion follows.

## Compact decision assessment

This assessment separates current evidence, architectural consequence, and an
untested proposal. It does not score A by patch size, passed-row count, or the
unmatched elapsed times above.

| Question | Current code or test evidence | Consequence and remaining uncertainty |
|---|---|---|
| What ownership does A remove? | With `deployment_supervised=true`, the launcher bypasses the resident CJ path and constructs one CP-resident service client, supervisor, and logical handle (`federation.py:332-388`). Each task launches the task-worker module directly through the existing Slurm plan (`slurm.py:41-69`). | A removes per-task full-CJ reconstruction and, for Client API scripts, the additional backend/trainer session. This is a design consequence, not yet a matched cost saving: CP CPU/RSS, startup decomposition, and transfer bytes remain unmeasured. |
| What ownership does A introduce? | The SJ service owns authenticated acquisition, attempt identity, stale-result admission, and ACK (`federation.py:60-196`). The CP supervisor owns route selection, staging, cancellation, allocation settlement, result verification, and publication (`supervisor.py:195-348`). The worker owns only task execution and durable result commit (`worker.py:181-215`). | The ownership chain is explicit, but it duplicates portions of current client task routing, lifecycle events, and config interpretation in new A adapters. Attempt state is memory-only; CP restart/adoption, retry, and lost-ACK reconciliation are unproven. |
| Where do CellNet and credentials live? | Acquire/publish remain authenticated Cell requests at the SJ/CP boundary (`federation.py:83-115,199-280`). Before launch, the supervisor removes auth token, token signature, and SSID; the Slurm adapter also removes them from the launch plan (`supervisor.py:277-282`, `slurm.py:55-68`). The worker constructs no Cell (`worker.py:96-106,125-178`). | A concretely reduces federation-credential exposure in application compute. It is process-boundary reduction, not a security sandbox: the worker still runs supplied Python with filesystem and any scheduler-granted network access. |
| Where do filters, commit, cancellation, and publication live? | Configured task filters are selected from deployed config (`config.py:89-145`) and execute around task data/results in the fresh worker (`worker.py:87-93,189-207`); supervisor hooks can additionally filter before staging and after settlement (`supervisor.py:208-215,248-259,306-324`). Artifacts are eager FOBS payloads with payload-first, manifest-last fsync/link commit and identity/digest verification (`artifacts.py:72-162`). Cancellation terminates the active launcher handle, and publication occurs only after terminal RC0 and verified result (`supervisor.py:195-198,284-320`). | Pure Shareable filters fit; site privacy filters and Engine/Cell-dependent filters do not yet have a proven host/config mapping. Durable eager commit makes GPU release independent of network ACK, but large-result copy/streaming cost and retention policy remain unmeasured. |
| What does a worker failure affect? | Nonzero exit, timeout, corrupt/missing result, or filter failure enters the supervisor exception path, terminates any active handle, and publishes an `EXECUTION_EXCEPTION` result through `service.fail` (`supervisor.py:284-348`, `federation.py:275-280`). The v8 T4 run showed that `CrossSiteModelEval` can tolerate such results and still let the server report `COMPLETED`. | Worker failure is bounded to the attempt/allocation, but the Controller still defines whether the logical workflow fails. A deliberately does not add an unconditional server panic: doing so would override optional-task, quorum, and expected-failure semantics. A common test-specific required-result assertion is preferable when the qualification contract is stricter than the Controller. |
| What does a supervisor failure affect? | An exception in acquisition/publication or a sustained service outage makes the logical handle cancel its active supervisor and return `EXECUTION_ERROR`; explicit abort also cancels the supervisor (`federation.py:295-329`). | Unlike a worker failure, loss of the CP-resident supervisor loses the current A control owner. Durable artifacts may remain, but adoption/reconciliation is not implemented; no restart or retry claim is made. |
| What compatibility migration is real? | `in_process` is mapped to the named Python script. `external_process` is parsed to locate a Python script and arguments, then mapped to that same direct runtime; v9 restores omitted standard task-mode defaults before explicit overrides (`config.py:25-86`). The worker embeds `InProcessClientAPI` per task (`worker.py:125-178`). When `deployment_supervised=false`, the original launcher path is called unchanged (`federation.py:356-359`). | This is configuration-to-direct-script compatibility, not preservation of an external backend, live session, Attach, arbitrary shell command, or job-lifetime Python state. The old resident path remains the explicit compatibility bypass; it does not prove that those workloads migrated into A. |
| What must an operator deploy and observe? | Enable the launcher flag, deploy the existing job config/custom files, ensure a shared absolute run/state path and one-node Slurm plan, and retain supervisor events plus scheduler/job evidence (`federation.py:335-388`, `slurm.py:47-69`, `supervisor.py:200-206,239-282`). Current remote evidence observed one fresh allocation per logical task, settlement before publication, and clean immediate/+30 teardown. | A adds no standing worker/CJ, but it adds an SJ service and CP supervisor and creates one worker allocation plus durable input/result artifacts per task. Matched idle CPU/RSS, task-start latency, disk/serialization volume, and actual slow-upload evidence are still required for a cost claim. |

The strongest concrete reason to choose A is the enforced compute/publication
boundary: application code receives no federation credentials or Cell, commits
an eager verified result, and releases the worker allocation before the CP
publishes and waits for server ACK. That could materially simplify GPU
lifetime and contain application-worker failure for task-scoped workloads.

The strongest reason to reject A today is compatibility and control-path
duplication: its new service, supervisor, artifact protocol, and config adapter
cover a narrower surface than the existing CJ, while Attach, resident sessions,
Engine/Cell-dependent components, site privacy policy, CP recovery, retry, DDP,
and general streaming remain unsupported or unqualified. The v8 T4 default
translation defect is concrete evidence that this adapter surface carries
semantic migration risk.

The conclusion would change in A's favor if the authorized v9 full T4 row,
actual slow-upload row, corrected nonzero-exit row, and unchanged resident
bypass all satisfy their exact gates, and a matched A/B/D run shows material
task-boundary or standing-resource savings without new compatibility failures.
It would change against A if representative required workloads depend broadly
on the missing CJ surface, if failures recur at the translation/ownership
boundary, or if matched measurements show no operational saving large enough
to pay for the new control path. CP restart/retry, DDP, and general recovery
remain untested proposals either way.

### Final-result settlement versus phased D

The retained D common-T4 run did not lose server acceptance: the SJ accepted
and ACKed the result, and the transfer child then wrote a durable local
`TASK_COMPLETE` receipt after cleanup. The failure arose because CP supervision
still considered the physical handle unsettled; it reacted to the SJ becoming
done/unreachable and cancelled that handle before consuming the already-written
receipt. `TASK_COMPLETE` was a local receipt file, not a child RPC to a parent
endpoint.

A's meaningful structural difference is narrower. Its worker commits a local
artifact, exits, and fully settles before the CP supervisor directly issues the
single publication request to the SJ (`supervisor.py:284-321`,
`federation.py:258-273`). It has no separately scheduled transfer child and no
post-ACK physical-handle settlement/receipt-consumption phase. This removes the
specific D ownership edge that failed in the retained run; it does not mean A
has stronger server acceptance semantics.

The A endpoint is the `DeploymentTaskService` hosted in the server job process
and registered on that job's Cell (`federation.py:60-97`). Admission and
`ServerRunner` workflow teardown share `wf_lock`: result processing holds that
lock while checking `status/current_wf` and accepting the submission, while
finalization must acquire it before clearing `current_wf`
(`server_runner.py:164-176,450-476`). The service sends ACK only after the
communicator records `result_received_time` (`federation.py:175-193`). Thus the
Controller cannot count A's logical client result and finalize the workflow
before that direct server acceptance has occurred. D also accepted and ACKed
its result before the SJ ended, so this property is shared rather than an A
advantage. The differentiator is what client-side physical ownership remains
after that ACK.

A does **not** yet prove full terminal reconciliation. After acceptance, the
service removes its attempt record before the ACK reply is observed by the CP
(`federation.py:185-193`). If that response is lost during job/Cell shutdown,
`publish()` raises, a later failure publication can be rejected as stale, and
the logical handle still treats a subsequent server `END_RUN` as success
without querying accepted-attempt state (`federation.py:267-280,305-328`).
This is not D's observed post-ACK handle/receipt race, but current evidence does
not establish that it is less consequential. Successful prior A runs show the
normal ordering, not immunity to lost-ACK ambiguity; current tests do not close
it, and no recovery implementation or new proof is claimed here.

## Test evidence

Executed on macOS with Python 3.14 and CPU/local-process workers for the
current v9 candidate (historical deployed results retain their exact versions):

```text
tests/unit_test/private/fed/deployment_supervisor_test.py
34 passed

deployment supervisor + existing server/Slurm focused regressions
171 passed

exact phased-D reference 23faec2cf focused task-scope/Slurm suite
272 passed

examples/advanced/deployment-supervisor-a/demo.py
2 clients x 2 rounds completed; four expected results printed
```

The focused tests cover broadcast/send, multiple clients and rounds, fresh
Executor workers, explicit cross-task state, existing hello-numpy execution,
real in-process/external-process config translation, worker job filters, server
acceptance before ACK, filter order, publication callbacks,
settle-before-publish, timeout, cancellation/no later admission, descendant
cleanup, credential stripping, Slurm plan adaptation, artifact
tampering/stale identity, eager-only handoff, and resident preflight rejection.
The two additional bounded G06 tests use the same ordinary `Executor` source
prepared for A/B parity: `G06FaultExecutor.execute()` either raises the declared
application exception or exits with status 7. Both modes settle nonzero and
leave no result artifact or publication in the local A worker path. These local
results do not claim that the queued real-cluster rows have run.

The frozen v7 overlay is
`/private/tmp/architecture-a-v7-overlay.tar.gz`, SHA-256
`fce58251104ceadd9ff5de5ff98f4522f1ff2ae1253e18fa68871fd1e21335c0`,
applied to starting revision `9759e2594dc6949ef80cedf739c244334fae5f70`.
It adds bounded retry and blocking normalization for transient NFS `EAGAIN`
while opening committed artifacts, reads the terminal Slurm status from
`poll()` after `wait()`, and normalizes the Slurm manager's running/unknown
status for the supervisor contract.

The follow-on v8 overlay is
`/private/tmp/architecture-a-v8-overlay.tar.gz`, SHA-256
`ffbb417fb3534bdd08e97f2a97271b6e0b7513d9da24572cf75a69828582402f`.
Relative to immutable v7, only `config.py`, `supervisor.py`, and `worker.py`
change. The deployed job's absolute `custom/` directory is now carried in the
worker specification and added to the fresh worker's module search path before
loading ordinary Executors, task components, or filters. The deployment
supervisor still does not import job code. A regression launches a module that
is absent from the ambient `PYTHONPATH`, proving the deployed ordinary-Executor
path used by the shared A/B G06 fixture. This adapter gap was found after G01;
G01 remains exact evidence for v7 rather than being relabeled as a v8 run.

The real one-server/two-client/three-round in-process-source run used NVFlare
job `000eeefc-c557-4e35-90d8-d99e6f3907ac` and reached exact
`FINISHED:COMPLETED` in 44.762163 seconds. The server allocation and all six
client task allocations completed with scheduler return code 0. Each client
recorded three distinct acquired, launched, allocation-settled, and published
attempts. This proves the bounded real-federation row; it is not a matched
performance result. The server used Slurm allocation `11`; site-1 used
`12`, `15`, and `17`, while site-2 used `13`, `14`, and `16`. All completed
with scheduler result `0:0`.

Supported shutdown returned stopped/RC0. Immediate and 30-second audits on both
nodes found no owned NVFlare, job, or task-worker processes; Slurm allocations;
GPU compute processes; zombies or deleted working directories; relevant
listeners; or `CLOSE_WAIT` sockets. The sealed 81-file evidence bundle is
`/srv/nvflare-a-g01-shared/architecture-a-v7-g01-20260916/architecture-a-v7-g01-evidence.tar.gz`,
SHA-256 `3d7d5debf7232577e2ff01d4d06d305fde90bd6dba1f9d4325daa6a6da2f7029`;
its internal manifest SHA-256 is
`ad046848406e419bb4de04cd73da48809abbe89d918c3080cce842a008656214`.

This run used the in-process source mapping and synthetic hello-pt. Although
the client selected `cuda:0`, neither a contemporaneous worker device line nor
an in-task GPU sample was retained, so the runner's GPU preflight is not
credited as application CUDA execution. The external-process source mapping
with real CUDA work and the controlled slow-upload observation are recorded
separately when complete.

The later v7 external-source mapping/CUDA job
`d6397644-e42d-4978-9f24-2528e8965d6d` completed two clients and three rounds
with six fresh one-GPU task allocations, actual in-worker CUDA samples on both
participating GPUs, accepted results, supported shutdown, and clean immediate
and +30-second audits. This is a real T2b **configuration-to-direct-script**
mapping and T3 CUDA pass. It is not evidence that A ran or preserved the
original external trainer/backend/session topology. The retained operational
aggregate reports 44.787 seconds for the whole unmatched job and 43 GPU-seconds
of allocation; those descriptive values are not comparative performance data.

The v8 bounded exception job
`e824e0c5-4fcf-490f-8edd-6b84ed3a3719` passed T6a: both byte-identical ordinary
Executors reached `execute()`, raised the declared exception, settled their
Slurm allocations nonzero, created no result artifact or successful
publication, caused no successful round/job outcome, and passed supported
shutdown plus immediate/+30 cleanup. The attempted exit-7 job
`f005a9b1-e5fe-4dc4-a196-f9ab01c8f6dd` is not evidence: its exported component
used colon syntax and failed configuration before `execute()`. The corrected
dotted-path job `313da03d-dfd3-4e22-8293-1c38f8c31666` passed T6b: both
ordinary Executors reached the intended `os._exit(7)`, Slurm allocations 38
and 39 recorded `FAILED 7:0`, no successful result or publication occurred,
the job ended with bounded `EXECUTION_EXCEPTION`, and supported shutdown plus
immediate/+30 cleanup passed. Neither fault row should be rerun.

The v8 full common CSE job `56b97498-85cb-4979-a5a9-de1d6de39a7a`
**failed T4 qualification**. Both sites completed three train tasks and saved
distinct local models, but their `submit_model` and first `validate` workers
failed because the A configuration translator omitted the
`ClientAPIExecutor` default task-name declarations. Only five of eight tasks
per site were attempted. The server's `FINISHED:COMPLETED` terminal label is
existing `CrossSiteModelEval`/`ServerRunner` error-tolerance behavior and does
not override the missing-result gate. Cleanup was clean. Frozen v8 must not be
silently patched or retried. This was an A adapter-default defect, not evidence
that A's durable model-continuity requirement is inherently harder than B's;
both candidates must prove the same final-save/load/submit identity and
cross-site isolation contract.

The minimal v9 candidate restores the complete standard task
exchange defaults before applying explicit exported overrides. Its overlay is
`/private/tmp/architecture-a-v9-overlay.tar.gz`, SHA-256
`7a139f9fafa9a49ef62ee48c0bf75811c53124a34613cdb0ef0dc2198f9edec6`;
the only v8-to-v9 file change is `config.py`. An unconditional A-specific
server-panic policy is deliberately not included because it would override
controller semantics for optional tasks, quorum, and expected failures. The
bounded v9 T4 job `f1066f00-b85b-4ede-8004-3b8cbfb293ec` reached
`FINISHED:COMPLETED`: all emitted tasks completed and published, and each
site's final-save/load/submit digest matched while site digests remained
distinct. It emitted only 14 tasks, however—three validations per site rather
than four—so it is not a full T4 pass.

The absent identity was server best. The common client did return
`accuracy`, but A did not copy the input task cookie jar to the new result
Shareable as standard `ClientRunner._process_task` does. Consequently
`IntimeModelSelector` could not match `CONTRIBUTION_ROUND` to the current round,
did not fire `GLOBAL_BEST_MODEL_AVAILABLE`, and the persistor exposed only the
server-final checkpoint. This is another bounded A adapter defect, not a
change in the shared state contract. The v10 worker restores standard cookie
propagation after result filters. Its overlay is
`/private/tmp/architecture-a-v10-overlay.tar.gz`, SHA-256
`144fe2abbdce07cd97a62674276ddf0787191a56cc1b9588b8db669430cc3401`;
only `worker.py` differs from v9. The unchanged common workload is queued for
one separately identified v10 validation.

Remaining deployed rows have runnable handoffs rather than PASS status:

- T5 64 MiB actual-upload export and F3 settings:
  `/private/tmp/architecture-a-v8-t5-slow-upload-export.tar.gz`, SHA-256
  `81aefb4878db58d1af9c7ccf37a2f57242e1db42a47d8731934d71c2f02957dd`.
- T7a should reuse the portable standard resident hello-pt export, SHA-256
  `75a6f6d98d7de20f2d9fb8ee37e776b0953c0512da91a44f16f6cf6dbf69b059`,
  with `deployment_supervised=false`. A separately named hello-numpy fallback
  remains preserved and is not an additional scheduled row.

## Runnable entry points

- `examples/advanced/deployment-supervisor-a/demo.py`
- `examples/advanced/deployment-supervisor-a/config.json`
- `examples/advanced/deployment-supervisor-a/client.py`
- `tests/unit_test/private/fed/deployment_supervisor_test.py`

Run from the example directory:

```bash
PYTHONPATH=../../.. python demo.py --config config.json --workspace /tmp/nvflare-architecture-a
```
