# Experimental Slurm: whole CJ per task (Arch D, N=1)

This is an architecture experiment, **not a supported launcher mode**. The default
Slurm launcher is unchanged. This version uses the existing task/result transport;
it does not implement a new durable server commit protocol.

## What is implemented

1. A logical job handle lives in the existing CP. It remains registered for job
   heartbeat, cancellation, and final outcome accounting while no CJ exists.
2. A configured SJ component answers an authenticated, **nonassigning** availability
   probe. It supports the built-in ordinary broadcast/send scheduler only.
3. When work is available, CP submits a Slurm allocation running the entire CJ:
   config construction, Cell, ClientRunner, filters, and Executor.
4. The CJ processes at most one ordinary synchronous task. Only a successful
   result-submission ACK and successful worker cleanup produce an attempt-specific
   receipt. The parent also requires a successful Slurm result.
5. CP waits for Slurm terminal accounting/cleanup before permitting another CJ.
   It holds no Slurm allocation between attempts. A stale readiness token that
   produced TRY_AGAIN does not repeatedly trigger more allocations.
6. SJ explicitly notifies CP when scheduling ends. A missing SJ endpoint without
   that notice is a bounded communication failure, never inferred success.

Slurm's queue/accounting/cancellation behavior is inherited. In particular, an
accounting outage can delay allocation settlement; the new communication timeout
does not impose a new universal deadline on Slurm itself.

The counter example is deliberately small: it requests a GPU allocation but does
not run CUDA kernels. It demonstrates allocation lifetime and file-backed state
continuity, not training throughput, GPU utilization, or model correctness.

## Code ownership

Task-scoped execution is runtime behavior, not a Slurm feature. The experiment
separates the shared internal implementation from its first launcher adapter:

| Location | Responsibility |
|---|---|
| `nvflare/private/fed/task_scope/launcher.py` | CP logical participation, readiness probes, terminal notifications, attempt/receipt lifecycle |
| `nvflare/private/fed/task_scope/runner.py` and `worker.py` | One-task CJ execution and post-cleanup receipt |
| `nvflare/private/fed/task_scope/server.py` and `protocol.py` | SJ readiness/terminal handling and shared protocol |
| Existing `nvflare/app_opt/job_launcher/slurm/launcher.py` | Physical allocation submission, tracking, accounting and cancellation; task scope only selects the shared runtime wrapper |

The private implementation does not import Slurm. An adapter must provide a
physical handle whose `wait()` includes allocation settlement/cleanup, plus a
workspace that survives worker exit. Moving code does **not** establish support
for another launcher; only the Slurm adapter is implemented in this experiment.

## Job participation must outlive individual CJs

In the ordinary workflow path, SP compares the job IDs reported by CP with the
expected participants and forwards missing-job reports to SJ. SJ's dead-client
monitor ages those reports; it does not simply count currently running CJs.

The experimental logical handle stays in CP's job registry while its physical
Slurm allocation is absent. CP therefore continues reporting participation during
idle gaps and queueing. Individual CJ exits do not finish that logical handle
unless the attempt fails or the logical job ends. This uses no synthetic CJ
heartbeat and does not disable genuine participant-loss detection.

| Situation | Intended interpretation |
|---|---|
| CP owns the logical job; all CJs intentionally absent between tasks | Participants remain registered; job is not dead |
| CP owns the logical job; next allocation is pending | Participant remains registered; task deadlines still apply |
| Active CJ fails | Failed attempt; this experiment terminates logical participation without automatic retry |
| CP loses its job registration, or the site is declared disconnected | Existing missing-participant detection and job policy still apply |
| SJ ends scheduling while a CJ is finishing | Parent waits for worker cleanup, receipt and Slurm settlement before reporting its outcome |

This distinction covers the ordinary workflow dead-client monitor, not every
application's liveness rules. CCWF, TIE and XGBoost controllers have independent
status/progress timers. Keeping CP's job registration does not satisfy those
timers or clear a previously issued dead-client report. Those workflows remain
outside this experiment's scope.

## Run on a real Slurm deployment

Use a dedicated test workspace. Follow the existing
[Slurm deployment guide](../../../docs/user_guide/admin_guide/deployment/slurm_job_launcher.rst)
to prepare one server and two clients, with a shared workspace per client visible
at the same absolute path on CP and compute nodes. All parents and worker Python
environments/images must contain this prototype revision.

Run CP in a CPU-only service/allocation. Recycling CJs cannot release GPUs that
the deployment independently reserves for its long-lived parent.

Before starting each CP, keep the existing
`nvflare.app_opt.job_launcher.slurm.ClientSlurmJobLauncher` component in its
prepared `local/resources.json` and add `"task_scoped": true` to that component's
arguments. Preserve all other site-specific arguments, mounts, scheduler commands,
account, partition, Python path, and resource-manager configuration. The existing
launcher still owns every physical allocation. Leave the server launcher unchanged.

Optional extra experimental launcher arguments:

```json
{
  "task_scoped": true,
  "task_probe_interval": 2.0,
  "task_probe_timeout": 5.0,
  "task_communication_timeout": 120.0
}
```

From this example directory, export the job:

```bash
python job.py --output /absolute/test-job-exports --clients site-1 site-2 --rounds 3 --gap-seconds 30 --gpus 1
```

Submit the exported `slurm-task-scope` directory through the normal NVFlare admin
job workflow. The exported server app already includes `TaskScopedServer` from
`nvflare.private.fed.task_scope.server`. Re-export jobs created before this
prototype's module relocation; the old experimental paths are not aliases.
The controller waits for every client's result, then deliberately schedules no
work for the configured gap (including after the final round).

Use a long enough gap to observe allocation exit after CJ teardown. Per-task
timeout includes Slurm queuing and CJ startup; increase it in `GapController`
for a busy cluster. A readiness probe does not reserve the advertised task.

For a crash test, export a **new job** with `--crash-round 1`. Each client exits
with code 1 before checkpointing round 1. Do not reuse the first job's checkpoint
directory. For a CPU-only scheduler smoke test, use `--gpus 0`; that is not a GPU
resource-release test.

## Evidence to collect

Per client, `<workspace>/<job_id>/.task_scope/events.jsonl` records:

- `waiting_for_work`, `submitting`, scheduler ID returned, `allocation_released`,
  receipt, and `logical_job_finished`;
- wall-clock and monotonic timestamps;
- unique attempt IDs and Slurm IDs, without model or credential bytes.

Here `allocated` means sbatch returned an ID; it does **not** prove the allocation
has left PENDING. Slurm accounting is required to measure actual resource use.

Each attempt has its own receipt directory. The example also produces:

- Client: `task_scope_counter.json`, containing restored round/value, PID and
  Slurm job ID.
- Server: `task_scope_results.json`, containing each site's results per round.

For the relevant scheduler IDs, collect:

```bash
squeue --jobs=12345,12346,12347 --format="%.18i %.12T %.30j %.30b"
sacct --jobs=12345,12346,12347 --format=JobIDRaw,State,ExitCode,Start,End,Elapsed,AllocTRES -P
```

Replace the example IDs with those recorded by the experiment. Observe squeue
during a gap, not only after completion. Acceptance evidence should show:

| Case | Required evidence |
|---|---|
| Three successful tasks | Three distinct scheduler IDs; values 1, 2, 3 restored across fresh CJs; server job completes |
| Idle gap | Previous allocation terminal; no new allocation until another task is advertised |
| All CJs absent beyond the configured dead-client grace | CPs continue reporting the logical job; SJ remains active and the next task succeeds |
| Long upload/cleanup | GPU allocation remains until the **whole CJ** exits; this baseline does not release at local save |
| Worker exit 1 / SIGKILL | No successful receipt accepted; no automatic retry; bounded job failure under full-client policy |
| Abort while idle | No new sbatch; logical parent handle terminates |
| Abort while pending/running | Existing Slurm cancellation and accounting settle the owned allocation |
| Missing terminal notice | No invented success; communication timeout or existing server cleanup resolves participation |
| Two concurrent jobs | Separate CP handles; no per-job overlap; verify scheduler/resource-manager admission |

## Impact demonstrated by the implementation

| Area | Prototype change or semantic impact |
|---|---|
| CP | Shared private runtime keeps logical participation and supervises sequential physical allocations; CP is not merely relaunching on exit |
| SJ | A new authenticated readiness operation and parent-targeted terminal notification; ordinary GET_TASK cannot serve as a pure probe |
| CJ | An alternate runner/entrypoint processes one task and records a post-cleanup receipt |
| Existing runtime seams | Defaulted app-runner injection in `worker_process.main`; overridable runner class in `ClientAppRunner`; defaults unchanged |
| Launcher | Existing `ClientSlurmJobLauncher` supplies physical handles to the shared private runtime when `task_scoped=true`; its scheduler machinery is unchanged |
| Application state | Executor instances, handlers and filters are reconstructed each incarnation; the example explicitly checkpoints its state |
| Lifecycle events | START_RUN, ABOUT_TO_END_RUN and END_RUN run per incarnation, not once per logical job |
| Filters/security | Normal task filters remain in CJ alongside application code; this does not provide A/B-style isolation |
| Result delivery | Existing eager result submission, not an atomic accepted-result/state commit; publication can still be semantically rejected after a transport ACK |
| Workspace/logs | Shared workspace survives, but ordinary per-process archival/log behavior remains; complete final-log publication is not newly guaranteed |
| Operator status | Existing parent status words may display STOPPED during idle or STARTING during admission; use the attempt journal/Slurm for this experiment |

## Explicit limitations

- Single-node Slurm allocations only; no multi-node fan-out, DDP, or framework
  collective validation in this first experiment.
- Every Executor must explicitly declare `supports_task_scoped_process = True`.
  This is an author assertion, not an automatic proof. All job components must
  tolerate repeated initialization/cleanup and use explicit cross-task state.
- No unchanged stateful legacy Executors, XGBoost histogram, Flower, split
  learning, CCWF/aux tasks, or Attach/background trainer sessions.
- Eager ordinary task results only. Pass-through/lazy result sources are rejected
  because the CJ is going away. No large-model streaming validation yet.
- No crash recovery/adoption after CP restart, persistent coordinator ledger,
  automatic task retries, or exactly-once external side effects. The example's
  checkpoint/replay logic is not a general transaction framework.
- No automatic resource renegotiation for task-specific requirements: each
  incarnation reuses the job's Slurm resource request. Framework-side admission
  remains job-scoped; Slurm owns the physical GPU allocation release.
- No claim that unit tests prove real Slurm resource release or federation-wide
  finalization correctness. Cluster testing is still required.

## The proposed pull / compute / push variant

This baseline is **one complete CJ per ordinary task**. It does not implement
three separate CJs for pulling inputs, computation, and pushing results.

That proposed variant would add explicit phase dispatch, durable inter-phase
artifact references, phase-specific allocations, and cancellation/recovery across
the phases. Its compute allocation could end before network publication, unlike
this baseline. If all phase CJs retain today's full credentials/runtime access,
it is a phased form of D; task scheduling alone does not introduce A/B's trusted
supervisor versus restricted application-worker boundary.

## Local validation

Shared-runtime tests live in `tests/unit_test/private/fed/task_scope/`; adapter and
example tests remain in `tests/unit_test/app_opt/job_launcher/slurm_task_scope_*_test.py`.
They exercise runner behavior, receipt ordering, pure probes, cancellation and
readiness races using test doubles. Existing Slurm tests are also run for regression
coverage. The example exporter can run without Slurm. Real `sbatch` execution,
Cell-connected end-to-end restart, GPU release, and container cases must be measured
on the cluster; they are not implied by these tests passing.

Validation performed on 2026-09-10:

| Check | Result |
|---|---|
| New server/launcher/receipt/runner/worker tests | Passed locally with mocked runtime/scheduler boundaries |
| CP job list → SP missing-job sync → SJ dead-client policy | Five local tests passed using real methods with mocked transport/clock: all CJs absent stays alive; missing registrations and real parent-loss reports still fail |
| Counter in three fresh Python processes | Values 1 → 2 → 3 restored; distinct PIDs |
| Replayed counter operation | Does not increment twice |
| Counter `os._exit(1)` injection | Process exits 1 without advancing its checkpoint |
| Exported job | Custom components bundled; nondefault controller/executor parameters preserved |
| Combined prototype and launcher/client/server lifecycle regression suite | 966 passed locally after rebasing onto current `main`; see command below |
| Dependency boundary | Private task-scope modules have no optional-launcher imports; importing them with Slurm imports blocked succeeds |
| Scoped Black/isort/flake8 and diff whitespace checks | Passed |
| Actual Slurm, GPU release, full CP↔SJ↔CJ transport | **Not run: Slurm cluster access required** |

Combined regression command from the repository root:

```bash
python -m pytest -q tests/unit_test/app_opt/job_launcher \
  tests/unit_test/private/fed/task_scope \
  tests/unit_test/private/fed/client/client_runner_test.py \
  tests/unit_test/private/fed/client/client_executor_test.py \
  tests/unit_test/private/fed/app/job_process_cleanup_test.py \
  tests/unit_test/private/fed/server/fed_server_test.py \
  tests/unit_test/apis/impl/wf_comm_server_test.py
```
