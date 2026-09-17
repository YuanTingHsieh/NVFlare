# Deployment-supervised task runtime (Architecture A prototype)

This prototype makes a site-deployment supervisor, not a per-job Client Job
(CJ), own task acquisition, worker launch/cancellation, durable handoff, and
publication. A fresh worker runs either an ordinary `Executor` or an unchanged
Client API script. The worker receives only staged files and never owns a FLARE
Cell/session.

Run the ordinary Executor demonstration from this directory:

```bash
PYTHONPATH=../../.. python demo.py --config config.json --workspace /tmp/nvflare-architecture-a
```

The production server integration is deliberately not claimed complete. The
`ControllerTaskServiceAdapter` exercises broadcast/send scheduling semantics,
attempt validation, and publication, but existing Controllers/WFCommServer are
not yet wired to this deployment-owned service. The local process launcher is a
working backend; Slurm can reuse the same `JobHandleSpec` settlement contract,
but that adapter and real GPU/allocation qualification remain future work.

Fresh task workers cannot preserve arbitrary Python object state. Applications
may declare a `state_id` and use `NVFLARE_TASK_STATE_DIR` (scripts) or the
`deployment_task_state_dir` context property (Executors) for application-specific
state. Components requiring a live Cell, auxiliary tasks, Attach, interactive
sessions, or job-lifetime in-memory callbacks must keep using the unchanged
resident CJ path until an explicit adapter exists.
