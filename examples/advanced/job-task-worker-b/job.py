# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Build or simulate the bounded Architecture B hello-pt candidate."""

import argparse
import inspect
import os
import shlex
import sys

from components import SlowResultFilter, SlowUploadVerificationFilter

from nvflare import FedJob
from nvflare.apis.dxo import DataKind
from nvflare.app_common.aggregators.intime_accumulate_model_aggregator import InTimeAccumulateWeightedAggregator
from nvflare.app_common.executors.client_api_executor import ClientAPIExecutor
from nvflare.app_common.shareablegenerators.full_model_shareable_generator import FullModelShareableGenerator
from nvflare.app_common.widgets.intime_model_selector import IntimeModelSelector
from nvflare.app_common.widgets.validation_json_generator import ValidationJsonGenerator
from nvflare.app_common.workflows.cross_site_model_eval import CrossSiteModelEval
from nvflare.app_common.workflows.scatter_and_gather import ScatterAndGather
from nvflare.app_opt.pt.file_model_locator import PTFileModelLocator
from nvflare.app_opt.pt.file_model_persistor import PTFileModelPersistor
from nvflare.client.config import ExchangeFormat, TransferType
from nvflare.job_config.defs import FilterType
from nvflare.private.fed.job_task_worker.executor import ClientAPIJobTaskWorkerExecutor, JobTaskWorkerExecutor
from nvflare.private.fed.job_task_worker.slurm import SlurmTaskWorkerLauncher


def _sources(workload_dir):
    if workload_dir:
        hello_pt = os.path.abspath(workload_dir)
    else:
        directory = os.path.dirname(os.path.abspath(__file__))
        hello_pt = os.path.abspath(os.path.join(directory, "..", "..", "hello-world", "hello-pt"))
    if not os.path.isfile(os.path.join(hello_pt, "client.py")) or not os.path.isfile(
        os.path.join(hello_pt, "model.py")
    ):
        raise ValueError("workload directory must contain client.py and model.py")
    sys.path.insert(0, hello_pt)
    from model import SimpleNetwork

    fault_executor = os.path.join(os.path.dirname(os.path.abspath(__file__)), "g06_fault_executor.py")
    return SimpleNetwork, os.path.join(hello_pt, "client.py"), os.path.join(hello_pt, "model.py"), fault_executor


def _slurm_launcher(args, site):
    executables = {name: os.path.join(args.slurm_bin, name) for name in ("sbatch", "squeue", "sacct", "scancel")}
    worker_resources = {
        "nodes": 1,
        "cpus_per_node": args.cpus_per_node,
        "mem_per_node": args.memory_gib * 1024,
        "time": args.time_limit,
    }
    if args.gpus_per_node:
        worker_resources["gpus_per_node"] = args.gpus_per_node
    return SlurmTaskWorkerLauncher(
        workspace_path=args.site_workspace.format(site=site),
        sandbox=args.sandbox,
        image=args.image,
        python_path=args.python_path,
        executables=executables,
        sbatch_directives={"partition": args.partition} if args.partition else {},
        worker_resources=worker_resources,
        poll_interval=args.slurm_poll_interval,
        pending_timeout=args.pending_timeout,
        task_environment={"PYTHONHASHSEED": str(args.seed)},
    )


def build_job(args):
    simple_network, client_script, model_source, fault_executor = _sources(args.workload_dir)
    if args.transport_payload_mib:
        name = "architecture-b-g05-slow-upload"
    elif args.failure_mode:
        name = f"architecture-b-g06-{args.failure_mode}"
    elif args.legacy_resident:
        name = "architecture-b-g07-legacy-resident"
    else:
        name = f"architecture-b-hello-pt-{args.dataset}-{args.mode}"
    job = FedJob(name=name, min_clients=args.clients)
    shareable_generator_id = job.to_server(FullModelShareableGenerator(), id="shareable_generator")
    model_args = {"seed": args.seed} if "seed" in inspect.signature(simple_network).parameters else {}
    persistor_id = job.to_server(PTFileModelPersistor(model=simple_network(**model_args)), id="persistor")
    aggregator_id = job.to_server(
        InTimeAccumulateWeightedAggregator(expected_data_kind=DataKind.WEIGHTS), id="aggregator"
    )
    job.to_server(
        ScatterAndGather(
            min_clients=args.clients,
            num_rounds=args.rounds,
            start_round=0,
            wait_time_after_min_received=0,
            aggregator_id=aggregator_id,
            persistor_id=persistor_id,
            shareable_generator_id=shareable_generator_id,
            train_task_name="train",
            train_timeout=600,
            ignore_result_error=False,
            task_check_period=0.5,
            persist_every_n_rounds=1,
            snapshot_every_n_rounds=1,
            memory_gc_rounds=1,
            enable_tensor_disk_offload=False,
        )
    )
    job.to_server(IntimeModelSelector(key_metric="accuracy"))
    locator_id = job.to_server(PTFileModelLocator(pt_persistor_id=persistor_id), id="model_locator")
    if not args.failure_mode and not args.train_only:
        job.to_server(
            CrossSiteModelEval(
                model_locator_id=locator_id,
                submit_model_timeout=600,
                validation_timeout=600,
                cleanup_models=False,
                validation_task_name="validate",
                submit_model_task_name="submit_model",
                participating_clients=[f"site-{index + 1}" for index in range(args.clients)],
                wait_for_clients_timeout=300,
            )
        )
    job.to_server(ValidationJsonGenerator())
    if args.transport_payload_mib:
        job.to_server(
            SlowUploadVerificationFilter(args.transport_payload_mib),
            filter_type=FilterType.TASK_RESULT,
            tasks=["train"],
        )

    common_workload = bool(args.workload_dir)
    batch_size = args.batch_size
    if batch_size is None:
        batch_size = 8 if common_workload and args.dataset == "synthetic" else 64 if common_workload else 16
    epochs = args.epochs if args.epochs is not None else 1 if common_workload else 2
    num_workers = args.num_workers
    if num_workers is None:
        num_workers = 0 if common_workload and args.dataset == "synthetic" else 2
    train_size = args.train_size if args.train_size is not None else 32 if common_workload else 50000
    test_size = args.test_size if args.test_size is not None else 16 if common_workload else 10000
    for index in range(args.clients):
        site = f"site-{index + 1}"
        launcher_id = ""
        if args.slurm:
            launcher_id = "architecture_b_slurm"
            job.to(_slurm_launcher(args, site), target=site, id=launcher_id)
        if args.failure_mode:
            job.to(fault_executor, target=site)
            job.to(
                JobTaskWorkerExecutor(
                    application_path="g06_fault_executor:G06FaultExecutor",
                    application_args={"mode": args.failure_mode},
                    launcher_id=launcher_id,
                    worker_timeout=args.worker_timeout,
                    worker_environment={"PYTHONHASHSEED": str(args.seed)},
                ),
                target=site,
                tasks=["train"],
            )
            continue
        common = {
            "execution_mode": args.mode,
            "launcher_id": launcher_id,
            "resources": [model_source],
            "script_resource": client_script,
            "params_exchange_format": ExchangeFormat.PYTORCH,
            "server_expected_format": ExchangeFormat.NUMPY,
            "params_transfer_type": TransferType.FULL,
            "worker_timeout": args.worker_timeout,
            "task_wait_timeout": 600.0,
            "result_wait_timeout": 600.0,
            "launch_once": False,
            "worker_environment": {"PYTHONHASHSEED": str(args.seed)},
        }
        if common_workload:
            task_script = client_script
            train_args = (
                f"--batch-size {batch_size} --epochs {epochs} --num-workers {num_workers} "
                f"--seed {args.seed} --site-seed-offset {index + 1} "
                f"--dataset-root {shlex.quote(args.data_root)}"
            )
            if args.dataset == "synthetic":
                train_args += f" --synthetic-data --train-size {train_size} --test-size {test_size}"
        else:
            task_script = client_script
            train_args = f"--batch_size {batch_size} --epochs {epochs} --num_workers {num_workers}"
            if args.dataset == "synthetic":
                train_args += f" --synthetic_data --train_size {train_size} --test_size {test_size}"
            else:
                train_args += f" --data_root {shlex.quote(args.data_root)}"
        common["script_resource"] = task_script
        if args.transport_payload_mib:
            train_args += f" --transport_payload_mib {args.transport_payload_mib}"
        if args.legacy_resident:
            # Keep the serialized entrypoint relative to the deployed custom/
            # directory.  ScriptRunner preserves an absolute in-process source
            # path in config, which is valid only on the exporting machine.
            job.to(task_script, target=site)
            executor = ClientAPIExecutor(
                execution_mode="in_process",
                task_script_path=os.path.basename(task_script),
                task_script_args=train_args,
                params_exchange_format=ExchangeFormat.PYTORCH,
                server_expected_format=ExchangeFormat.NUMPY,
                params_transfer_type=TransferType.FULL,
            )
        elif args.mode == "in_process":
            executor = ClientAPIJobTaskWorkerExecutor(
                task_script_path=task_script,
                task_script_args=train_args,
                **common,
            )
        else:
            executor = ClientAPIJobTaskWorkerExecutor(
                command=["python3", "-u", f"custom/{os.path.basename(task_script)}", *shlex.split(train_args)],
                launch_timeout=300.0,
                shutdown_timeout=30.0,
                stop_grace_period=30.0,
                heartbeat_interval=5.0,
                heartbeat_timeout=30.0,
                **common,
            )
        tasks = ["train"] if args.train_only else ["train", "validate", "submit_model"]
        job.to(executor, target=site, tasks=tasks)
        if args.publication_delay:
            job.to(
                SlowResultFilter(args.publication_delay),
                target=site,
                filter_type=FilterType.TASK_RESULT,
                tasks=["train", "validate", "submit_model"],
            )
    return job


def define_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("in_process", "external_process"), default="in_process")
    parser.add_argument("--clients", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--num-workers", "--num_workers", dest="num_workers", type=int)
    parser.add_argument("--train-size", "--train_size", dest="train_size", type=int)
    parser.add_argument("--test-size", "--test_size", dest="test_size", type=int)
    parser.add_argument("--dataset", choices=("synthetic", "cifar10"), default="synthetic")
    parser.add_argument("--data-root", default="/tmp/nvflare/data")
    parser.add_argument("--workload-dir", default="")
    parser.add_argument("--seed", type=int, default=202610)
    parser.add_argument("--publication-delay", type=float, default=0.0)
    parser.add_argument("--failure-mode", choices=("exception", "nonzero"))
    parser.add_argument("--legacy-resident", action="store_true")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--transport-payload-mib", type=int, default=0)
    parser.add_argument("--worker-timeout", type=float, default=600.0)
    parser.add_argument("--export")
    parser.add_argument("--simulate")
    parser.add_argument("--slurm", action="store_true")
    parser.add_argument("--site-workspace", default="", help="Absolute template containing {site}.")
    parser.add_argument("--sandbox", choices=("none", "apptainer", "enroot"), default="none")
    parser.add_argument("--image")
    parser.add_argument("--python-path", default=sys.executable)
    parser.add_argument("--slurm-bin", default="/usr/bin")
    parser.add_argument("--partition", default="")
    parser.add_argument("--gpus-per-node", type=int, default=1)
    parser.add_argument("--cpus-per-node", type=int, default=4)
    parser.add_argument("--memory-gib", type=int, default=16)
    parser.add_argument("--time-limit", default="00:10:00")
    parser.add_argument("--slurm-poll-interval", type=float, default=1.0)
    parser.add_argument("--pending-timeout", type=float, default=300.0)
    return parser


def main():
    args = define_parser().parse_args()
    if args.failure_mode and args.legacy_resident:
        raise ValueError("--failure-mode and --legacy-resident are mutually exclusive")
    if args.transport_payload_mib < 0:
        raise ValueError("--transport-payload-mib must not be negative")
    if args.transport_payload_mib and not args.train_only:
        raise ValueError("--transport-payload-mib requires --train-only")
    if args.transport_payload_mib and (args.failure_mode or args.legacy_resident or args.publication_delay):
        raise ValueError("the isolated G05 transport fixture cannot be combined with other gate fixtures")
    if args.legacy_resident and args.slurm:
        raise ValueError("the legacy resident regression must be reported separately from the B Slurm worker path")
    if args.slurm and (not args.site_workspace or "{site}" not in args.site_workspace):
        raise ValueError("--slurm requires an absolute --site-workspace template containing {site}")
    if bool(args.export) == bool(args.simulate):
        raise ValueError("choose exactly one of --export or --simulate")
    job = build_job(args)
    if args.export:
        job.export_job(args.export)
    else:
        job.simulator_run(args.simulate, gpu="0")


if __name__ == "__main__":
    main()
