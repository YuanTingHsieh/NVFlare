#!/usr/bin/env python3
"""Memory profiling for FedAvg jobs using Recipe API and PocEnv."""

import os
import shutil
import signal
import threading
import time
from typing import Tuple

import psutil
from large_model import GigabyteModel

from nvflare.apis.dxo import DataKind
from nvflare.app_common.aggregators.intime_accumulate_model_aggregator import InTimeAccumulateWeightedAggregator
from nvflare.app_common.shareablegenerators.full_model_shareable_generator import FullModelShareableGenerator
from nvflare.app_common.workflows.fedavg import FedAvg
from nvflare.app_common.workflows.scatter_and_gather import ScatterAndGather
from nvflare.app_opt.pt.file_model_persistor import PTFileModelPersistor
from nvflare.app_opt.pt.job_config.model import PTModel
from nvflare.app_opt.tensor_stream.client import TensorClientStreamer
from nvflare.app_opt.tensor_stream.server import TensorServerStreamer
from nvflare.client.config import ExchangeFormat
from nvflare.job_config.api import FedJob
from nvflare.job_config.script_runner import ScriptRunner
from nvflare.recipe.poc_env import PocEnv
from nvflare.recipe.spec import Recipe


def get_model_size_gb(model) -> float:
    """Calculate model size in GB."""
    total_params = sum(p.numel() for p in model.parameters())
    # Assuming float32 (4 bytes per parameter)
    size_bytes = total_params * 4
    size_gb = size_bytes / (1024**3)
    return size_gb


def get_process_memory(pid: int) -> int:
    """Get memory usage of a single process (NOT including children) in MB.

    Note: We don't include children here because we iterate over all processes
    separately in get_server_client_memory(), so including children would
    result in double-counting.
    """
    try:
        process = psutil.Process(pid)
        mem_info = process.memory_info()
        return mem_info.rss // (1024 * 1024)  # Convert to MB
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return 0


def get_server_client_memory(poc_workspace: str = None, debug: bool = False) -> Tuple[int, int, int, int, int, int]:
    """Get memory usage of server and client processes separately.

    Returns: (server_parent_mem, server_job_mem, client_parent_mem, client_job_mem, server_total, client_total)
    """
    server_parent_mem = 0
    server_job_mem = 0
    client_parent_mem = 0
    client_job_mem = 0
    found_processes = []

    try:
        for proc in psutil.process_iter(["pid", "name", "cmdline", "cwd"]):
            try:
                cmdline = proc.info.get("cmdline", [])
                if not cmdline:
                    continue

                cmdline_str = " ".join(cmdline)
                cwd = proc.info.get("cwd", "") or ""  # Handle None case

                # Look for nvflare processes - be more specific
                is_poc_process = False

                # Priority 1: Check for the actual job process modules
                if (
                    "nvflare.private.fed.app.server.runner_process" in cmdline_str
                    or "nvflare.private.fed.app.client.worker_process" in cmdline_str
                ):
                    is_poc_process = True
                # Priority 2: Check if in POC workspace
                elif poc_workspace and cwd and poc_workspace in cwd:
                    is_poc_process = True
                # Priority 3: Check for startup directory (POC processes run from startup/)
                elif cwd and "/startup" in cwd:
                    is_poc_process = True

                if not is_poc_process:
                    continue

                found_processes.append((proc.pid, cmdline_str[:80], cwd))

                if debug:
                    print(f"Found POC process (PID {proc.pid}): {cmdline_str[:150]}")
                    print(f"  CWD: {cwd}")

                # Identify the actual job processes based on how they're started in nvflare.private
                # From server_engine.py line 266: python -m nvflare.private.fed.app.server.runner_process
                # From client_executor.py line 202: python -m nvflare.private.fed.app.client.worker_process
                is_server_job = "-m" in cmdline and "nvflare.private.fed.app.server.runner_process" in cmdline_str
                is_client_job = "-m" in cmdline and "nvflare.private.fed.app.client.worker_process" in cmdline_str

                # Parent processes run from startup directories
                # Server parent: runs from /server/startup and starts server_train
                # Client parent: runs from /site-N/startup and starts client_train
                is_server_parent = (
                    ("server" in cwd and "startup" in cwd) or "server_train" in cmdline_str
                ) and not is_server_job
                is_client_parent = (
                    ("site-" in cwd and "startup" in cwd) or "client_train" in cmdline_str
                ) and not is_client_job

                mem = get_process_memory(proc.pid)

                # Categorize into 4 buckets
                if is_server_job:
                    server_job_mem += mem
                    if debug:
                        print(f"  -> Server JOB: {mem} MB")
                elif is_server_parent:
                    server_parent_mem += mem
                    if debug:
                        print(f"  -> Server PARENT: {mem} MB")
                elif is_client_job:
                    client_job_mem += mem
                    if debug:
                        print(f"  -> Client JOB: {mem} MB")
                elif is_client_parent:
                    client_parent_mem += mem
                    if debug:
                        print(f"  -> Client PARENT: {mem} MB")
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
            except Exception as e:
                # Catch any other unexpected errors for individual processes
                if debug:
                    print(f"  Warning: Error accessing process: {e}")
                continue

    except Exception as e:
        # If psutil.process_iter itself fails or other critical errors
        if debug:
            print(f"  Warning: Error iterating processes: {e}")
        # Return zeros - no data available
        return 0, 0, 0, 0, 0, 0

    server_total = server_parent_mem + server_job_mem
    client_total = client_parent_mem + client_job_mem

    if debug:
        if not found_processes:
            print("  WARNING: No POC processes found!")
            if poc_workspace:
                print(f"  POC workspace: {poc_workspace}")
        else:
            print(f"  Found {len(found_processes)} POC processes")
            print(f"  Server: {server_total} MB (Parent: {server_parent_mem} MB, Job: {server_job_mem} MB)")
            print(f"  Client: {client_total} MB (Parent: {client_parent_mem} MB, Job: {client_job_mem} MB)")

    return server_parent_mem, server_job_mem, client_parent_mem, client_job_mem, server_total, client_total


class MemoryMonitor:
    """Monitor memory usage during job execution."""

    def __init__(self, output_file: str, poc_workspace: str = None, debug: bool = False):
        self.output_file = output_file
        self.poc_workspace = poc_workspace
        self.debug = debug
        self.peak_total = 0
        self.peak_server_parent = 0
        self.peak_server_job = 0
        self.peak_client_parent = 0
        self.peak_client_job = 0
        self.peak_server_total = 0
        self.peak_client_total = 0
        self.peak_time = 0
        self.start_time = None
        self.stop_flag = threading.Event()
        self.thread = None
        self.debug_printed = False

    def _monitor_loop(self):
        """Monitoring loop that runs in a separate thread."""
        with open(self.output_file, "w") as f:
            f.write(
                "Time(s) Total(MB) ServerParent(MB) ServerJob(MB) ClientParent(MB) ClientJob(MB) ServerTotal(MB) ClientTotal(MB)\n"
            )

            while not self.stop_flag.is_set():
                try:
                    elapsed = time.time() - self.start_time

                    # Print debug info only once after processes have had time to spawn
                    debug_now = self.debug and not self.debug_printed and elapsed > 20
                    if debug_now:
                        print("\n=== DEBUG: Process Detection (at ~20s) ===")
                        self.debug_printed = True

                    server_parent, server_job, client_parent, client_job, server_total, client_total = (
                        get_server_client_memory(self.poc_workspace, debug=debug_now)
                    )
                    total_mem = server_total + client_total

                    if debug_now:
                        print(f"Total: {total_mem} MB")
                        print("=================================\n")

                    # Track peaks
                    if total_mem > self.peak_total:
                        self.peak_total = total_mem
                        self.peak_server_parent = server_parent
                        self.peak_server_job = server_job
                        self.peak_client_parent = client_parent
                        self.peak_client_job = client_job
                        self.peak_server_total = server_total
                        self.peak_client_total = client_total
                        self.peak_time = elapsed

                    # Write to file
                    f.write(
                        f"{elapsed:.1f} {total_mem} {server_parent} {server_job} {client_parent} {client_job} {server_total} {client_total}\n"
                    )
                    f.flush()

                except (psutil.NoSuchProcess, psutil.AccessDenied, ProcessLookupError) as e:
                    # Processes may have exited - this is normal at job completion
                    if self.debug:
                        print(f"[Monitor] Process access error (processes may have exited): {e}")
                    # Write zeros to indicate no processes found
                    elapsed = time.time() - self.start_time
                    f.write(f"{elapsed:.1f} 0 0 0 0 0 0 0\n")
                    f.flush()
                except Exception as e:
                    # Unexpected error - log but continue monitoring
                    print(f"[Monitor] Unexpected error: {e}")
                    elapsed = time.time() - self.start_time
                    f.write(f"{elapsed:.1f} 0 0 0 0 0 0 0\n")
                    f.flush()

                # Sleep for 200ms
                self.stop_flag.wait(0.2)

    def start(self):
        """Start monitoring in a background thread."""
        self.start_time = time.time()
        self.stop_flag.clear()
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop monitoring."""
        self.stop_flag.set()
        if self.thread:
            self.thread.join(timeout=2)

        print(f"\nPeak memory: {self.peak_total} MB (at {self.peak_time:.0f}s)")
        print(
            f"  Server: {self.peak_server_total} MB (Parent: {self.peak_server_parent} MB, Job: {self.peak_server_job} MB)"
        )
        print(
            f"  Client: {self.peak_client_total} MB (Parent: {self.peak_client_parent} MB, Job: {self.peak_client_job} MB)"
        )

        # If in debug mode, show detailed process classification at the current time
        if self.debug:
            print("\n=== DEBUG: Process Detection (current state) ===")
            get_server_client_memory(self.poc_workspace, debug=True)
            print("=================================\n")

        return (
            self.peak_total,
            self.peak_server_parent,
            self.peak_server_job,
            self.peak_client_parent,
            self.peak_client_job,
            self.peak_server_total,
            self.peak_client_total,
        )


class CustomFedAvgRecipe(Recipe):
    """Custom FedAvg Recipe using InTime aggregation (already memory-efficient)."""

    def __init__(self, n_clients: int = 1):
        # Create FedJob
        job = FedJob(name="fedavg", min_clients=n_clients)

        # Model with persistor that keeps PyTorch tensors (no NumPy conversion)
        model = GigabyteModel()
        persistor = PTFileModelPersistor(model=model, allow_numpy_conversion=False)
        pt_model = PTModel(model=model, persistor=persistor)
        job.to_server(pt_model)

        # FedAvg with InTime aggregation (already memory-efficient)
        controller = FedAvg(
            num_clients=n_clients,
            num_rounds=3,
        )
        job.to_server(controller, id="controller")

        # Add tensor streaming
        job.to_server(TensorServerStreamer(), id="tensor_server_streamer")
        job.to_clients(TensorClientStreamer(), id="tensor_client_streamer")

        # Client script
        client_runner = ScriptRunner(
            script="minimal_client.py", script_args="", server_expected_format=ExchangeFormat.PYTORCH
        )
        job.to_clients(client_runner)

        # Initialize Recipe with the job
        super().__init__(job)


class ScatterGatherRecipe(Recipe):
    """Scatter and Gather Recipe."""

    def __init__(self, n_clients: int = 1):
        # Create FedJob
        job = FedJob(name="scatter_gather", min_clients=n_clients)

        # Model with persistor that keeps PyTorch tensors (no NumPy conversion)
        model = GigabyteModel()
        persistor = PTFileModelPersistor(model=model, allow_numpy_conversion=False)
        pt_model = PTModel(model=model, persistor=persistor)
        job.to_server(pt_model)

        # Aggregator
        aggregator = InTimeAccumulateWeightedAggregator(expected_data_kind=DataKind.WEIGHTS)
        job.to_server(aggregator, id="aggregator")

        # Shareable generator
        shareable_gen = FullModelShareableGenerator()
        job.to_server(shareable_gen, id="shareable_generator")

        # Controller - MUST reference the component IDs
        controller = ScatterAndGather(
            min_clients=n_clients,
            num_rounds=3,
            wait_time_after_min_received=0,
            aggregator_id="aggregator",
            persistor_id="persistor",
            shareable_generator_id="shareable_generator",
        )
        job.to_server(controller, id="controller")

        # Add tensor streaming
        job.to_server(TensorServerStreamer(), id="tensor_server_streamer")
        job.to_clients(TensorClientStreamer(), id="tensor_client_streamer")

        # Client script
        client_runner = ScriptRunner(
            script="minimal_client.py", script_args="", server_expected_format=ExchangeFormat.PYTORCH
        )
        job.to_clients(client_runner)

        # Initialize Recipe with the job
        super().__init__(job)


def profile_job(job_name: str, recipe: Recipe, job_num: int, n_clients: int = 1, debug: bool = False):
    """Profile a single job using Recipe.execute()."""
    print("\n" + "=" * 60)
    print(f"Profiling: {job_name}")
    print("=" * 60)

    # Create PocEnv with --once flag to disable daemon mode
    env = PocEnv(num_clients=n_clients, use_once=True)
    poc_workspace = env.poc_workspace

    # Start memory monitoring
    output_file = f"results/poc_job{job_num}.dat"
    monitor = MemoryMonitor(output_file, poc_workspace=poc_workspace, debug=debug)
    monitor.start()

    try:
        # Execute recipe and get Run object
        print("Deploying job...")
        run = recipe.execute(env=env)

        print(f"Job ID: {run.get_job_id()}")
        print("Waiting for processes to start...")
        time.sleep(10)  # Give processes time to start

        print("Monitoring job execution...")

        # Wait for job to complete (timeout after 5 minutes)
        result = run.get_result(timeout=300)

        if result:
            print(f"\nJob completed! Result workspace: {result}")
        else:
            print("\nJob did not complete within timeout or was stopped early")

        # Keep monitoring for a bit after completion to capture final state
        print("Capturing final memory state...")
        time.sleep(2)

    finally:
        # Stop monitoring
        (
            peak_total,
            peak_server_parent,
            peak_server_job,
            peak_client_parent,
            peak_client_job,
            peak_server_total,
            peak_client_total,
        ) = monitor.stop()

        # Stop POC and clean up workspace
        print("\nCleaning up...")
        env.stop(clean_poc=True)
        print("Cleanup complete")

        return (
            peak_total,
            peak_server_parent,
            peak_server_job,
            peak_client_parent,
            peak_client_job,
            peak_server_total,
            peak_client_total,
        )


def main():
    """Main profiling script."""
    # Create results directory
    os.makedirs("results", exist_ok=True)

    # Clean up old results
    for file in ["results/poc_job1.dat", "results/poc_job2.dat", "results/poc_summary.txt"]:
        if os.path.exists(file):
            os.remove(file)

    n_clients = 1
    results = []

    # Calculate actual model size
    test_model = GigabyteModel()
    model_size_gb = get_model_size_gb(test_model)
    total_params = sum(p.numel() for p in test_model.parameters())
    del test_model  # Free memory

    print("=" * 60)
    print("Memory Profiling: 2-Way Comparison")
    print("=" * 60)
    print(f"Model: {model_size_gb:.2f} GB ({total_params / 1e6:.0f}M parameters)")
    print(f"Clients: {n_clients}")
    print("Rounds: 3")
    print("=" * 60)

    # Job 1: FedAvg (InTime aggregation - already memory-efficient)
    recipe1 = CustomFedAvgRecipe(n_clients=n_clients)
    result1 = profile_job("FedAvg (InTime)", recipe1, 1, n_clients, debug=False)
    results.append(("Job 1: FedAvg (InTime)", *result1))

    # Job 2: Scatter and Gather
    recipe2 = ScatterGatherRecipe(n_clients=n_clients)
    result2 = profile_job("Scatter and Gather", recipe2, 2, n_clients, debug=False)
    results.append(("Job 2: Scatter and Gather", *result2))

    # Write summary
    with open("results/poc_summary.txt", "w") as f:
        f.write(f"Model Size: {model_size_gb:.2f} GB ({total_params / 1e6:.0f}M parameters)\n")
        f.write(f"Clients: {n_clients}, Rounds: 3\n")
        f.write("=" * 60 + "\n\n")

        for i, (
            name,
            peak_total,
            server_parent,
            server_job,
            client_parent,
            client_job,
            server_total,
            client_total,
        ) in enumerate(results):
            f.write(f"{name}\n")
            f.write("=" * len(name) + "\n")
            f.write(f"Peak memory: {peak_total} MB ({peak_total / 1024:.2f} GB)\n")
            f.write(f"  Server: {server_total} MB ({server_total / 1024:.2f} GB)\n")
            f.write(f"    - Parent: {server_parent} MB\n")
            f.write(f"    - Job:    {server_job} MB\n")
            f.write(f"  Client: {client_total} MB ({client_total / 1024:.2f} GB)\n")
            f.write(f"    - Parent: {client_parent} MB\n")
            f.write(f"    - Job:    {client_job} MB\n")
            f.write(f"Results saved to: results/poc_job{i + 1}.dat\n\n")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    with open("results/poc_summary.txt", "r") as f:
        print(f.read())

    print("\n" + "=" * 60)
    print("PROFILING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
