#!/usr/bin/env python3
"""Export all jobs for PoC profiling."""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

from job1_fedavg_standard import job as job1
from job2_scatter_gather import job as job2
from job3_fedavg_memory_efficient import job as job3

# Export jobs
print("Exporting jobs for PoC mode...")

job1.export_job("poc_jobs/job1_fedavg_standard")
print("✓ Job 1 exported")

job2.export_job("poc_jobs/job2_scatter_gather")
print("✓ Job 2 exported")

job3.export_job("poc_jobs/job3_fedavg_memory_efficient")
print("✓ Job 3 exported")

print("\nAll jobs exported to poc_jobs/")
print("Ready for PoC profiling!")
