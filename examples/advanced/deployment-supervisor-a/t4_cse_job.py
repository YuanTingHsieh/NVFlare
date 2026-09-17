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
"""Export A's exact common 16-task hello-pt continuity workload."""

import argparse
import os
import sys
from pathlib import Path

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
from nvflare.job_config.api import FedJob


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--common-root", required=True)
    parser.add_argument("--dataset-root", required=True)
    args = parser.parse_args()
    common_root = Path(args.common_root).resolve()
    if not (common_root / "client.py").is_file() or not (common_root / "model.py").is_file():
        parser.error("common-root must contain client.py and model.py")

    sys.path.insert(0, str(common_root))
    from model import SimpleNetwork

    job = FedJob(name="architecture-a-t4-common-cse", min_clients=2, mandatory_clients=["site-1", "site-2"])
    shareable_generator_id = job.to_server(FullModelShareableGenerator(), id="shareable_generator")
    persistor_id = job.to_server(PTFileModelPersistor(model=SimpleNetwork()), id="persistor")
    aggregator_id = job.to_server(
        InTimeAccumulateWeightedAggregator(expected_data_kind=DataKind.WEIGHTS), id="aggregator"
    )
    job.to_server(
        ScatterAndGather(
            min_clients=2,
            num_rounds=3,
            aggregator_id=aggregator_id,
            persistor_id=persistor_id,
            shareable_generator_id=shareable_generator_id,
            train_task_name="train",
            train_timeout=600,
            wait_time_after_min_received=0,
            ignore_result_error=False,
        )
    )
    job.to_server(IntimeModelSelector(key_metric="accuracy"))
    model_locator_id = job.to_server(PTFileModelLocator(pt_persistor_id=persistor_id), id="model_locator")
    job.to_server(
        CrossSiteModelEval(
            model_locator_id=model_locator_id,
            participating_clients=["site-1", "site-2"],
            submit_model_timeout=600,
            validation_timeout=600,
        )
    )
    job.to_server(ValidationJsonGenerator())

    output = str(Path(args.output).resolve())
    previous_dir = os.getcwd()
    try:
        os.chdir(common_root)
        for site, offset in (("site-1", 1), ("site-2", 2)):
            task_args = (
                f"--batch-size 8 --epochs 1 --num-workers 0 --seed 202610 --site-seed-offset {offset} "
                f"--dataset-root {args.dataset_root} --synthetic-data --train-size 32 --test-size 16"
            )
            executor = ClientAPIExecutor(
                execution_mode="in_process",
                task_script_path="client.py",
                task_script_args=task_args,
                task_wait_timeout=600.0,
                result_wait_timeout=600.0,
                params_exchange_format="pytorch",
                params_transfer_type="FULL",
            )
            job.to(executor, site, tasks=["train", "validate", "submit_model"])
            job.to(str(common_root / "client.py"), site)
            job.to(str(common_root / "model.py"), site)
            job.job.add_resource_spec(site, {"num_of_gpus": 1})
        job.export_job(output)
    finally:
        os.chdir(previous_dir)


if __name__ == "__main__":
    main()
