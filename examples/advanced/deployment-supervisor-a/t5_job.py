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

import argparse

from t5_components import SimpleNetwork, SlowUploadExecutor, SlowUploadVerificationFilter

from nvflare.apis.dxo import DataKind
from nvflare.app_common.aggregators.intime_accumulate_model_aggregator import InTimeAccumulateWeightedAggregator
from nvflare.app_common.shareablegenerators.full_model_shareable_generator import FullModelShareableGenerator
from nvflare.app_common.workflows.scatter_and_gather import ScatterAndGather
from nvflare.app_opt.pt.file_model_persistor import PTFileModelPersistor
from nvflare.job_config.api import FedJob
from nvflare.job_config.defs import FilterType


def main():
    parser = argparse.ArgumentParser(description="Export the Architecture A real slow-upload fixture")
    parser.add_argument("--output", required=True)
    parser.add_argument("--client", default="site-1")
    parser.add_argument("--payload-mib", type=int, default=64)
    args = parser.parse_args()
    if args.payload_mib < 1:
        parser.error("payload-mib must be positive")

    job = FedJob(name="architecture-a-t5-slow-upload", min_clients=1, mandatory_clients=[args.client])
    shareable_generator_id = job.to_server(FullModelShareableGenerator(), id="shareable_generator")
    persistor_id = job.to_server(PTFileModelPersistor(model=SimpleNetwork()), id="persistor")
    aggregator_id = job.to_server(
        InTimeAccumulateWeightedAggregator(expected_data_kind=DataKind.WEIGHTS), id="aggregator"
    )
    job.to_server(
        ScatterAndGather(
            min_clients=1,
            num_rounds=1,
            aggregator_id=aggregator_id,
            persistor_id=persistor_id,
            shareable_generator_id=shareable_generator_id,
            train_task_name="train",
            train_timeout=600,
        )
    )
    job.to(
        SlowUploadVerificationFilter(expected_mib=args.payload_mib),
        "server",
        tasks=["train"],
        filter_type=FilterType.TASK_RESULT,
    )
    job.to(SlowUploadExecutor(payload_mib=args.payload_mib), args.client, tasks=["train"])
    job.job.add_resource_spec(args.client, {"num_of_gpus": 1})
    job.export_job(args.output)


if __name__ == "__main__":
    main()
