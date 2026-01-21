# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""
Job 2: Scatter and Gather (standard workflow)

Expected memory: ~5 GB (similar to standard FedAvg - no optimization available)
"""

from small_model import GigabyteModel

from nvflare.apis.dxo import DataKind
from nvflare.app_common.aggregators.intime_accumulate_model_aggregator import InTimeAccumulateWeightedAggregator
from nvflare.app_common.shareablegenerators.full_model_shareable_generator import FullModelShareableGenerator
from nvflare.app_common.workflows.scatter_and_gather import ScatterAndGather
from nvflare.app_opt.pt.job_config.model import PTModel
from nvflare.app_opt.tensor_stream.client import TensorClientStreamer
from nvflare.app_opt.tensor_stream.server import TensorServerStreamer
from nvflare.client.config import ExchangeFormat
from nvflare.job_config.api import FedJob
from nvflare.job_config.script_runner import ScriptRunner

# Create job
job = FedJob(name="job2_scatter_gather")

# Add model with persistor that keeps PyTorch tensors (no NumPy conversion)
from nvflare.app_opt.pt.file_model_persistor import PTFileModelPersistor

model = GigabyteModel()
persistor = PTFileModelPersistor(model=model, allow_numpy_conversion=False)
pt_model = PTModel(model=model, persistor=persistor)
job.to_server(pt_model)

# Add shareable generator
shareable_gen = FullModelShareableGenerator()
job.to_server(shareable_gen, id="shareable_generator")

# Add aggregator
aggregator = InTimeAccumulateWeightedAggregator(expected_data_kind=DataKind.WEIGHTS)
job.to_server(aggregator, id="aggregator")

# Add Scatter and Gather workflow
workflow = ScatterAndGather(
    min_clients=1,
    num_rounds=3,
    wait_time_after_min_received=0,
    aggregator_id="aggregator",
    persistor_id="persistor",
    shareable_generator_id="shareable_generator",
)
job.to_server(workflow)

# Add Tensor Streaming to avoid OOM with 2.43GB model
job.to_server(TensorServerStreamer(), "tensor_server_streamer")
job.to_clients(TensorClientStreamer(), "tensor_client_streamer")

# Add client
client_runner = ScriptRunner(
    script="minimal_client.py",
    script_args="",
    server_expected_format=ExchangeFormat.PYTORCH,
)
job.to_clients(client_runner)

if __name__ == "__main__":
    job.simulator_run(workspace="/tmp/nvflare/job2_scatter_gather", n_clients=1)
