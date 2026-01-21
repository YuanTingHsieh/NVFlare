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
Job 1: FedAvg with standard aggregation (memory_efficient=False)

Expected memory: ~5 GB (3 clients × 1GB + buffer + result + framework)
"""

from small_model import GigabyteModel

from nvflare.app_common.workflows.fedavg import FedAvg
from nvflare.app_opt.pt.file_model_persistor import PTFileModelPersistor
from nvflare.app_opt.pt.job_config.model import PTModel
from nvflare.app_opt.tensor_stream.client import TensorClientStreamer
from nvflare.app_opt.tensor_stream.server import TensorServerStreamer
from nvflare.client.config import ExchangeFormat
from nvflare.job_config.api import FedJob
from nvflare.job_config.script_runner import ScriptRunner

# Create job
job = FedJob(name="job1_fedavg_standard")

# Add model with persistor that keeps PyTorch tensors (no NumPy conversion)
model = GigabyteModel()
persistor = PTFileModelPersistor(model=model, allow_numpy_conversion=False)
pt_model = PTModel(model=model, persistor=persistor)
job.to_server(pt_model)

# Add FedAvg workflow with STANDARD aggregation (memory_efficient=False)
workflow = FedAvg(num_clients=1, num_rounds=3, memory_efficient=False)  # STANDARD MODE - more memory
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
    job.simulator_run(workspace="/tmp/nvflare/job1_fedavg_standard", n_clients=1, log_config="full")
