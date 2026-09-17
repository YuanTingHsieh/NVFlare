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
"""Export the named T7a legacy resident hello-numpy regression."""

import argparse
import os
from pathlib import Path

from nvflare.app_common.np.recipes.fedavg import NumpyFedAvgRecipe
from nvflare.client.config import TransferType
from nvflare.recipe import SimEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--clients", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()
    if args.clients < 1 or args.rounds < 1:
        parser.error("clients and rounds must be positive")

    source_root = Path(__file__).resolve().parents[3]
    client_dir = source_root / "examples" / "hello-world" / "hello-numpy"
    output = str(Path(args.output).resolve())
    previous_dir = os.getcwd()
    try:
        os.chdir(client_dir)
        recipe = NumpyFedAvgRecipe(
            name="architecture-a-t7a-legacy-hello-numpy",
            min_clients=args.clients,
            num_rounds=args.rounds,
            model=[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            train_script="client.py",
            train_args="--update_type full",
            launch_external_process=False,
            params_transfer_type=TransferType.FULL,
        )
        recipe.export(output, env=SimEnv(num_clients=args.clients))
    finally:
        os.chdir(previous_dir)


if __name__ == "__main__":
    main()
