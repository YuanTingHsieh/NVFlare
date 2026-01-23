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
Minimal client that just returns model without training.
This focuses the memory profile on server-side aggregation.
"""

import nvflare.client as flare


def main():
    # Initialize NVFlare client
    flare.init()
    sys_info = flare.system_info()
    client_name = sys_info["site_name"]

    print(f"[{client_name}] Starting minimal client for memory profiling")

    while flare.is_running():
        # Receive model from server
        input_model = flare.receive()
        print(f"[{client_name}] Received model at round {input_model.current_round}")

        # No training - just send model back immediately
        # This focuses the memory profile on server-side aggregation
        output_model = flare.FLModel(
            params=input_model.params,
            metrics={"accuracy": 0.5},
            meta={"NUM_STEPS_CURRENT_ROUND": 100},
        )

        print(f"[{client_name}] Sending model back to server")
        flare.send(output_model)

        # Explicitly release references to free memory
        del input_model
        del output_model
        import gc

        gc.collect()

    print(f"[{client_name}] Complete")


if __name__ == "__main__":
    main()
