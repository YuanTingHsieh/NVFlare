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
"""Unmodified-style managed Client API task script."""

import nvflare.client as flare


def main():
    flare.init()
    while flare.is_running():
        model = flare.receive()
        flare.send(
            flare.FLModel(
                params={name: value + 1 for name, value in model.params.items()},
                current_round=model.current_round,
                metrics={"managed": True},
            )
        )


if __name__ == "__main__":
    main()
