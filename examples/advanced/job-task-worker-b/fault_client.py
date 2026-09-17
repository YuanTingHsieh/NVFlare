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
"""Deliberately fail one Client API task for the bounded G06 gate."""

import argparse
import os

import nvflare.client as flare


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--failure-mode", choices=("exception", "nonzero"), required=True)
    args = parser.parse_args()

    flare.init()
    if not flare.is_running():
        raise RuntimeError("fault client stopped before receiving its task")
    flare.receive()
    if args.failure_mode == "nonzero":
        os._exit(7)
    raise RuntimeError("deliberate Architecture B application exception")


if __name__ == "__main__":
    main()
