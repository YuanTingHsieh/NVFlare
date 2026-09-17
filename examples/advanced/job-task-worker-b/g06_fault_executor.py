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
"""Shared test-only A/B Executor for the bounded G06 failure rows."""

import os

from nvflare.apis.executor import Executor


class G06FaultExecutor(Executor):
    def __init__(self, mode):
        super().__init__()
        if mode not in ("exception", "nonzero"):
            raise ValueError("mode must be 'exception' or 'nonzero'")
        self.mode = mode

    def execute(self, task_name, shareable, fl_ctx, abort_signal):
        print(f"G06_FAULT_POINT mode={self.mode} execute=true", flush=True)
        if self.mode == "exception":
            raise RuntimeError("intentional G06 application exception from Executor.execute")
        os._exit(7)
