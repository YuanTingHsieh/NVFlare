# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

from typing import Optional

from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.executors.launcher_executor import (
    LauncherExecutor,
    model_data_to_shareable,
    shareable_to_model_data,
)


class CustomExecutor(LauncherExecutor):
    def __init__(
        self,
        data_exchanger_id: str,
        data_exchange_path: str,
        file_accessor_id: Optional[str] = None,
        get_timeout: Optional[float] = 10000,
    ):
        """CustomExecutor for autonomous vehicle training."""
        super().__init__(
            data_exchanger_id=data_exchanger_id,
            file_accessor_id=file_accessor_id,
            data_exchange_path=data_exchange_path,
        )

        self._timeout = get_timeout
        self._rounds = 0

    def _prepare_for_launch(self, shareable: Shareable):
        # dump weights for outer script to read
        model_data = shareable_to_model_data(shareable=shareable)
        self.data_exchanger.put(self._from_nvflare, data=model_data)
        self.data_exchanger.put("round_starts", {"round": self._rounds})

    def _get_result(self, task_name: str, fl_ctx: FLContext) -> Shareable:
        try:
            output_model_data = self.data_exchanger.get(self._to_nvflare, self._timeout)
            round_ends = self.data_exchanger.get("round_ends", self._timeout)
            external_round = round_ends["round"]
            if external_round != self._rounds:
                raise RuntimeError("rounds mismatch.")
            self._rounds += 1
            result = model_data_to_shareable(output_model_data)
            return result
        except Exception as e:
            err_msg = f"External training is not finished within timeout ({self._timeout}) seconds: {e}."
            self.log_exception(fl_ctx, err_msg)
            self.system_panic(err_msg, fl_ctx)
            return None
