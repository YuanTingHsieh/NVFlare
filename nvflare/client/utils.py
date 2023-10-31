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

from typing import Dict

from nvflare.apis.fl_context import FLContext
from nvflare.app_common.data_exchange.constants import ExchangeFormat
from nvflare.fuel.utils.constants import Mode
from nvflare.fuel.utils.pipe.file_pipe import FilePipe
from nvflare.fuel.utils.pipe.ipc_ppipe import IPCPPipe
from nvflare.fuel.utils.pipe.pipe import Pipe


def numerical_params_diff(original: Dict, new: Dict) -> Dict:
    """Calculates the numerical parameter difference.

    Args:
        original: A dict of numerical values.
        new: A dict of numerical values.

    Returns:
        A dict with same key as original dict,
        value are the difference between original and new.
    """
    diff_dict = {}
    for k in original:
        if k not in new:
            continue
        if isinstance(new[k], list) and isinstance(original[k], list):
            diff = [new[k][i] - original[k][i] for i in range(len(new[k]))]
        else:
            diff = new[k] - original[k]

        diff_dict[k] = diff
    return diff_dict


DIFF_FUNCS = {ExchangeFormat.PYTORCH: numerical_params_diff, ExchangeFormat.NUMPY: numerical_params_diff}


def get_external_pipe_class(pipe: Pipe) -> str:
    if isinstance(pipe, IPCPPipe):
        return "IPCAPipe"
    elif isinstance(pipe, FilePipe):
        return "FilePipe"
    return ""


def get_external_pipe_args(pipe: Pipe, fl_ctx: FLContext) -> dict:
    args = {}
    if isinstance(pipe, IPCPPipe):
        args["site_name"] = fl_ctx.get_identity_name()
        args["root_url"] = pipe.cell.core_cell.root_url
        args["secure"] = pipe.cell.core_cell.secure
        args["workspace_dir"] = fl_ctx.get_engine().get_workspace().get_root_dir()
    elif isinstance(pipe, FilePipe):
        args["root_path"] = pipe.root_path
        args["mode"] = Mode.ACTIVE
    return args
