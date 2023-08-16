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

import inspect
import os
from importlib import util
from threading import Thread
from types import ModuleType
from typing import Optional

from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.launcher import Launcher
from nvflare.app_common.model_exchange.ml_to_fl import NVF_DECORATOR


def _get_decorated_method(decorator_name: str, module: ModuleType) -> Optional[str]:
    members = inspect.getmembers(module)
    result = None
    for m in members:
        # Check if the attribute is a function or a method
        if inspect.isfunction(m[1]) or inspect.ismethod(m[1]):
            # Check if the attribute has the custom attribute set by the decorator
            if hasattr(m[1], "__wrapped__"):
                if getattr(m[1], NVF_DECORATOR) == decorator_name:
                    result = m[0]
                    break
    return result


class ML2FLLauncher(Launcher):
    def __init__(self, script: str):
        """ML2FLLauncher.

        Args:
            script (str): Script that contains decorated code using ML-to-FL.
        """
        super().__init__()

        self._app_dir = None
        self._script = script
        self._module = None
        self._train = None
        self._global_eval = None
        self._threads = {}

    def initialize(self, fl_ctx: FLContext):
        self._app_dir = self.get_app_dir(fl_ctx)
        if not self._script.endswith(".py"):
            raise RuntimeError(f"script ({self._script}) needs to be a python script.")
        try:
            original_directory = os.getcwd()
            os.chdir(self._app_dir)
            module_path = os.path.join(self._app_dir, self._script)
            module_spec = util.spec_from_file_location(self._script[:-3], module_path)
            self._module = util.module_from_spec(module_spec)
            module_spec.loader.exec_module(self._module)
            self._train = _get_decorated_method("fl_train", self._module)
            if self._train is None:
                raise RuntimeError("no method is decorated using 'fl_train'")
            self._global_eval = _get_decorated_method("fl_evaluate", self._module)
            os.chdir(original_directory)
        except Exception:
            raise RuntimeError(f"can't load python module from ({self._script})")

    def _train_workflow(self):
        original_directory = os.getcwd()
        os.chdir(self._app_dir)
        if self._global_eval is not None:
            evaluate_fn = getattr(self._module, self._global_eval)
            evaluate_fn()
        train_fn = getattr(self._module, self._train)
        train_fn()
        os.chdir(original_directory)

    def launch_task(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> bool:
        if task_name in self._threads:
            return False

        if task_name == "train":
            thread = Thread(target=self._train_workflow)
            thread.start()
            self._threads[task_name] = thread
            return True
        return False

    def stop_task(self, task_name: str, fl_ctx: FLContext) -> None:
        if task_name in self._threads:
            self._threads[task_name].join()
            self._threads.pop(task_name)
