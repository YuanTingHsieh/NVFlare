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
"""Translate deployed client configuration into managed task-worker routes."""

import os
import shlex
from dataclasses import replace

from nvflare.app_common.app_constant import AppConstants
from nvflare.client.config import ConfigKey
from nvflare.fuel.utils.config_factory import ConfigFactory
from nvflare.private.fed.deployment_supervisor.supervisor import WorkerDefinition

_CLIENT_API_EXECUTOR = "nvflare.app_common.executors.client_api_executor.ClientAPIExecutor"
_TASK_EXCHANGE_KEYS = {
    "params_exchange_format": ConfigKey.EXCHANGE_FORMAT,
    "server_expected_format": ConfigKey.SERVER_EXPECTED_FORMAT,
    "params_transfer_type": ConfigKey.TRANSFER_TYPE,
    "train_task_name": ConfigKey.TRAIN_TASK_NAME,
    "evaluate_task_name": ConfigKey.EVAL_TASK_NAME,
    "submit_model_task_name": ConfigKey.SUBMIT_MODEL_TASK_NAME,
    "train_with_evaluation": ConfigKey.TRAIN_WITH_EVAL,
}
_TASK_EXCHANGE_DEFAULTS = {
    ConfigKey.EXCHANGE_FORMAT: "raw",
    ConfigKey.SERVER_EXPECTED_FORMAT: "numpy",
    ConfigKey.TRANSFER_TYPE: "FULL",
    ConfigKey.TRAIN_TASK_NAME: AppConstants.TASK_TRAIN,
    ConfigKey.EVAL_TASK_NAME: AppConstants.TASK_VALIDATION,
    ConfigKey.SUBMIT_MODEL_TASK_NAME: AppConstants.TASK_SUBMIT_MODEL,
    ConfigKey.TRAIN_WITH_EVAL: False,
}


def _component_path(definition):
    if not isinstance(definition, dict):
        raise TypeError("component definition must be a dict")
    return definition.get("path") or definition.get("class_path") or definition.get("name")


def _script_worker(args, custom_dir):
    mode = args.get("execution_mode")
    if mode == "in_process":
        script = args.get("task_script_path")
        arguments = args.get("task_script_args", "")
    elif mode == "external_process":
        command = args.get("command")
        tokens = shlex.split(command) if isinstance(command, str) else list(command or [])
        script_index = next((i for i, token in enumerate(tokens) if token.endswith(".py")), None)
        if script_index is None:
            raise ValueError("managed external_process command must identify a Python task script")
        script = tokens[script_index]
        arguments = shlex.join(tokens[script_index + 1 :])
        if script.startswith("custom/"):
            script = script[len("custom/") :]
    else:
        raise ValueError(f"managed ClientAPIExecutor does not support execution_mode={mode!r}")
    if not isinstance(script, str) or not script:
        raise ValueError("managed ClientAPIExecutor requires a task script")
    entrypoint = script if os.path.isabs(script) else os.path.join(custom_dir, script)
    # FedJob omits constructor arguments that retain ClientAPIExecutor defaults.
    # Rebuild the complete trainer-side contract before applying serialized
    # overrides so task-mode helpers behave exactly like the source executor.
    exchange = dict(_TASK_EXCHANGE_DEFAULTS)
    exchange.update({target: args[source] for source, target in _TASK_EXCHANGE_KEYS.items() if source in args})
    return WorkerDefinition(
        kind="script",
        entrypoint=entrypoint,
        arguments=arguments if isinstance(arguments, str) else shlex.join(arguments),
        state_id="client-api",
        task_exchange=exchange,
        memory_gc_rounds=args.get("memory_gc_rounds", 0),
        cuda_empty_cache=args.get("cuda_empty_cache", False),
        source_mode=mode,
    )


def _matching_filters(config, section, task_name):
    result = []
    for chain in config.get(section, []) or []:
        tasks = chain.get("tasks", []) if isinstance(chain, dict) else []
        if "*" in tasks or task_name in tasks:
            result.extend(chain.get("filters", []))
    return result


def _task_components(config):
    """Return only explicitly configured task-local components.

    The fresh worker exposes these through a small component registry, not a
    reconstructed Client Engine or Cell.  Components that need either remain
    unsupported by the managed-A boundary.
    """
    result = []
    for definition in config.get("components", []) or []:
        if not isinstance(definition, dict) or not isinstance(definition.get("id"), str) or not definition["id"]:
            raise ValueError("managed task component requires a nonempty id")
        if not _component_path(definition):
            raise ValueError(f"managed task component '{definition['id']}' requires a path")
        result.append(definition)
    return result


def load_worker_routes(workspace, job_id):
    """Load routes without constructing job code in the deployment supervisor."""
    config_dir = workspace.get_app_config_dir(job_id)
    loaded = ConfigFactory.load_config("config_fed_client.json", [config_dir])
    if loaded is None:
        raise RuntimeError("managed deployment supervisor cannot find config_fed_client")
    config = loaded.to_dict()
    custom_dir = workspace.get_app_custom_dir(job_id)
    components = _task_components(config)
    routes = {}
    for executor_entry in config.get("executors", []) or []:
        if not isinstance(executor_entry, dict):
            raise TypeError("executor entry must be a dict")
        tasks = executor_entry.get("tasks", [])
        definition = executor_entry.get("executor")
        path = _component_path(definition)
        args = definition.get("args", {}) if isinstance(definition, dict) else {}
        for task_name in tasks:
            worker = (
                _script_worker(args, custom_dir)
                if path == _CLIENT_API_EXECUTOR
                else WorkerDefinition(kind="executor", entrypoint=path, factory_args=args)
            )
            worker = replace(
                worker,
                module_search_paths=[custom_dir],
                task_components=components,
                input_filters=_matching_filters(config, "task_data_filters", task_name),
                result_filters=_matching_filters(config, "task_result_filters", task_name),
            )
            routes[task_name] = worker
    if not routes:
        raise RuntimeError("managed deployment supervisor found no executor routes")
    return routes
