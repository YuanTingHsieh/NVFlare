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
"""Configuration for framework-owned task-scope transfer processes."""

import re

from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_constant import SystemConfigs
from nvflare.fuel.utils.config_service import ConfigService
from nvflare.private.fed.client.client_json_config import ClientJsonConfigurator
from nvflare.private.fed.client.client_runner import ClientRunnerConfig, TaskRouter
from nvflare.private.fed_json_config import FedJsonConfigurator
from nvflare.private.json_configer import ConfigContext, ConfigError

PUBLICATION_SECTION = "task_scope_publication"
PUBLICATION_COMPONENT_CAPABILITY = "supports_task_scope_publication"
PUBLICATION_ACK_PROP = "__task_scope_publication_ack"
FRAMEWORK_TRANSFER_COMPONENT_PATHS = {
    "nvflare.app_common.logging.job_log_streamer.JobLogStreamer",
}


class TaskScopeTransferConfigurator(ClientJsonConfigurator):
    """Build only the framework transfer graph and explicit publication hooks.

    Pull and push must not construct the application's Executor, Learner, filters,
    or general component graph. Push may construct components explicitly registered
    under ``task_scope_publication.components``. These components and their nested
    dependencies are owned by the push process and receive its real lifecycle and
    send/ACK events.
    """

    def __init__(self, *args, include_publication_components: bool, **kwargs):
        super().__init__(*args, **kwargs)
        self.include_publication_components = include_publication_components

    def process_config_element(self, config_ctx: ConfigContext, node):
        element = node.element
        path = node.path()

        if path == "format_version":
            self.format_version = element
            return

        if re.search(r"^components\.#[0-9]+$", path) and self._is_framework_transfer_component(element):
            self._register_component(element, config_ctx, node, require_publication_capability=False)
            return

        if self.include_publication_components and re.search(rf"^{PUBLICATION_SECTION}\.components\.#[0-9]+$", path):
            self._register_component(element, config_ctx, node, require_publication_capability=True)

    @staticmethod
    def _is_framework_transfer_component(element):
        if not isinstance(element, dict):
            return False
        class_path = element.get("path") or element.get("class_path") or element.get("name")
        return isinstance(class_path, str) and class_path.split("#", 1)[0] in FRAMEWORK_TRANSFER_COMPONENT_PATHS

    def _register_component(self, element, config_ctx, node, require_publication_capability):
        if not isinstance(element, dict):
            raise ConfigError("task-scope transfer component must be a component configuration")
        component_id = element.get("id")
        if not isinstance(component_id, str) or not component_id:
            raise ConfigError("task-scope transfer component requires a nonempty string id")
        if component_id in self.components:
            raise ConfigError(f'duplicate task-scope transfer component id "{component_id}"')

        component = self.authorize_and_build_component(element, config_ctx, node)
        if not isinstance(component, FLComponent):
            raise ConfigError(
                f'task-scope transfer component "{component_id}" must be an FLComponent, ' f"but got {type(component)}"
            )
        if require_publication_capability and getattr(component, PUBLICATION_COMPONENT_CAPABILITY, False) is not True:
            raise ConfigError(
                f'task-scope publication component "{component_id}" must declare '
                f"{PUBLICATION_COMPONENT_CAPABILITY}=True"
            )
        self.components[component_id] = component

    def finalize_config(self, config_ctx: ConfigContext):
        section = self.config_data.get(PUBLICATION_SECTION)
        if section is not None:
            if not isinstance(section, dict):
                raise ConfigError(f'"{PUBLICATION_SECTION}" must be a dictionary')
            components = section.get("components", [])
            if not isinstance(components, list):
                raise ConfigError(f'"{PUBLICATION_SECTION}.components" must be a list')

        FedJsonConfigurator.finalize_config(self, config_ctx)
        self.runner_config = ClientRunnerConfig(
            task_router=TaskRouter(),
            task_data_filters={},
            task_result_filters={},
            components=self.components,
            handlers=self.handlers,
            default_task_fetch_interval=self._default_task_fetch_interval,
        )

        ConfigService.initialize(
            section_files={},
            config_path=[self.app_root],
            parsed_args=self.args,
            var_dict=self.cmd_vars,
        )
        ConfigService.add_section(
            section_name=SystemConfigs.APPLICATION_CONF,
            data=self.config_data,
        )
