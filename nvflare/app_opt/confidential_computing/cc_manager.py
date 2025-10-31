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

import random
import sys
import threading
import time

from nvflare.apis.app_validation import AppValidationKey
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.server_engine_spec import ServerEngineSpec
from nvflare.app_opt.confidential_computing.cc_authorizer import CCAuthorizer, CCTokenGenerateError, CCTokenVerifyError
from nvflare.fuel.f3.cellnet.core_cell import make_reply
from nvflare.fuel.f3.cellnet.defs import MessageHeaderKey
from nvflare.fuel.f3.cellnet.defs import ReturnCode as F3ReturnCode
from nvflare.fuel.f3.cellnet.fqcn import FQCN
from nvflare.fuel.hci.conn import Connection
from nvflare.fuel.hci.server.constants import ConnProps
from nvflare.fuel.utils.admin_name_utils import is_valid_admin_client_name
from nvflare.private.defs import ClientType, new_cell_message
from nvflare.private.fed.server.training_cmds import TrainingCommandModule

CC_TOKEN = "_cc_token"
CC_ISSUER = "_cc_issuer"
CC_NAMESPACE = "_cc_namespace"
CC_INFO = "_cc_info"
CC_TOKEN_VALIDATED = "_cc_token_validated"
CC_VERIFY_ERROR = "_cc_verify_error."

CC_ISSUER_ID = "issuer_id"
TOKEN_GENERATION_TIME = "token_generation_time"
TOKEN_EXPIRATION = "token_expiration"

CC_VERIFICATION_FAILED = "not meeting CC requirements"

# Dedicated CC validation channel and topics
CC_CHANNEL = "cc_validation"
CC_TOPIC_REQUEST_TOKEN = "request_fresh_token"
CC_TOPIC_GET_SITES = "get_sites"
CC_TOPIC_VALIDATE_ALL = "validate_all_tokens"


class CCManager(FLComponent):
    def __init__(
        self,
        cc_issuers_conf: list[dict[str, str]],
        cc_verifier_ids: list[str],
        verify_frequency: int = 600,
        cc_enabled_sites: list[str] = [],
    ):
        """Manage all confidential computing related tasks.

        This manager does the following tasks:
            1. obtaining its own CC token
            2. preparing the CC tokens to be sent to other sites
            3. validating the CC tokens received from other sites
            4. not allowing the system to start if failed to get CC token
            5. shutdown the system if CC tokens expired

        Note:
            arguments example:
                "cc_issuers_conf": [
                    {
                        "issuer_id": "mock_authorizer",
                        "token_expiration": 100
                    }
                ],
                "cc_verifier_ids": [
                    "mock_authorizer"
                ],
                "verify_frequency": 120,
                "cc_enabled_sites": [
                    "server1",
                    "site-1",
                    "site-2"
                ]

        Args:
            cc_issuers_conf: configuration of the CC token issuers.
                Each item in the list is a dict that contains the CC token issuer component ID,
                and the token expiration time in seconds.
            cc_verifier_ids: CC token verifiers component IDs
            verify_frequency: CC tokens verification frequency
            cc_enabled_sites: list of sites that are enabled for CC
        """
        FLComponent.__init__(self)
        self.site_name = None
        self.cc_issuers_conf = cc_issuers_conf
        self.cc_verifier_ids = cc_verifier_ids
        self.cc_enabled_sites = cc_enabled_sites

        if not isinstance(verify_frequency, int):
            raise ValueError(f"verify_frequency must be int, but got {type(verify_frequency).__name__}")
        self.verify_frequency = int(verify_frequency)

        self.verify_time = None
        self.cc_issuers = {}
        self.cc_verifiers = {}

        # Track validation state
        self.system_validated = False

        # Store engine reference for cell handlers
        self.engine = None

        # Use RLock (reentrant lock) to allow same thread to acquire multiple times
        self.lock = threading.RLock()

        # Cross-site validation support
        self.cross_validation_thread = None
        self.cross_validation_interval = verify_frequency

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.SYSTEM_BOOTSTRAP:
            try:
                self._setup_cc_authorizers(fl_ctx)
                self.site_name = fl_ctx.get_identity_name()
                self.engine = fl_ctx.get_engine()
                self.logger.info(f"Initialized CCManager for site: {self.site_name}")
            except Exception as e:
                err = f"Exception in attestation preparation: {e}"
                self.logger.error(err)
                self._initiate_shutdown(err, fl_ctx)
        elif event_type == EventType.BEFORE_CLIENT_REGISTER:
            # Client side: prepare CC tokens for registration
            self._generate_and_attach_tokens(fl_ctx)
        elif event_type == EventType.AFTER_CLIENT_REGISTER:
            # Client side: validate server's CC token
            self._validate_server_tokens(fl_ctx)
        elif event_type == EventType.CLIENT_REGISTER_RECEIVED:
            # Server side: validate client's token and prepare server's token
            # Skip CC processing for admin clients (they don't participate in FL jobs)
            client_type = fl_ctx.get_prop(FLContextKey.CLIENT_TYPE)
            if client_type == ClientType.ADMIN:
                self.logger.info(f"Skipping CC validation for admin client (type={client_type})")
                return

            self.logger.info(f"Processing CC validation for client (type={client_type})")

            self._validate_peer_tokens(fl_ctx)
            self._generate_and_attach_tokens(fl_ctx)
        elif event_type == EventType.BEFORE_CHECK_CLIENT_RESOURCES:
            # Server side: job scheduler check client resources
            # Perform cross-site validation before scheduling jobs
            # Use lock to prevent race condition on system_validated check/set
            with self.lock:
                if not self.system_validated:
                    validation_passed = self._perform_cross_site_validation(fl_ctx)
                    if validation_passed:
                        self.system_validated = True
                        self.logger.info("System cross-site validation completed successfully")
        elif event_type == EventType.AFTER_CHECK_CLIENT_RESOURCES:
            client_resource_result = fl_ctx.get_prop(FLContextKey.RESOURCE_CHECK_RESULT)
            if client_resource_result:
                for site_name, check_result in client_resource_result.items():
                    is_resource_enough, reason = check_result
                    if not is_resource_enough and reason.startswith(CC_VERIFY_ERROR):
                        self.logger.error(f"Client resource check failed: {reason}")
                        self._initiate_shutdown(reason, fl_ctx)
                        break
        elif event_type == EventType.SUBMIT_JOB:
            job_meta = fl_ctx.get_prop(FLContextKey.JOB_META, {})
            byoc = job_meta.get(AppValidationKey.BYOC, False)
            if byoc:
                fl_ctx.set_prop(
                    key=FLContextKey.JOB_BLOCK_REASON, value="BYOC job not allowed for CC", sticky=False, private=True
                )
        elif event_type == EventType.SYSTEM_START:
            # Register CC validation channel handlers and start validation
            self._register_cc_handlers(fl_ctx)
            self._start_cross_site_validation(fl_ctx)
        elif event_type == EventType.SYSTEM_END:
            # Stop cross-site validation
            self._stop_cross_site_validation()

    def _setup_cc_authorizers(self, fl_ctx):
        engine = fl_ctx.get_engine()
        for conf in self.cc_issuers_conf:
            issuer_id = conf.get(CC_ISSUER_ID)
            # TODO: should we make expiration an instance variable of the issuer?
            expiration = conf.get(TOKEN_EXPIRATION)
            issuer = engine.get_component(issuer_id)
            if not isinstance(issuer, CCAuthorizer):
                raise RuntimeError(f"cc_issuer_id {issuer_id} must be a CCAuthorizer, but got {type(issuer).__name__}")
            self.cc_issuers[issuer] = expiration

        for v_id in self.cc_verifier_ids:
            verifier = engine.get_component(v_id)
            if not isinstance(verifier, CCAuthorizer):
                raise RuntimeError(f"cc_verifier_id {v_id} must be a CCAuthorizer, but got {type(verifier).__name__}")
            namespace = verifier.get_namespace()
            if namespace in self.cc_verifiers.keys():
                raise RuntimeError(f"Authorizer with namespace: {namespace} already exist.")
            self.cc_verifiers[namespace] = verifier

    def _generate_and_attach_tokens(self, fl_ctx: FLContext):
        """Generate and attach CC tokens for sending to peer."""
        cc_infos = self._generate_fresh_tokens_for_validation()
        cc_infos_to_send = self._format_tokens_for_transmission(cc_infos)

        # Set CC_INFO directly for both client and server
        fl_ctx.set_prop(key=CC_INFO, value=cc_infos_to_send, sticky=False, private=False)
        self.logger.info("Prepared CC tokens for peer")

    def _validate_peer_tokens(self, fl_ctx: FLContext):
        """Validate peer's CC info during registration."""
        peer_ctx = fl_ctx.get_peer_context()
        if not peer_ctx:
            self.logger.error("No peer context received in registration response!")
            self._initiate_shutdown("No peer context received in registration response!", fl_ctx)
            return
        peer_cc_info = peer_ctx.get_prop(CC_INFO)

        if not peer_cc_info:
            self.logger.error("No CC infos received in registration response!")
            self._initiate_shutdown("Peer did not provide CC info", fl_ctx)
            return

        peer_name = peer_ctx.get_identity_name()
        participants_cc_info = {peer_name: peer_cc_info}

        err = self._validate_participants_tokens(participants_cc_info)
        if err:
            self.logger.error(f"Peer CC info validation failed: {err}")
            self._initiate_shutdown(f"Peer CC info validation failed: {err}", fl_ctx)
            return

        self.logger.info(f"Validated peer CC info: {peer_name}")

    def _validate_server_tokens(self, fl_ctx: FLContext):
        """Client side: Validate server's CC info from registration response.

        The server's CC info was extracted by the communicator and stored in fl_ctx
        (not peer_ctx, which is used server-side).
        """
        # Get server CC info from main context (stored by communicator)
        server_cc_info = fl_ctx.get_prop(CC_INFO)

        if not server_cc_info:
            self.logger.error("No server CC info received in registration response!")
            self._initiate_shutdown("Server did not provide CC info", fl_ctx)
            return

        # Use server name for validation
        participants_cc_info = {self.site_name: server_cc_info}

        err = self._validate_participants_tokens(participants_cc_info)
        if err:
            self.logger.error(f"Server CC info validation failed: {err}")
            self._initiate_shutdown(f"Server CC info validation failed: {err}", fl_ctx)
            return

        self.logger.info("Validated server CC info")

    def _format_tokens_for_transmission(self, site_cc_info: list[dict[str, str]]) -> list[dict[str, str]]:
        cc_info = []
        for i in site_cc_info:
            namespace = i.get(CC_NAMESPACE)
            token = i.get(CC_TOKEN)
            cc_info.append({CC_TOKEN: token, CC_NAMESPACE: namespace, CC_TOKEN_VALIDATED: False})
        return cc_info

    def _validate_participants_tokens(self, participants_tokens: dict[str, list[dict[str, str]]]) -> str:
        self.logger.info(f"Validating participant tokens {participants_tokens=}")
        _, invalid_participant_list = self._verify_participants_tokens(participants_tokens)
        if invalid_participant_list:
            invalid_participant_string = ",".join(invalid_participant_list)
            return f"Participant {invalid_participant_string}" + CC_VERIFICATION_FAILED
        else:
            return ""

    def _verify_participants_tokens(
        self, participants_tokens: dict[str, list[dict[str, str]]]
    ) -> tuple[dict[str, bool], list[str]]:
        """Verifies tokens for all participants.

        Args:
            participants_tokens: dict of participant name to list of tokens

        Returns:
            tuple of (result, invalid_participant_list)
            result: dict of participant name to bool
            invalid_participant_list: list of invalid participants
        """
        result = {}
        invalid_participant_list = []
        if not participants_tokens:
            return result, invalid_participant_list
        for k, cc_info in participants_tokens.items():
            if k not in self.cc_enabled_sites:
                result[k] = True
                continue
            for v in cc_info:
                token = v.get(CC_TOKEN, "")
                namespace = v.get(CC_NAMESPACE, "")
                verifier = self.cc_verifiers.get(namespace, None)
                try:
                    if verifier and verifier.verify(token):
                        result[k + "." + namespace] = True
                    else:
                        invalid_participant_list.append(k + " namespace: {" + namespace + "}")
                except CCTokenVerifyError:
                    invalid_participant_list.append(k + " namespace: {" + namespace + "}")
        self.logger.info(f"CC - results from _verify_participants_tokens: {result}, {invalid_participant_list=}")
        return result, invalid_participant_list

    def _perform_cross_site_validation(self, fl_ctx: FLContext) -> bool:
        """Perform cross-site validation and shutdown system on failure.

        Returns:
            True if validation passed, False if validation failed (system will shutdown)
        """
        try:
            self.logger.info("Performing cross-site validation")

            # Collect fresh tokens from all sites
            all_tokens = self._collect_all_site_tokens(fl_ctx)

            if not all_tokens:
                self.logger.error("No tokens collected for validation")
                self._initiate_shutdown("No tokens collected for validation", fl_ctx)
                return False

            # Validate all tokens
            err = self._validate_participants_tokens(all_tokens)
            if err:
                self.logger.error(f"Cross-site validation failed: {err}")
                self._initiate_shutdown(f"Cross-site validation failed: {err}", fl_ctx)
                return False
            else:
                self.logger.info("Cross-site validation passed")
                return True

        except Exception as e:
            self.logger.exception(f"Exception in cross-site validation: {e}")
            self._initiate_shutdown(f"Exception in cross-site validation: {e}", fl_ctx)
            return False

    def _get_all_sites(self, fl_ctx: FLContext) -> list[str]:
        """Get list of all sites (server + participating clients), excluding admin clients.

        This method works differently depending on the context:
        - Server (ServerEngine): Gets all registered clients via engine.get_clients()
        - Client (ClientEngine): Dynamically requests current site list from server

        Admin clients are excluded from CC verification as they don't participate in FL jobs.

        Args:
            fl_ctx: FLContext

        Returns:
            List of site names (excluding admin clients).
        """
        engine = fl_ctx.get_engine()

        # Check if this is server engine
        if isinstance(engine, ServerEngineSpec):
            # Server side: get from engine
            all_sites = [self.site_name]
            clients = engine.get_clients()
            if clients:
                # Filter out admin clients
                all_sites.extend([client.name for client in clients if not is_valid_admin_client_name(client.name)])
            self.logger.info(f"Server: Found {len(all_sites)} sites (excluding admin clients): {all_sites}")
            return all_sites
        else:
            # Client side: dynamically request current site list from server
            all_sites = self._request_sites_from_server(fl_ctx)
            if all_sites:
                self.logger.info(f"Client: Received {len(all_sites)} sites from server: {all_sites}")
                return all_sites
            else:
                # Fallback: only self known
                self.logger.warning("Client: Failed to get sites from server, using self only")
                return [self.site_name]

    def _request_sites_from_server(self, fl_ctx: FLContext) -> list[str]:
        """Client side: Request current list of participating sites from server."""
        engine = fl_ctx.get_engine()
        cell = engine.get_cell()

        if not cell:
            self.logger.error("Cell not available")
            return []

        try:
            request_message = new_cell_message({}, {"requester": self.site_name})

            response = cell.send_request(
                target=FQCN.ROOT_SERVER,
                channel=CC_CHANNEL,
                topic=CC_TOPIC_GET_SITES,
                request=request_message,
                timeout=10.0,
                optional=True,
            )

            if response:
                return_code = response.get_header(MessageHeaderKey.RETURN_CODE)
                if return_code == F3ReturnCode.OK:
                    payload = response.payload
                    if isinstance(payload, dict):
                        sites = payload.get("sites")
                        if sites and isinstance(sites, list):
                            return sites

            return []

        except Exception as e:
            self.logger.exception(f"Error requesting sites from server: {e}")
            return []

    def _start_cross_site_validation(self, fl_ctx: FLContext):
        """Start periodic cross-site validation thread on ALL sites."""
        if self.cross_validation_thread is not None:
            self.logger.info("Cross-site validation already running")
            return

        self.cross_validation_thread = threading.Thread(
            target=self._cross_site_validation_loop, args=[fl_ctx], daemon=True, name="CCManager-CrossSiteValidation"
        )
        self.cross_validation_thread.start()
        self.logger.info(f"Started cross-site validation with interval {self.cross_validation_interval}s")

    def _stop_cross_site_validation(self):
        """Stop cross-site validation thread."""
        if self.cross_validation_thread is not None:
            self.logger.info("Stopping cross-site validation")
            self.cross_validation_thread = None

    def _cross_site_validation_loop(self, fl_ctx: FLContext):
        """Periodic cross-site validation - runs on ALL sites."""
        # Keep a reference to the current thread
        my_thread = threading.current_thread()

        # Add random jitter (0-20% of interval) to avoid thundering herd
        # This prevents all sites from validating at exactly the same time
        jitter = random.uniform(0, self.cross_validation_interval * 0.2)
        time.sleep(jitter)
        self.logger.info(f"Cross-site validation thread started with {jitter:.1f}s jitter")

        while self.cross_validation_thread is my_thread:
            time.sleep(self.cross_validation_interval)

            if self.cross_validation_thread is not my_thread:
                break

            # Use lock to prevent concurrent validation on the same site
            if not self.lock.acquire(blocking=False):
                self.logger.warning("Cross-site validation already in progress, skipping this cycle")
                continue

            try:
                self.logger.info(f"Site {self.site_name} triggering periodic cross-site validation")
                self._perform_cross_site_validation(fl_ctx)
            finally:
                self.lock.release()

    def _collect_all_site_tokens(self, fl_ctx: FLContext) -> dict[str, list[dict[str, str]]]:
        """Collect FRESH CC tokens from all participants using dedicated CC channel.

        This method:
        1. Requests fresh tokens from ALL other sites
        2. Returns all fresh tokens for validation

        This ensures all tokens are freshly generated and synchronized.
        """
        all_tokens = {}
        engine = fl_ctx.get_engine()
        cell = engine.get_cell()

        if cell is None:
            self.logger.error("Cell is not available, cannot collect tokens via CC channel")
            return {}

        # Step 1: Generate this site's token first
        all_tokens[self.site_name] = self._generate_fresh_tokens_for_validation()
        self.logger.info(f"Generated fresh token for this site: {self.site_name}")

        # Step 2: Get list of all sites (exclude self)
        all_site_names = self._get_all_sites(fl_ctx)
        other_sites = [site for site in all_site_names if site != self.site_name]

        if not other_sites:
            self.logger.info("No other sites for validation, only this site")
            return all_tokens

        # Step 3: Request fresh tokens from all other known sites
        self.logger.info(f"Requesting fresh tokens from {len(other_sites)} other sites: {other_sites}")

        # Send requests to all other sites
        request_message = new_cell_message({}, {"requester": self.site_name})

        for target_site in other_sites:
            try:
                # Determine target FQCN (this depends on your topology)
                # For simplicity, assume we can reach sites directly
                target_fqcn = self._get_site_fqcn(target_site)

                if not target_fqcn:
                    self.logger.warning(f"Cannot determine FQCN for site {target_site}, skipping")
                    continue

                response = cell.send_request(
                    target=target_fqcn,
                    channel=CC_CHANNEL,
                    topic=CC_TOPIC_REQUEST_TOKEN,
                    request=request_message,
                    timeout=10.0,
                    optional=True,
                )

                if response:
                    return_code = response.get_header(MessageHeaderKey.RETURN_CODE)
                    if return_code == F3ReturnCode.OK:
                        payload = response.payload
                        if isinstance(payload, dict):
                            site_name = payload.get("site_name")
                            cc_info = payload.get("cc_info")
                            if site_name and cc_info:
                                all_tokens[site_name] = cc_info
                                self.logger.info(f"Received fresh token from site {site_name}")
                            else:
                                self.logger.warning(f"Invalid payload from {target_site}: {payload}")
                        else:
                            self.logger.warning(f"Invalid response type from {target_site}: {type(payload)}")
                    else:
                        error_msg = response.get_header(MessageHeaderKey.ERROR, "unknown error")
                        self.logger.error(f"Failed to get token from {target_site}: {error_msg}")
                else:
                    self.logger.warning(f"No response from site {target_site}")

            except Exception as e:
                self.logger.exception(f"Error requesting token from {target_site}: {e}")

        self.logger.info(f"Collected fresh tokens from {len(all_tokens)} sites: {list(all_tokens.keys())}")
        return all_tokens

    def _get_site_fqcn(self, site_name: str) -> str:
        """Get the FQCN (Fully Qualified Cell Name) for a site.

        For clients, this would typically be the client's FQCN.
        For server, it's typically FQCN.ROOT_SERVER.
        """
        # Check if this is the server
        if site_name.lower().startswith("server"):
            return FQCN.ROOT_SERVER

        # For clients, we need to construct their FQCN
        # This depends on your network topology
        # Common pattern: client name is the FQCN
        return site_name

    def _initiate_shutdown(self, reason: str, fl_ctx: FLContext):
        """Shutdown this site due to CC validation failure."""
        self.logger.critical(f"Shutting down site {self.site_name} due to: {reason}")
        # Always use a thread for shutdown to prevent deadlocks and allow graceful cleanup
        threading.Thread(
            target=self._shutdown_system, args=[reason, fl_ctx], daemon=False, name="CCManager-Shutdown"
        ).start()

    def _register_cc_handlers(self, fl_ctx: FLContext):
        """Register handlers for dedicated CC validation channel."""
        engine = fl_ctx.get_engine()
        cell = engine.get_cell()

        if cell is None:
            self.logger.warning("Cell is not available, cannot register CC validation handlers")
            return

        # Register handler for token refresh requests
        cell.register_request_cb(
            channel=CC_CHANNEL, topic=CC_TOPIC_REQUEST_TOKEN, cb=self._handle_token_refresh_request
        )

        # Server-only: Register handler for site list requests
        if isinstance(engine, ServerEngineSpec):
            cell.register_request_cb(channel=CC_CHANNEL, topic=CC_TOPIC_GET_SITES, cb=self._handle_get_sites_request)
            self.logger.info(f"Registered server CC handlers on channel '{CC_CHANNEL}'")
        else:
            self.logger.info(f"Registered client CC handlers on channel '{CC_CHANNEL}'")

    def _generate_fresh_tokens_for_validation(self) -> list[dict[str, str]]:
        """Generate completely fresh tokens for validation request.

        This creates NEW tokens on-the-fly for each validation request.
        Note: Caller must hold lock to ensure thread-safety.

        Returns:
            List of fresh CC tokens ready for validation
        """
        fresh_tokens = []

        for issuer, expiration in self.cc_issuers.items():
            try:
                # Generate a brand new token for this specific request
                new_token = issuer.generate()
                namespace = issuer.get_namespace()

                if not new_token:
                    self.logger.error(f"Failed to generate token for namespace {namespace}")
                    continue

                fresh_tokens.append({CC_TOKEN: new_token, CC_NAMESPACE: namespace, CC_TOKEN_VALIDATED: False})

                self.logger.info(f"Generated fresh token for validation: namespace={namespace}")

            except CCTokenGenerateError as e:
                self.logger.error(f"Error generating token: {e}")
                # Continue with other issuers even if one fails
                continue

        return fresh_tokens

    def _handle_token_refresh_request(self, request):
        """Handle request from another site to generate and return fresh token.

        This is called when another site initiates cross-site validation.

        IMPORTANT: Multiple validation events can happen simultaneously, so we generate
        a FRESH token for EACH request (nonce-based tokens are single-use).
        """
        try:
            # Extract requester from payload (we sent it there)
            payload = request.payload if hasattr(request, "payload") else {}
            requester = payload.get("requester", "unknown") if isinstance(payload, dict) else "unknown"
            self.logger.info(f"Received token refresh request from {requester}")

            # Generate fresh tokens for THIS request
            cc_info = self._generate_fresh_tokens_for_validation()

            if not cc_info:
                self.logger.error("Failed to generate any tokens for validation request")
                return make_reply(F3ReturnCode.ERROR, "Failed to generate tokens", None)

            # Return fresh tokens
            payload = {"site_name": self.site_name, "cc_info": cc_info}

            self.logger.info(f"Sending {len(cc_info)} fresh CC token(s) for site {self.site_name} to {requester}")
            return make_reply(F3ReturnCode.OK, "", payload)

        except Exception as e:
            self.logger.exception(f"Error handling token refresh request: {e}")
            return make_reply(F3ReturnCode.ERROR, f"Failed to generate token: {str(e)}", None)

    def _handle_get_sites_request(self, request):
        """Server side: Handle request for current list of participating sites."""
        try:
            # Extract requester from payload
            payload = request.payload if hasattr(request, "payload") else {}
            requester = payload.get("requester", "unknown") if isinstance(payload, dict) else "unknown"
            self.logger.info(f"Received sites list request from {requester}")

            # Get current list of participating sites (excluding admin clients)
            all_sites = [self.site_name]  # Start with server

            if self.engine and isinstance(self.engine, ServerEngineSpec):
                clients = self.engine.get_clients()
                if clients:
                    # Filter out admin clients
                    all_sites.extend([client.name for client in clients if not is_valid_admin_client_name(client.name)])

            # Return sites list
            response_payload = {"sites": all_sites}

            self.logger.info(f"Sending {len(all_sites)} sites to {requester}: {all_sites}")
            return make_reply(F3ReturnCode.OK, "", response_payload)

        except Exception as e:
            self.logger.exception(f"Error handling get_sites request: {e}")
            return make_reply(F3ReturnCode.ERROR, f"Failed to get sites: {str(e)}", None)

    def _shutdown_system(self, reason: str, fl_ctx: FLContext):
        """Shuts down the entire NVFlare system due to CC validation failure."""
        engine = fl_ctx.get_engine()

        # Check if this is server or client
        if isinstance(engine, ServerEngineSpec):
            # Server side
            try:
                run_processes = engine.run_processes
                running_jobs = list(run_processes.keys())
                for job_id in running_jobs:
                    engine.job_runner.stop_run(job_id, fl_ctx)

                conn = Connection(app_ctx=engine, props={ConnProps.ADMIN_SERVER: engine.server.admin_server})
                cmd = TrainingCommandModule()
                args = ["shutdown", "all"]
                cmd.validate_command_targets(conn, args[1:])
                cmd.shutdown(conn, args)

                self.logger.error(f"CC system shutdown initiated from server! Reason: {reason}")
            except Exception as e:
                self.logger.exception(f"Error during server shutdown: {e}")
                # Force exit as last resort
                sys.exit(1)
        else:
            # Client side - just exit
            self.logger.critical(f"CC validation failed on client! Shutting down. Reason: {reason}")
            sys.exit(1)
