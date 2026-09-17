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

import hashlib
import time

from nvflare.apis.dxo import from_shareable
from nvflare.apis.filter import Filter


class SlowResultFilter(Filter):
    """CPU-CJ publication delay used to verify that compute has already settled."""

    def __init__(self, delay: float):
        super().__init__()
        self.delay = delay

    def process(self, shareable, fl_ctx):
        time.sleep(self.delay)
        shareable.set_header("architecture_b_publication_delay", self.delay)
        return shareable


class SlowUploadVerificationFilter(Filter):
    """Verify that the committed G05 transport bytes reached the server."""

    def __init__(self, expected_mib: int):
        super().__init__()
        self.expected_mib = expected_mib
        self.expected_bytes = expected_mib * 1024 * 1024

    def process(self, shareable, fl_ctx):
        dxo = from_shareable(shareable)
        payload = dxo.meta.pop("architecture_b_g05_payload", None)
        declared_bytes = dxo.meta.pop("architecture_b_g05_payload_bytes", None)
        declared_digest = dxo.meta.pop("architecture_b_g05_payload_sha256", None)
        if not isinstance(payload, bytes):
            raise ValueError("missing Architecture B G05 byte payload")
        digest = hashlib.sha256(payload).hexdigest()
        if len(payload) != self.expected_bytes or declared_bytes != len(payload) or declared_digest != digest:
            raise ValueError("Architecture B G05 byte payload failed size/digest verification")
        shareable.set_header("architecture_b_g05_payload_verified", True)
        shareable.set_header("architecture_b_g05_payload_bytes", len(payload))
        shareable.set_header("architecture_b_g05_payload_sha256", digest)
        dxo.update_shareable(shareable)
        self.log_info(fl_ctx, f"verified Architecture B G05 transport payload: {len(payload)} bytes, sha256={digest}")
        return shareable
