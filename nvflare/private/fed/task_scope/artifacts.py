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
"""Durable, attempt-local input/result handoffs between phased CJs.

The manifest is published last. Its presence commits a complete payload, not a
server result ACK. The attempt directory must be on storage shared by the site's
CPU and GPU allocations and retained until result submission is acknowledged.
"""

import hashlib
import json
import os
import stat
import tempfile

from nvflare.apis.shareable import ReservedHeaderKey, Shareable
from nvflare.fuel.utils import fobs
from nvflare.fuel.utils.fobs.decomposers.via_downloader import _GRAPH_LEAF_TYPES, LazyDownloadRef, _iter_graph_children

_VERSION = 1
_MAX_MANIFEST_SIZE = 65536
_CHUNK_SIZE = 1024 * 1024


def _validate_identity(attempt, job_id, kind):
    if kind not in ("input", "result"):
        raise ValueError("artifact kind must be input or result")
    for name, value in (("attempt", attempt), ("job_id", job_id)):
        _validate_text(name, value)


def _validate_text(name, value):
    if not isinstance(value, str) or not value or len(value) > 4096:
        raise ValueError(f"invalid artifact {name}")


def _check_eager_data(data):
    if not isinstance(data, Shareable):
        raise TypeError("artifact data must be a Shareable")
    pending = [data]
    visited = set()
    while pending:
        value = pending.pop()
        if isinstance(value, Shareable) and value.get_header(ReservedHeaderKey.PASS_THROUGH):
            raise ValueError("artifact requires eager data; pass-through mode is unsupported")
        if isinstance(value, LazyDownloadRef) or type(value).__module__ == "nvflare.app_opt.pt.lazy_tensor_dict":
            raise ValueError("artifact requires eager data; lazy/pass-through references are unsupported")
        if isinstance(value, _GRAPH_LEAF_TYPES) or id(value) in visited:
            continue
        visited.add(id(value))
        pending.extend(_iter_graph_children(value))


def _fingerprint(stream):
    digest = hashlib.sha256()
    size = 0
    while chunk := stream.read(_CHUNK_SIZE):
        digest.update(chunk)
        size += len(chunk)
    return {"size": size, "sha256": digest.hexdigest()}


def _sync_directory(directory):
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _publish(directory, name, writer):
    """Publish exclusively: another writer cannot replace a completed artifact."""
    fd, temporary = tempfile.mkstemp(prefix=f".{name}.", dir=directory)
    try:
        with os.fdopen(fd, "w+b") as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, os.path.join(directory, name))
        _sync_directory(directory)
    finally:
        os.unlink(temporary)


def _open_regular(path):
    # Do not follow a replaced payload/manifest to another job's files.
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        os.close(fd)
        raise ValueError("artifact must be a regular file")
    return os.fdopen(fd, "rb")


def write_artifact(directory, attempt, job_id, kind, task_name, task_id, data: Shareable):
    """Commit eager data without a live Cell or a whole-artifact bytes buffer.

    Native FOBS includes file-datum contents in the stream; it does not persist
    their source paths. Tensor decomposers may still allocate per-tensor buffers.
    A failed write must use a new attempt rather than overwrite partial state.
    """
    _validate_identity(attempt, job_id, kind)
    _validate_text("task_name", task_name)
    _validate_text("task_id", task_id)
    _check_eager_data(data)
    manifest = {
        "version": _VERSION,
        "attempt": attempt,
        "job_id": job_id,
        "kind": kind,
        "task_name": task_name,
        "task_id": task_id,
    }

    def write_payload(stream):
        fobs.dump_to_stream(data, stream, max_value_size=_CHUNK_SIZE, fobs_ctx={"native": True})
        stream.flush()
        stream.seek(0)
        manifest.update(_fingerprint(stream))

    _publish(directory, f"{kind}.fobs", write_payload)
    encoded_manifest = json.dumps(manifest).encode("utf-8")
    if len(encoded_manifest) > _MAX_MANIFEST_SIZE:
        raise ValueError("artifact manifest too large")
    _publish(directory, f"{kind}.json", lambda stream: stream.write(encoded_manifest))


def read_artifact(directory, attempt, job_id, kind):
    """Validate the committed manifest and payload before deserializing it."""
    _validate_identity(attempt, job_id, kind)
    with _open_regular(os.path.join(directory, f"{kind}.json")) as stream:
        encoded = stream.read(_MAX_MANIFEST_SIZE + 1)
    if len(encoded) > _MAX_MANIFEST_SIZE:
        raise ValueError("artifact manifest too large")
    manifest = json.loads(encoded)
    expected = {"version": _VERSION, "attempt": attempt, "job_id": job_id, "kind": kind}
    if (
        not isinstance(manifest, dict)
        or type(manifest.get("version")) is not int
        or any(manifest.get(key) != value for key, value in expected.items())
    ):
        raise ValueError("invalid or stale artifact manifest")
    for name in ("task_name", "task_id"):
        _validate_text(name, manifest.get(name))
    with _open_regular(os.path.join(directory, f"{kind}.fobs")) as stream:
        fingerprint = _fingerprint(stream)
        if any(manifest.get(key) != value for key, value in fingerprint.items()):
            raise ValueError("artifact payload size or checksum mismatch")
        stream.seek(0)
        data = fobs.load_from_stream(stream, fobs_ctx={"native": True})
    _check_eager_data(data)
    return {"task_name": manifest["task_name"], "task_id": manifest["task_id"], "data": data}
