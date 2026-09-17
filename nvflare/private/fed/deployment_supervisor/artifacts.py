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
"""Manifest-last durable handoff shared by the supervisor and task worker."""

import errno
import hashlib
import json
import os
import stat
import tempfile
import time

from nvflare.apis.shareable import ReservedHeaderKey, Shareable
from nvflare.fuel.utils import fobs
from nvflare.fuel.utils.fobs.decomposers.via_downloader import _GRAPH_LEAF_TYPES, LazyDownloadRef, _iter_graph_children

_VERSION = 1
_CHUNK_SIZE = 1024 * 1024
_MAX_MANIFEST = 65536
_OPEN_RETRY_TIMEOUT = 5.0
_OPEN_RETRY_INTERVAL = 0.05


def _validate_text(name, value):
    if not isinstance(value, str) or not value or len(value) > 4096:
        raise ValueError(f"invalid artifact {name}")


def _open_regular(path):
    deadline = time.monotonic() + _OPEN_RETRY_TIMEOUT
    while True:
        try:
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            break
        except OSError as e:
            if e.errno not in (errno.EAGAIN, errno.EWOULDBLOCK) or time.monotonic() >= deadline:
                raise
            time.sleep(_OPEN_RETRY_INTERVAL)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError("artifact must be a regular file")
        # O_NONBLOCK prevents an attacker-controlled FIFO from blocking open().
        # Once fstat proves this is a regular file, restore blocking I/O so a
        # shared filesystem cannot surface transient EAGAIN during reads.
        os.set_blocking(fd, True)
    except BaseException:
        os.close(fd)
        raise
    return os.fdopen(fd, "rb")


def _fingerprint(stream):
    digest = hashlib.sha256()
    size = 0
    while chunk := stream.read(_CHUNK_SIZE):
        digest.update(chunk)
        size += len(chunk)
    return size, digest.hexdigest()


def _require_eager(data):
    pending = [data]
    visited = set()
    while pending:
        value = pending.pop()
        if isinstance(value, Shareable) and value.get_header(ReservedHeaderKey.PASS_THROUGH):
            raise ValueError("durable handoff requires eager data; pass-through is unsupported")
        if isinstance(value, LazyDownloadRef) or type(value).__module__ == "nvflare.app_opt.pt.lazy_tensor_dict":
            raise ValueError("durable handoff requires eager data; lazy references are unsupported")
        if isinstance(value, _GRAPH_LEAF_TYPES) or id(value) in visited:
            continue
        visited.add(id(value))
        pending.extend(_iter_graph_children(value))


def _publish_exclusive(directory, name, writer):
    fd, temporary = tempfile.mkstemp(prefix=f".{name}.", dir=directory)
    try:
        with os.fdopen(fd, "w+b") as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, os.path.join(directory, name))
        dir_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    finally:
        os.unlink(temporary)


def write_artifact(directory, *, attempt, job_id, task_id, task_name, kind, data):
    """Write immutable FOBS payload first and its commit manifest last."""
    for name, value in (("attempt", attempt), ("job_id", job_id), ("task_id", task_id), ("task_name", task_name)):
        _validate_text(name, value)
    if kind not in ("input", "result"):
        raise ValueError("artifact kind must be input or result")
    if not isinstance(data, Shareable):
        raise TypeError("artifact data must be a Shareable")
    _require_eager(data)
    os.makedirs(directory, mode=0o700, exist_ok=True)
    manifest = {
        "version": _VERSION,
        "attempt": attempt,
        "job_id": job_id,
        "task_id": task_id,
        "task_name": task_name,
        "kind": kind,
    }

    def write_payload(stream):
        fobs.dump_to_stream(data, stream, max_value_size=_CHUNK_SIZE, fobs_ctx={"native": True})
        stream.flush()
        stream.seek(0)
        manifest["size"], manifest["sha256"] = _fingerprint(stream)

    _publish_exclusive(directory, f"{kind}.fobs", write_payload)
    encoded = json.dumps(manifest, sort_keys=True).encode("utf-8")
    if len(encoded) > _MAX_MANIFEST:
        raise ValueError("artifact manifest too large")
    _publish_exclusive(directory, f"{kind}.json", lambda stream: stream.write(encoded))


def read_artifact(directory, *, attempt, job_id, task_id, kind):
    """Read only a complete artifact whose identity and digest match."""
    with _open_regular(os.path.join(directory, f"{kind}.json")) as stream:
        encoded = stream.read(_MAX_MANIFEST + 1)
    if len(encoded) > _MAX_MANIFEST:
        raise ValueError("artifact manifest too large")
    manifest = json.loads(encoded)
    expected = {
        "version": _VERSION,
        "attempt": attempt,
        "job_id": job_id,
        "task_id": task_id,
        "kind": kind,
    }
    if not isinstance(manifest, dict) or any(manifest.get(k) != v for k, v in expected.items()):
        raise ValueError("invalid or stale artifact manifest")
    _validate_text("task_name", manifest.get("task_name"))
    with _open_regular(os.path.join(directory, f"{kind}.fobs")) as stream:
        size, digest = _fingerprint(stream)
        if size != manifest.get("size") or digest != manifest.get("sha256"):
            raise ValueError("artifact payload size or checksum mismatch")
        stream.seek(0)
        data = fobs.load_from_stream(stream, fobs_ctx={"native": True})
    if not isinstance(data, Shareable):
        raise TypeError("artifact data must be a Shareable")
    _require_eager(data)
    return manifest["task_name"], data
