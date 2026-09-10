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

import errno
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import ReservedHeaderKey, Shareable
from nvflare.apis.utils.decomposers.flare_decomposers import ContextDecomposer
from nvflare.app_common.decomposers.numpy_decomposers import NumpyArrayDecomposer
from nvflare.fuel.utils import fobs
from nvflare.fuel.utils.fobs.datum import Datum
from nvflare.fuel.utils.fobs.decomposer import DictDecomposer
from nvflare.fuel.utils.fobs.decomposers.via_downloader import LazyDownloadRef
from nvflare.private.fed.task_scope import artifacts


@pytest.fixture(autouse=True)
def register_decomposers():
    fobs.register(DictDecomposer(Shareable))
    fobs.register(NumpyArrayDecomposer)
    fobs.register(ContextDecomposer)


def _write(directory, data=None, kind="result", **kwargs):
    values = dict(attempt="attempt-1", job_id="job-1", kind=kind, task_name="train", task_id="task-1")
    values.update(kwargs)
    artifacts.write_artifact(directory, data=data if data is not None else Shareable({"value": 42}), **values)


def _read(directory, kind="result", **kwargs):
    values = dict(attempt="attempt-1", job_id="job-1", kind=kind)
    values.update(kwargs)
    return artifacts.read_artifact(directory, **values)


@pytest.mark.parametrize("kind", ["input", "result"])
def test_eager_model_roundtrip_has_no_live_producer_dependency(tmp_path, kind):
    # Larger than FOBS's datum threshold; force native tensor serialization even
    # though the same decomposer normally supports Cell-based downloads.
    weights = np.arange(3 * 1024 * 1024, dtype=np.float32)
    data = Shareable({"weights": {"layer.weight": weights}, "metrics": {"accuracy": 0.9}})
    data.set_header("round", 2)
    expected = weights.copy()
    with patch.object(NumpyArrayDecomposer, "to_downloadable", side_effect=AssertionError("must not use a Cell")):
        _write(tmp_path, data, kind)
        weights.fill(-1)
        del data
        result = _read(tmp_path, kind)
    assert result["task_name"] == "train"
    assert result["task_id"] == "task-1"
    assert result["data"].get_header("round") == 2
    np.testing.assert_array_equal(result["data"]["weights"]["layer.weight"], expected)
    assert (tmp_path / f"{kind}.fobs").stat().st_mode & 0o777 == 0o600
    assert (tmp_path / f"{kind}.json").stat().st_mode & 0o777 == 0o600


def test_file_datums_include_contents_instead_of_producer_paths(tmp_path, monkeypatch):
    source = tmp_path / "producer.dat"
    source.write_bytes(b"weights saved by the GPU process")
    _write(tmp_path, Shareable({"weights": Datum.file_datum(str(source))}))
    source.unlink()
    restored = tmp_path / "restored"
    restored.mkdir()
    monkeypatch.setattr("nvflare.fuel.utils.fobs.lobs.get_datum_dir", lambda: str(restored))
    result = _read(tmp_path)["data"]["weights"]
    assert Path(result.value).parent == restored
    assert Path(result.value).read_bytes() == b"weights saved by the GPU process"


def test_input_and_result_are_distinct_write_once_commits(tmp_path):
    _write(tmp_path, kind="input")
    _write(tmp_path, Shareable({"value": 43}))
    with pytest.raises(FileExistsError):
        _write(tmp_path, Shareable({"value": 99}))
    assert _read(tmp_path, "input")["data"]["value"] == 42
    assert _read(tmp_path)["data"]["value"] == 43
    assert not list(tmp_path.glob(".*"))


def test_runtime_peer_context_and_task_headers_survive_phase_handoff(tmp_path):
    peer_context = FLContext()
    peer_context.set_prop(FLContextKey.CURRENT_RUN, "job-1", private=False)
    peer_context.set_prop("round", 3, private=False)
    data = Shareable({"weight": 2})
    data.set_peer_context(peer_context)
    data.set_peer_props({"round": 3})
    data.add_cookie("round", 3)
    _write(tmp_path, data, kind="input")
    restored = _read(tmp_path, kind="input")["data"]
    assert restored.get_peer_context() is not peer_context
    assert restored.get_peer_context().get_job_id() == "job-1"
    assert restored.get_peer_context().get_prop("round") == 3
    assert restored.get_peer_props()["round"] == 3
    assert restored.get_cookie("round") == 3


@pytest.mark.parametrize("changes", [{"attempt": "other"}, {"job_id": "other"}, {"kind": "input"}, {"version": 2}])
def test_rejects_stale_or_mismatched_manifest_before_decode(tmp_path, changes):
    _write(tmp_path)
    path = tmp_path / "result.json"
    manifest = json.loads(path.read_text())
    manifest.update(changes)
    path.write_text(json.dumps(manifest))
    with patch.object(fobs, "load_from_stream", side_effect=AssertionError("must not decode")):
        with pytest.raises(ValueError, match="stale"):
            _read(tmp_path)


@pytest.mark.parametrize("corruption", [b"bad payload", None])
def test_rejects_corrupt_or_truncated_payload_before_decode(tmp_path, corruption):
    _write(tmp_path)
    path = tmp_path / "result.fobs"
    data = path.read_bytes()
    path.write_bytes(corruption if corruption else data[:-1])
    with patch.object(fobs, "load_from_stream", side_effect=AssertionError("must not decode")):
        with pytest.raises(ValueError, match="checksum"):
            _read(tmp_path)


@pytest.mark.parametrize("content", [b'{"attempt":', b"x" * 65537])
def test_partial_or_oversized_manifest_is_not_a_commit(tmp_path, content):
    (tmp_path / "result.json").write_bytes(content)
    with pytest.raises(ValueError):
        _read(tmp_path)


def test_serialization_failure_publishes_nothing(tmp_path):
    def fail_after_write(data, stream, **kwargs):
        stream.write(b"partial")
        raise OSError(errno.ENOSPC, "disk full")

    with patch.object(fobs, "dump_to_stream", side_effect=fail_after_write):
        with pytest.raises(OSError, match="disk full"):
            _write(tmp_path)
    assert not list(tmp_path.iterdir())
    with pytest.raises(FileNotFoundError):
        _read(tmp_path)


def test_manifest_publication_failure_does_not_expose_uncommitted_payload(tmp_path):
    real_publish = artifacts._publish

    def publish(directory, name, writer):
        if name.endswith(".json"):
            raise OSError(errno.EACCES, "cannot commit")
        return real_publish(directory, name, writer)

    with patch.object(artifacts, "_publish", side_effect=publish):
        with pytest.raises(OSError, match="cannot commit"):
            _write(tmp_path)
    assert (tmp_path / "result.fobs").exists()
    with pytest.raises(FileNotFoundError):
        _read(tmp_path)
    with pytest.raises(FileExistsError):
        _write(tmp_path)


@pytest.mark.parametrize("filename", ["result.json", "result.fobs"])
def test_rejects_symlink_at_read_boundary(tmp_path, filename):
    _write(tmp_path)
    path = tmp_path / filename
    original = tmp_path / "other-job"
    path.rename(original)
    path.symlink_to(original)
    with pytest.raises(OSError):
        _read(tmp_path)


@pytest.mark.parametrize("kind", ["../result", "other", "", None])
def test_kind_cannot_select_an_arbitrary_path(tmp_path, kind):
    with pytest.raises(ValueError, match="kind"):
        _write(tmp_path, kind=kind)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("field", ["attempt", "job_id", "task_name", "task_id"])
def test_requires_full_task_identity(tmp_path, field):
    with pytest.raises(ValueError, match=field):
        _write(tmp_path, **{field: ""})
    assert not list(tmp_path.iterdir())


def test_rejects_lazy_downloads_nested_inside_model_payload(tmp_path):
    data = Shareable({"weights": {"layer": [LazyDownloadRef("site-1.job", "transaction", "tensor")]}})
    with pytest.raises(ValueError, match="eager"):
        _write(tmp_path, data)
    assert not list(tmp_path.iterdir())


def test_rejects_pass_through_mode_even_without_lazy_refs(tmp_path):
    data = Shareable({"weight": 2})
    data.set_header(ReservedHeaderKey.PASS_THROUGH, True)
    with pytest.raises(ValueError, match="pass-through mode"):
        _write(tmp_path, data)
    assert not list(tmp_path.iterdir())


def test_rejects_lazy_tensor_file_reference_without_importing_torch(tmp_path):
    lazy_type = type("_LazyRef", (), {"__module__": "nvflare.app_opt.pt.lazy_tensor_dict"})
    data = Shareable({"weights": lazy_type()})
    with pytest.raises(ValueError, match="eager"):
        _write(tmp_path, data)
    assert not list(tmp_path.iterdir())
