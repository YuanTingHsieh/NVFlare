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
"""Attempt receipt validation, independent of the launcher."""

import json

import pytest

from nvflare.private.fed.task_scope.protocol import (
    END_RUN,
    ERROR,
    IDLE,
    RECEIPT_FILE,
    STATUS,
    TASK_COMPLETE,
    read_receipt,
    write_receipt,
)


@pytest.mark.parametrize("status", [TASK_COMPLETE, IDLE, END_RUN])
def test_receipt_roundtrip_requires_matching_attempt(tmp_path, status):
    outcome = {STATUS: status, "task_id": "task-1" if status == TASK_COMPLETE else None}
    write_receipt(str(tmp_path), "attempt-1", outcome)
    assert read_receipt(str(tmp_path), "attempt-1") == dict(outcome, attempt="attempt-1")
    with pytest.raises(ValueError, match="stale"):
        read_receipt(str(tmp_path), "previous-attempt")
    with pytest.raises(FileExistsError):
        write_receipt(str(tmp_path), "attempt-1", outcome)


@pytest.mark.parametrize("outcome", [{STATUS: TASK_COMPLETE}, {STATUS: ERROR}, {STATUS: "unknown"}])
def test_receipt_rejects_incomplete_or_unrecognized_outcome(tmp_path, outcome):
    with pytest.raises(ValueError):
        write_receipt(str(tmp_path), "attempt-1", outcome)
    assert not (tmp_path / RECEIPT_FILE).exists()


def test_receipt_rejects_partial_json_and_symlink(tmp_path):
    path = tmp_path / RECEIPT_FILE
    path.write_text('{"attempt":')
    with pytest.raises(ValueError):
        read_receipt(str(tmp_path), "attempt-1")
    path.unlink()
    target = tmp_path / "other.json"
    target.write_text(json.dumps({"attempt": "attempt-1", STATUS: IDLE}))
    path.symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        read_receipt(str(tmp_path), "attempt-1")
