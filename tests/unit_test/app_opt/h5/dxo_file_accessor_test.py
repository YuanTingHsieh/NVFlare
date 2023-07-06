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

import os
import tempfile

import numpy as np
import pytest

from nvflare.apis.dxo import DXO, DataKind
from nvflare.app_common.abstract.fl_model import FLModel, TransferType
from nvflare.app_common.utils.fl_model_utils import FLModelUtils
from nvflare.app_opt.h5.dxo_file_accessor import H5DXOFileAccessor

TEST_CASES = [
    {"a": 1, "b": 3},
    {},
    {"abc": [1, 2, 3], "d": [4, 5]},
    {"abc": (1, 2, 3), "d": (4, 5)},
    {"hello": b"a string", "cool": 6},
    {f"layer{i}": np.random.rand(256, 256) for i in range(5)},
]


class TestH5DXOFileAccessor:
    def test_read_write_str(self):
        data = b"hello moto"
        with tempfile.TemporaryDirectory() as root_dir:
            x = H5DXOFileAccessor()
            file_path = os.path.join(root_dir, "test_file")
            x.write(data, file_path)
            result = x.read(file_path)
            assert result == data

    @pytest.mark.parametrize("weights", TEST_CASES)
    def test_read_write_dxo(self, weights):
        with tempfile.TemporaryDirectory() as root_dir:
            dxo = DXO(data=weights, data_kind=DataKind.WEIGHT_DIFF)

            x = H5DXOFileAccessor()
            file_path = os.path.join(root_dir, "test_file")
            x.write(dxo, file_path)
            result_dxo = x.read(file_path)

            for k, v in result_dxo.data.items():
                np.testing.assert_array_equal(weights[k], v)
            assert result_dxo.data_kind == dxo.data_kind

    @pytest.mark.parametrize("weights", TEST_CASES)
    def test_read_write_fl_model(self, weights):
        with tempfile.TemporaryDirectory() as root_dir:
            fl_model = FLModel(model=weights, transfer_type=TransferType.MODEL)
            fl_model_dxo = FLModelUtils.to_dxo(fl_model)
            x = H5DXOFileAccessor()
            file_path = os.path.join(root_dir, "test_file")
            x.write(fl_model_dxo, file_path)
            result_dxo = x.read(file_path)
            result_fl_model = FLModelUtils.from_dxo(result_dxo)
            for k, v in result_fl_model.model.items():
                np.testing.assert_array_equal(weights[k], v)
            assert fl_model.transfer_type == result_fl_model.transfer_type
