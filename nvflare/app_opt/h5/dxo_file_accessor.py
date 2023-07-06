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

import h5py

from nvflare.apis.dxo import DXO
from nvflare.app_opt.h5.utils import load_dict_contents_from_group, save_dict_contents_to_group
from nvflare.fuel.utils.pipe.file_accessor import FileAccessor

DEFAULT_KEY = "__default__"


class H5DXOFileAccessor(FileAccessor):
    def __init__(self):
        os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
        super().__init__()

    def write(self, data, file_path: str):
        with h5py.File(file_path, "w") as file:
            if not isinstance(data, DXO):
                file.create_dataset(DEFAULT_KEY, data=data)
            else:
                file.create_dataset("__data_kind__", data=data.data_kind.encode("utf-8"))

                save_dict_contents_to_group(file, "/__data__/", data.data)
                if data.meta:
                    save_dict_contents_to_group(file, "/__meta__/", data.meta)

    def read(self, file_path: str):
        with h5py.File(file_path, "r") as file:
            if len(file.keys()) == 1 and list(file.keys())[0] == DEFAULT_KEY:
                data = file[DEFAULT_KEY][()]
            else:

                data_kind = file["__data_kind__"][()].decode("utf-8")
                data_dict = load_dict_contents_from_group(file, "/__data__/")
                try:
                    meta_dict = load_dict_contents_from_group(file, "/__meta__/")
                except Exception:
                    meta_dict = None

                data = DXO(data_kind=data_kind, data=data_dict, meta=meta_dict)
        return data
