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

import h5py
import numpy as np

utf8_type = h5py.string_dtype("utf-8", 256)


def save_dict_contents_to_group(h5file, path: str, dic: dict):
    if not isinstance(dic, dict):
        raise ValueError("must provide a dictionary")
    if not isinstance(path, str):
        raise ValueError("path must be a string")
    if not isinstance(h5file, h5py.File):
        raise ValueError("must be an open h5py file")
    if not dic:
        h5file.create_group(path)

    # save items to the hdf5 file
    for key, item in dic.items():
        key = str(key)
        if isinstance(item, (list, tuple)):
            item = np.array(item)
        if not isinstance(key, str):
            raise ValueError("dict keys must be strings to save to hdf5")

        if isinstance(item, str):
            h5file[path + key] = np.array(item.encode("utf-8"), dtype=utf8_type)
        elif isinstance(item, bytes):
            h5file[path + key] = item
        elif isinstance(item, (np.int64, np.float64, float, np.float32, int)):
            h5file[path + key] = item
            if not h5file[path + key][()] == item:
                raise ValueError("The data representation in the HDF5 file does not match the original dict.")
        elif isinstance(item, np.ndarray):
            try:
                h5file[path + key] = item
            except:
                item = np.array(item).astype("|S9")
                h5file[path + key] = item
            if not np.array_equal(h5file[path + key][()], item):
                raise ValueError("The data representation in the HDF5 file does not match the original dict.")
        elif isinstance(item, dict):
            save_dict_contents_to_group(h5file, path + key + "/", item)
        else:
            raise ValueError(f"Cannot save data: key: {key} type: {type(item)}.")


def load_dict_contents_from_group(h5file, path: str):
    dic = {}
    for key, item in h5file[path].items():
        if isinstance(item, h5py.Dataset):
            if item.dtype == utf8_type:
                dic[key] = item[()].decode("utf-8")
            else:
                dic[key] = item[()]
        elif isinstance(item, h5py.Group):
            dic[key] = load_dict_contents_from_group(h5file, path + key + "/")
    return dic
