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

"""
Centralized cifar10 training
"""

import argparse
import os

import h5py
import torch
from simple_network import SimpleNetwork
from torch import nn
from torch.optim import SGD
from torch.utils.data import RandomSampler
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor
from utils import load_dict_contents_from_group, save_dict_contents_to_group


class Cifar10Trainer:
    def __init__(self, dataset_root: str, lr=0.01, epochs=5):
        self._lr = lr
        self._epochs = epochs

        # Training setup
        self.model = SimpleNetwork()
        self.device = "cuda:0"
        self.model.to(self.device)

        # Create Cifar10 dataset for training.
        transforms = Compose(
            [
                ToTensor(),
                Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        self._train_dataset = CIFAR10(root=dataset_root, transform=transforms, download=False, train=True)
        self._train_sampler = RandomSampler(self._train_dataset)
        self._train_loader = DataLoader(
            self._train_dataset, sampler=self._train_sampler, batch_size=4, num_workers=2, pin_memory=True
        )

        self.total_iterations = len(self._train_loader) * self._epochs

    def train(self, log_interval=100, weights=None):
        if weights is not None:
            # Set the model weights
            self.model.load_state_dict(state_dict=weights)

        # Basic training
        model = self.model
        loss = nn.CrossEntropyLoss()
        optimizer = SGD(model.parameters(), lr=self._lr, momentum=0.9)
        model.train()
        for epoch in range(self._epochs):
            running_loss = 0.0
            for i, batch in enumerate(self._train_loader):
                images, labels = batch[0].to(self.device), batch[1].to(self.device)
                optimizer.zero_grad()

                predictions = model(images)
                cost = loss(predictions, labels)
                cost.backward()
                optimizer.step()

                running_loss += cost.cpu().detach().numpy() / images.size()[0]
                if i != 0 and i % log_interval == 0:
                    avg_loss = running_loss / log_interval
                    print(f"Epoch: {epoch}/{self._epochs}, Iteration: {i}, " f"Loss: {avg_loss}")
                    running_loss = 0.0
                    break


def get_model_data(args):
    pipe_path = os.path.join(args.data_exchange_root, "pipe", "y")
    input_file = os.path.join(pipe_path, f"REQ.H5DXOFileAccessor.{args.from_nvflare}")

    with h5py.File(input_file, "r") as file:
        data_kind = file["__data_kind__"][()].decode("utf-8")
        data_dict = load_dict_contents_from_group(file, "/__data__/")
        try:
            meta_dict = load_dict_contents_from_group(file, "/__meta__/")
        except Exception:
            meta_dict = None

    return data_kind, data_dict, meta_dict


def write_output_data(args, data_kind, output_weights, meta_dict):
    pipe_path = os.path.join(args.data_exchange_root, "pipe", "x")
    output_file = os.path.join(pipe_path, f"REQ.H5DXOFileAccessor.{args.to_nvflare}")
    with h5py.File(output_file, "w") as file:
        file.create_dataset("__data_kind__", data=data_kind.encode("utf-8"))

        save_dict_contents_to_group(file, "/__data__/", output_weights)
        if meta_dict:
            save_dict_contents_to_group(file, "/__meta__/", meta_dict)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default="0.01", help="learning rate")
    parser.add_argument("--epochs", type=int, default=5, help="epochs")
    parser.add_argument("--dataset_root", type=str, default="/tmp/nvflare/cifar10")
    parser.add_argument("--data_exchange_root", type=str, default="./")
    parser.add_argument("--from_nvflare", type=str, default="from_nvflare")
    parser.add_argument("--to_nvflare", type=str, default="to_nvflare")
    args = parser.parse_args()

    lr = args.lr
    epochs = args.epochs
    input_data_kind, input_data_dict, input_meta_dict = get_model_data(args)
    input_weights = {k: torch.as_tensor(v) for k, v in input_data_dict["model"].items()}

    trainer = Cifar10Trainer(args.dataset_root, lr, epochs)
    trainer.train(weights=input_weights)
    output_weights = {k: v.cpu().numpy() for k, v in trainer.model.state_dict().items()}
    input_data_dict["model"] = output_weights

    write_output_data(args, input_data_kind, input_data_dict, input_meta_dict)


if __name__ == "__main__":
    main()
