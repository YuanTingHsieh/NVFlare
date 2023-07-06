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
Centralized cifar10 training with DDP
"""

import argparse
import os

import torch
import torch.distributed as dist
from simple_network import SimpleNetwork
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import SGD
from torch.utils.data import RandomSampler
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor

from nvflare.app_common.abstract.fl_model import FLModel, ParamsType
from nvflare.app_common.model_exchange.model_exchanger import ModelExchanger


class Cifar10Trainer:
    def __init__(self, dataset_root: str, lr=0.01, epochs=5):
        self._lr = lr
        self._epochs = epochs
        # Training setup
        self.raw_model = SimpleNetwork()

        self.local_rank = 0
        self.workers = 2  # for DataLoader

        self.device = "cuda:0"
        self.raw_model.to(self.device)

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
            self._train_dataset, sampler=self._train_sampler, batch_size=4, num_workers=self.workers, pin_memory=True
        )

        self.total_iterations = len(self._train_loader) * self._epochs

    def train(self, log_interval=100, weights=None):
        if weights is not None:
            # Set the model weights
            self.raw_model.load_state_dict(state_dict=weights)

        # Basic training
        model = DDP(self.raw_model, device_ids=[self.local_rank], output_device=self.local_rank)
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

    local_rank = int(os.environ["LOCAL_RANK"])

    input_weights = None
    if local_rank == 0:
        data_exchanger = FLModelExchanger(pipe_role="y")
        data_exchanger.initialize(args.data_exchange_root)
        input_fl_model = data_exchanger.get_model(args.from_nvflare)
        input_weights = {k: torch.as_tensor(v) for k, v in input_fl_model.params.items()}

    backend = "gloo"  # nccl, mpi, gloo
    dist.init_process_group(backend=backend)
    dist.broadcast_object_list([input_weights], src=0)

    trainer = Cifar10Trainer(args.dataset_root, lr, epochs)
    trainer.train(weights=input_weights)

    dist.barrier()

    if local_rank == 0:
        output_weights = trainer.raw_model.state_dict()
        output_fl_model = FLModel(
            params={k: v.cpu().numpy() for k, v in output_weights.items()}, params_type=ParamsType.WEIGHTS
        )
        data_exchanger.put_model(args.to_nvflare, model=output_fl_model)


if __name__ == "__main__":
    main()
