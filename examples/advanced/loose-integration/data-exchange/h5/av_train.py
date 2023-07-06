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
Centralized CIFAR10 training with DDP
"""

import argparse
import os
import time

import torch
import torch.distributed as dist
from nv_dxi import NVDXFUtils
from simple_network import SimpleNetwork
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import SGD
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor


class Cifar10Trainer:
    def __init__(self, local_rank, lr=0.01, epochs=5):
        self._lr = lr
        self._epochs = epochs
        # Training setup
        self.raw_model = SimpleNetwork()

        self.device = "cuda:0"
        self.raw_model.to(self.device)

        # Create CIFAR 10 dataset for training.
        transforms = Compose(
            [
                ToTensor(),
                Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        self._train_dataset = CIFAR10(root="/tmp/nvflare/cifar10", transform=transforms, download=False, train=True)
        self._train_sampler = DistributedSampler(self._train_dataset, rank=local_rank)
        self._train_loader = DataLoader(
            self._train_dataset, sampler=self._train_sampler, batch_size=4, num_workers=0, pin_memory=True
        )

        self.total_iterations = len(self._train_loader) * self._epochs

    def train(self, dxf_util, local_rank: int, log_interval: int = 100, weights=None):
        print(f"rank {local_rank} starts training")
        if weights is not None:
            # Set the model weights
            self.raw_model.load_state_dict(state_dict=weights)

        # Basic training
        model = DDP(self.raw_model, device_ids=[0], output_device=0)
        loss = nn.CrossEntropyLoss()
        optimizer = SGD(model.parameters(), lr=self._lr, momentum=0.9)
        model.train()
        for epoch in range(self._epochs):
            running_loss = 0.0
            self._train_sampler.set_epoch(epoch)
            for i, batch in enumerate(self._train_loader):
                images, labels = batch[0].to(self.device), batch[1].to(self.device)
                optimizer.zero_grad()

                predictions = model(images)
                cost = loss(predictions, labels)
                cost.backward()
                optimizer.step()

                running_loss += cost.cpu().detach().numpy() / images.size()[0]
                if local_rank == 0 and i != 0 and i % log_interval == 0:
                    avg_loss = running_loss / log_interval
                    print(f"Epoch: {epoch}/{self._epochs}, Iteration: {i}, " f"Loss: {avg_loss}")
                    step = len(self._train_loader) * epoch + i
                    # dxf_util.write_metrics_to_nvflare(f"metrics_{step}", {"loss": avg_loss}, step)
                    running_loss = 0.0

                _check_heartbeat(dxf_util, local_rank)

                if i != 0 and i % log_interval == 0:
                    break


def _check_heartbeat(dxf_util, local_rank):
    result = None
    if local_rank == 0:
        result = dxf_util.check_heartbeat()

    heartbeat_check_failed = torch.tensor(int(local_rank == 0 and result is False))
    dist.broadcast(heartbeat_check_failed, src=0)

    if heartbeat_check_failed.item() == 1:
        raise TimeoutError(f"Heartbeat has not been updated for {dxf_util.heartbeat_timeout} seconds.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default="0.01", help="learning rate")
    parser.add_argument("--epochs", type=int, default=1, help="epochs")
    parser.add_argument("--rounds", type=int, default=10, help="number of rounds")
    parser.add_argument("--timeout", type=int, default=10000, help="number of seconds to wait for start from NVFlare")
    parser.add_argument("--poll_period", type=int, default=1, help="number of seconds to wait until next poll")
    parser.add_argument("--data_exchange_root", type=str, default="/tmp/nvflare/av")
    parser.add_argument("--heartbeat_timeout", type=float, default=10000, help="heartbeat timeout")
    args = parser.parse_args()

    lr = args.lr
    epochs = args.epochs

    backend = "gloo"  # nccl, mpi
    dist.init_process_group(backend=backend)
    local_rank = int(os.environ["LOCAL_RANK"])

    trainer = Cifar10Trainer(local_rank, lr, epochs)
    data_exchange_path = os.path.abspath(args.data_exchange_root)
    if local_rank == 0:
        if not os.path.exists(data_exchange_path):
            print(f"creating {data_exchange_path}")
            os.makedirs(data_exchange_path, exist_ok=False)

    dist.barrier()

    dxf_util = NVDXFUtils(data_exchange_path, heartbeat_timeout=args.heartbeat_timeout)
    if local_rank == 0:
        dxf_util.start_heartbeat()

    for i in range(args.rounds):
        print(f"AV Train Running round {i}")

        nvflare_good = False
        start_time = time.time()
        while time.time() - start_time < args.timeout:
            _check_heartbeat(dxf_util, local_rank)

            try:
                # read round from NVFlare
                external_round = None
                round_start = dxf_util.get_data("round_starts")
                external_round = round_start["round"]
                print(f"round_starts: {external_round}")
                if external_round == i:
                    nvflare_good = True
                    break
            except Exception as e:
                print(f"can't get model_data from NVFlare: {e}")
            finally:
                time.sleep(args.poll_period)

        if not nvflare_good:
            print("NVFlare side died")
            break

        # read weights from NVFlare
        input_weights = dxf_util.get_data("from_nvflare")
        torch_weights = {k: torch.as_tensor(v) for k, v in input_weights.items()}

        trainer.train(dxf_util=dxf_util, weights=torch_weights, local_rank=local_rank)

        dist.barrier()

        if local_rank == 0:
            # dump information to NVFlare
            cpu_state_dict = {k: v.cpu() for k, v in trainer.raw_model.state_dict().items()}
            dxf_util.put_data("to_nvflare", cpu_state_dict)
            dxf_util.put_data("round_ends", {"round": i})
            dxf_util.remove("from_nvflare")

    if local_rank == 0:
        dxf_util.stop_heartbeat()


if __name__ == "__main__":
    main()
