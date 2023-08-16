# -*- coding: utf-8 -*-

import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from net import Net
from torch import nn

from nvflare.app_common.model_exchange.ml_to_fl import fl_evaluate, fl_train

BATCH_SIZE = 4
PATH = "./cifar_net.pth"
DATASET_ROOT = "/tmp/data"

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


trainset = torchvision.datasets.CIFAR10(root=DATASET_ROOT, train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root=DATASET_ROOT, train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)


@fl_train()
def train(total_epochs=2, lr=0.001):
    net = Net()
    net.load_state_dict(train.fl_model_params)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    for epoch in range(total_epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}")
                running_loss = 0.0
                break

    print("Finished Training")
    torch.save(net.state_dict(), PATH)

    train.fl_model_params = net.state_dict()


@fl_evaluate()
def evaluate():
    net = Net()
    pretrained_weights = evaluate.fl_model_params
    net.load_state_dict(pretrained_weights)
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct // total
    print(f"Accuracy of the network on the 10000 test images: {accuracy} %")
    evaluate.fl_metrics = accuracy


def main():
    train()
    evaluate()


if __name__ == "__main__":
    main()
