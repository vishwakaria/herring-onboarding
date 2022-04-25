from __future__ import print_function
import os
import pprint
import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.nn.parallel import DistributedDataParallel as DDP

dist.init_process_group("nccl")

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def average_gradients(model):
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= dist.get_world_size()

def train(model, device, train_loader, optimizer, epoch, rank, world_size):
    model.train()
    log_interval = 10
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        average_gradients(model)
        optimizer.step()
        if batch_idx % log_interval == 0 and rank == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data) * world_size, len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    if not torch.cuda.is_available():
        raise ValueError('cuda not available')
    torch.manual_seed(1)

    device = torch.device("cuda")
    
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    print("world size is: {}".format(world_size))
    print("my rank is: {}".format(local_rank))
    
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

    is_first_local_rank = (local_rank == 0)
    if is_first_local_rank:
        train_dataset = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    dist.barrier() #prevent other ranks from accessing the data early
    if not is_first_local_rank:
        train_dataset = datasets.MNIST('../data', train=True, download=False,
                       transform=transform)

    batch_size = 64
    test_batch_size = 1000
    sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, 
                                                              num_replicas=world_size, 
                                                              rank=local_rank) 
    train_kwargs = {'batch_size': batch_size, 
                    'shuffle': False, 
                    'sampler': sampler,
                    'pin_memory': True, 
                    'num_workers': 0}

    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
    print('For world size {}, train_loader length is {}'.format(world_size, len(train_loader))) 
    if local_rank == 0:
        test_dataset = datasets.MNIST('../data', train=False,
                       transform=transform)
        test_loader = torch.utils.data.DataLoader(test_dataset, 
                                                  batch_size=test_batch_size,
                                                  shuffle=True)
    
    model = Net().to(local_rank)
    ddp_model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    optimizer = optim.Adadelta(ddp_model.parameters(), lr=1.0)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
    for epoch in range(1, 2):
        sampler.set_epoch(epoch)
        train(ddp_model, local_rank, train_loader, optimizer, epoch, local_rank, world_size)
        if local_rank == 0:
            test(ddp_model, local_rank, test_loader)
        scheduler.step()


if __name__ == '__main__':
    main()
