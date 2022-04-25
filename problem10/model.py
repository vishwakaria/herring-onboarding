import os
import pprint
import argparse
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
#from custom_ddp import DistributedDataParallel as DDP
from torch.nn.modules import Module
from torch.autograd import Variable


dist.init_process_group("nccl")


def vocal_errors(func):
    """Errors emitted from hooks are swallowed by PyTorch
    and show up as unrecognizable RuntimeError in autograd.
    Lets print them first."""

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(
                    "Error in grad hook."
            )
            print("Error: ", e, type(e))
            import traceback
            traceback.print_stack()
            raise e

    return wrapper

class DistributedDataParallel(Module):
    def __init__(self,
                 module,
                 device_ids=None,
                 output_device=None,
                 dim=0,
                 broadcast_buffers=True,
                 process_group=None,
                 bucket_cap_mb=25,
                 find_unused_parameters=False,
                 check_reduction=False,
                 gradient_as_bucket_view=False,
                 gradient_accumulation_steps=None):

        super(DistributedDataParallel, self).__init__()

        self.module = module
        self._trainable_params = None
        self._num_gpus = dist.get_world_size()
        self._rank = dist.get_rank()


        #Register hook to receive gradients
        for param in self.module.parameters():
            param.register_hook(lambda grad: self._grad_multiply_hook(grad))

        #Broadcast parameters to make sure all devices have the same params to begin with
        for _, param in enumerate(self.module.parameters()):
            print('param on rank {} before broadcast: {}'.format(self._rank, param.detach().cpu().numpy()))
            self._broadcast(param)
            print('param on rank {} after broadcast: {}'.format(self._rank, param.detach().cpu().numpy()))
    
   
    def forward(self, *inputs, **kwargs):
        result = self.module(*inputs, **kwargs)
        self._final_callback_required = False
        return result

    def _get_trainable_params(self):
        if self._trainable_params is None:
            self._trainable_params = [
                param for param in self.module.parameters()
                if param.requires_grad
            ]
        return self._trainable_params

    @vocal_errors
    def _grad_multiply_hook(self, grad):
        grad = 0.1 * grad
        return grad

    def _broadcast(self, grad, rootRank=0):
        if self._rank == 0:
            dist.broadcast(grad, rootRank)



N = 10  # input size, weight size
STEPS = 10
NPDTYPE = np.float32 
DTYPE = torch.float32 

def debug_info():
    return "%4s/%4s\t" % (dist.get_rank(),
                          dist.get_world_size())


def get_weights(device):
    npa = np.asarray([[i * STEPS for i in range(N)]], dtype=NPDTYPE)
    return torch.nn.Parameter(torch.Tensor(npa).type(DTYPE).cuda())


def get_input():
    rank = dist.get_rank()
    # diversify gradients on different ranks with mean=0
    proportion = 0.5
    print(proportion)
    npa = np.asarray([[i + proportion for i in range(N)]], dtype=NPDTYPE)
    return torch.as_tensor(npa)

def get_label(device):
    npa = np.asarray([[0.0]], dtype=NPDTYPE)
    return torch.as_tensor(npa, device=device)

def average_gradients(model):
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= dist.get_world_size()

def train(layer, model, device, data, optimizer, epoch, rank, world_size):
    
    print("Epoch %d\n" %epoch)
    model.train()
    debug = True
    target = get_label(device)
    if debug: print(debug_info(), "1 Input:       ", data.cpu().numpy())

    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()
    output = model(data)
    if debug: print(debug_info(), "2 Output:      ", output.data.cpu().numpy())

#    loss = F.nll_loss(output, target)
    criterion = nn.MSELoss()
    loss = criterion(torch.sqrt(output), target)
    if debug: print(debug_info(), "3 Loss:         ", loss.data.cpu().numpy())

    loss.backward()
    if debug:
        print(debug_info(), "4 Weight Value:", layer.weight.data.cpu().numpy())
        print(debug_info(), "5 Weight Grad: ", layer.weight.grad.cpu().numpy())

    #average_gradients(model)
    optimizer.step()
    if debug:
        print(debug_info(), "6 Weight New:  ", layer.weight.data.cpu().numpy())


def iteration_step(step, device, model, criterion, layer, optimizer, debug):
    data = get_input(device)
    print("\nIteration %d" % step)
    if debug: print(debug_info(), "1 Input:       ", data.cpu().numpy())

    output = model(data)
    if debug: print(debug_info(), "2 Output:      ", output.data.cpu().numpy())

    loss = criterion(torch.sqrt(output), get_label(device))
    if debug: print(debug_info(), "3 Loss:         ", loss.data.cpu().numpy())

    loss.backward()

    if step < 0: return  # warm up step
    if debug:
        print(debug_info(), "4 Weight Value:", layer.weight.data.cpu().numpy())
        print(debug_info(), "5 Weight Grad: ", layer.weight.grad.cpu().numpy())

    optimizer.step()
    if debug:
        print(debug_info(), "6 Weight New:  ", layer.weight.data.cpu().numpy())

    optimizer.zero_grad()


def main():
    if not torch.cuda.is_available():
        raise ValueError('cuda not available')
    torch.manual_seed(1)

    device = torch.device("cuda")
    
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    print("world size is: {}".format(world_size))
    print("my rank is: {}".format(local_rank))
   
    layer = nn.Linear(10, 1, bias=False)
    layer.weight = get_weights(device)
    model = nn.Sequential(layer) 
    model.to(device)
    model = DistributedDataParallel(model, device_ids=[local_rank])


    optimizer = torch.optim.SGD(model.parameters(), lr = 1.)
    data = get_input().to(device)

    for epoch in range(0, STEPS):
        train(layer, model, device, data, optimizer, epoch, local_rank, world_size)
    
    resulting_weights = layer.weight.data.cpu().numpy()
    print(debug_info(), "result weights are:  ", resulting_weights)
    exit()

if __name__ == '__main__':
    main()
