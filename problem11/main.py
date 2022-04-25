import torch
import torch.distributed as dist
import random

dist.init_process_group('nccl');

def all_to_all(rank, world_size, device):
    if rank == 0:
        print("\nStarting all to all communication.") 
    tensors = torch.zeros([world_size, 5]).to(device)
    tensors[rank] += rank
    for i in range(0, world_size):
        if rank != i:
            dist.send(tensor=tensors[rank], dst = i)
            dist.recv(tensor=tensors[i], src = i) 
    if rank == 0:
        print('rank 0 has tensors {}'.format(tensors.cpu().numpy()))
    sum = torch.sum(tensors, 0)
    if rank == 0:
        print('After reduce, tensor is: {}'.format(sum.cpu().numpy()))

def iall_to_all(rank, world_size, device):
    if rank == 0:
        print("\nStarting non-blocking all to all communication.")
    tensors = torch.zeros([world_size, 5]).to(device)
    tensors[rank] += rank
    req = None
    for i in range(0, world_size):
        if rank != i:
            req = dist.isend(tensor=tensors[rank], dst = i)
            req = dist.irecv(tensor=tensors[i], src = i) 
            req.wait()
    if rank == 0:
        print('rank 0 has tensors {}'.format(tensors.cpu().numpy()))
    sum = torch.sum(tensors, 0)
    if rank == 0:
        print('After reduce, tensor is: {}'.format(sum.cpu().numpy()))

def aggregator(rank, world_size, device):
    if rank == 0:
        print("\nStarting aggregation.")
    tensor = torch.zeros(5).to(device) + rank
    driver_id = 0
    #print('aggregator rank is: {}'.format(driver_id))
    reduced_tensor = torch.zeros(5).to(device)
    if rank != driver_id:
        dist.send(tensor=tensor, dst=driver_id)
        dist.recv(tensor=reduced_tensor, src=driver_id)
    if rank == driver_id:
        all_tensors = torch.zeros([world_size, 5]).to(device)
        all_tensors[driver_id] = tensor
        for i in range(0, world_size):
            if i != rank:
                dist.recv(tensor=all_tensors[i], src=i)
        print('After recv, rank 0 has tensors {}'.format(all_tensors.cpu().numpy()))
        reduced_tensor = torch.sum(all_tensors, 0)
        for i in range(0, world_size):
            if i != rank:
                dist.send(tensor=reduced_tensor, dst=i)
        print('Rank {} sent reduced tensor: {}'.format(driver_id, reduced_tensor.cpu().numpy()))
    if world_size > 1 and rank == world_size - 1:
        print('Rank {} received reduced tensor: {}'.format(rank, reduced_tensor.cpu().numpy())) 

def main():
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    world_size = dist.get_world_size()
    device = torch.device("cuda")
    all_to_all(rank, world_size, device)
    iall_to_all(rank, world_size, device)
    aggregator(rank, world_size, device)

if __name__ == "__main__":
    main()

