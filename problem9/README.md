### Distributed training of neural networks using PyTorch collectives

We implement distributed training in PyTorch using AllReduce collective communication by modifying the previous Pytorch MNIST sample program.

1. In the sample code, setup Pytorch in distributed mode using the distributed module of PyTorch. 
For details look here (https://pytorch.org/docs/stable/distributed.html). Initialize the distributed environment using
[init_process_group](https://pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group). You can use the “nccl” backend.
2. Partition training/test dataset - In case of data parallel training, the workers train on non-overlapping data partitions. 
You will use the distributed sampler to distribute the data among workers. For more details look at
[torch.utils.data.distributed_sampler](https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler). 
Look at [distributed examples](https://github.com/pytorch/examples/tree/master/distributed/ddp)  for code references.
3. We will now use the allreduce collective to implement gradient aggregation. 
PyTorch provides collectives interfaces through the [torch.distributed package](https://pytorch.org/docs/stable/distributed.html). 
To perform gradient aggregation you will need to read the gradients after backward pass for each layer. 
Pytorch performs gradient computation using auto grad when you call .backward on a computation graph.
The gradient is stored in .grad attribute of the parameters. The parameters can be accessed using model.parameters().
4. To perform gradient aggregation, use the built in allreduce collective to sync gradients among different workers. 
Note the PyTorch allreduce call doesn’t have an ‘average’ operation. You can use the ‘sum’ operation and 
then get the average on each node by dividing with number of workers participating.
5. The workers update the grad variable with the received gradient and then continue training.
6. Note: Use [PyTorch torchrun](https://pytorch.org/docs/stable/elastic/run.html#launcher-api) to launch distributed training.



