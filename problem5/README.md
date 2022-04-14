### NCCL All Reduce

Prompt:
1. Initialize MPI and assign each process a GPU.
2. Create a tensor in GPU memory, fill it with data, all reduce across all processes on GPU using NCCL, 
and send the all reduced tensor to GPU. Use the PyTorch C++ library to create and handle tensors.
3. Build with cmake.

References:
1. [NCCL Official Docs](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/)

Expected Output:
```
Process 0 is running on device 0
7b6059672f82:104:104 [0] NCCL INFO Bootstrap : Using lo:127.0.0.1<0>
7b6059672f82:104:104 [0] NCCL INFO NET/Plugin: Failed to find ncclCollNetPlugin_v4 symbol.

7b6059672f82:104:104 [0] ofi_init:1134 NCCL WARN NET/OFI Only EFA provider is supported
7b6059672f82:104:104 [0] NCCL INFO NET/IB : No device found.
7b6059672f82:104:104 [0] NCCL INFO NET/Socket : Using [0]lo:127.0.0.1<0> [1]eth0:172.17.0.2<0>
7b6059672f82:104:104 [0] NCCL INFO Using network Socket
NCCL version 2.10.3+cuda11.3
7b6059672f82:104:104 [0] NCCL INFO Channel 00/32 :    0
7b6059672f82:104:104 [0] NCCL INFO Channel 01/32 :    0
7b6059672f82:104:104 [0] NCCL INFO Channel 02/32 :    0
7b6059672f82:104:104 [0] NCCL INFO Channel 03/32 :    0
7b6059672f82:104:104 [0] NCCL INFO Channel 04/32 :    0
7b6059672f82:104:104 [0] NCCL INFO Channel 05/32 :    0
7b6059672f82:104:104 [0] NCCL INFO Channel 06/32 :    0
7b6059672f82:104:104 [0] NCCL INFO Channel 07/32 :    0
7b6059672f82:104:104 [0] NCCL INFO Channel 08/32 :    0
7b6059672f82:104:104 [0] NCCL INFO Channel 09/32 :    0
7b6059672f82:104:104 [0] NCCL INFO Channel 10/32 :    0
7b6059672f82:104:104 [0] NCCL INFO Channel 11/32 :    0
7b6059672f82:104:104 [0] NCCL INFO Channel 12/32 :    0
7b6059672f82:104:104 [0] NCCL INFO Channel 13/32 :    0
7b6059672f82:104:104 [0] NCCL INFO Channel 14/32 :    0
7b6059672f82:104:104 [0] NCCL INFO Channel 15/32 :    0
7b6059672f82:104:104 [0] NCCL INFO Channel 16/32 :    0
7b6059672f82:104:104 [0] NCCL INFO Channel 17/32 :    0
7b6059672f82:104:104 [0] NCCL INFO Channel 18/32 :    0
7b6059672f82:104:104 [0] NCCL INFO Channel 19/32 :    0
7b6059672f82:104:104 [0] NCCL INFO Channel 20/32 :    0
7b6059672f82:104:104 [0] NCCL INFO Channel 21/32 :    0
7b6059672f82:104:104 [0] NCCL INFO Channel 22/32 :    0
7b6059672f82:104:104 [0] NCCL INFO Channel 23/32 :    0
7b6059672f82:104:104 [0] NCCL INFO Channel 24/32 :    0
7b6059672f82:104:104 [0] NCCL INFO Channel 25/32 :    0
7b6059672f82:104:104 [0] NCCL INFO Channel 26/32 :    0
7b6059672f82:104:104 [0] NCCL INFO Channel 27/32 :    0
7b6059672f82:104:104 [0] NCCL INFO Channel 28/32 :    0
7b6059672f82:104:104 [0] NCCL INFO Channel 29/32 :    0
7b6059672f82:104:104 [0] NCCL INFO Channel 30/32 :    0
7b6059672f82:104:104 [0] NCCL INFO Channel 31/32 :    0
7b6059672f82:104:104 [0] NCCL INFO Trees [0] -1/-1/-1->0->-1 [1] -1/-1/-1->0->-1 [2] -1/-1/-1->0->-1 [3] -1/-1/-1->0->-1 [4] -1/-1/-1->0->-1 [5] -1/-1/-1->0->-1 [6] -1/-1/-1->0->-1 [7] -1/-1/-1->0->-1 [8] -1/-1/-1->0->-1 [9] -1/-1/-1->0->-1 [10] -1/-1/-1->0->-1 [11] -1/-1/-1->0->-1 [12] -1/-1/-1->0->-1 [13] -1/-1/-1->0->-1 [14] -1/-1/-1->0->-1 [15] -1/-1/-1->0->-1 [16] -1/-1/-1->0->-1 [17] -1/-1/-1->0->-1 [18] -1/-1/-1->0->-1 [19] -1/-1/-1->0->-1 [20] -1/-1/-1->0->-1 [21] -1/-1/-1->0->-1 [22] -1/-1/-1->0->-1 [23] -1/-1/-1->0->-1 [24] -1/-1/-1->0->-1 [25] -1/-1/-1->0->-1 [26] -1/-1/-1->0->-1 [27] -1/-1/-1->0->-1 [28] -1/-1/-1->0->-1 [29] -1/-1/-1->0->-1 [30] -1/-1/-1->0->-1 [31] -1/-1/-1->0->-1
7b6059672f82:104:104 [0] NCCL INFO Connected all rings
7b6059672f82:104:104 [0] NCCL INFO Connected all trees
7b6059672f82:104:104 [0] NCCL INFO 32 coll channels, 32 p2p channels, 32 p2p channels per peer
7b6059672f82:104:104 [0] NCCL INFO comm 0x564d4af51740 rank 0 nranks 1 cudaDev 0 busId 170 - Init COMPLETE
Tensor on process 0is:  0
0
0
0
0
[ CUDAIntType{5} ]
Result is: 0
[ CUDAIntType{} ]
```