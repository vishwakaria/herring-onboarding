Prompt:
1. Allocate a GPU buffer of length 1 GB and fill it with a constant.
2. Allocate a buffer in CPU memory of length 1 GB and copy the contents of the GPU buffer into the CPU buffer.
3. Check each byte in the CPU memory and assert it is same as the constant you filled in GPU memory.
4. Measure and print the throughput of data transfer from GPU to CPU as GB/s


Notes:
1. This exercise requires the setup of an EC2 instance (to have access to GPU). 
 I have used a p3.16x instance which has 8 GPUs, and [Amazon Deep Learning AMI](https://aws.amazon.com/machine-learning/amis/) with Ubuntu 18.04 OS.
2. 


