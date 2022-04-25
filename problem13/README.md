### CUDA Streams

Write a CUDA program to transfer an integer array from CPU to GPU, mutliply each entry in array by 2 on GPU, and transfer the result array back to CPU.
1. Allocate an integer array with 1GB worth of data on CPU (hence number of integers should be 1GB/sizeof(int)) and fill it with some random values.
2. [Copy the CPU buffer into a GPU buffer](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1gc263dbe6574220cc776b45438fc351e8).
3. Execute a CUDA kernel to multiply each element in the array by 2. https://cuda-tutorial.readthedocs.io/en/latest/tutorials/tutorial01/ is a good tutorial that explains how to execute a CUDA kernel.
4. [Copy the results on GPU back to CPU.](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1gc263dbe6574220cc776b45438fc351e8)
5. Measure the timing for steps b-d using [CUDA events](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html). (https://developer.nvidia.com/blog/how-implement-performance-metrics-cuda-cc/)
 6. Optionally, use nsys profiler to profile this CUDA program.

Expected Output:
```asm
ubuntu@ip-10-0-0-32:~/src/onboarding/problem15/build$ ./cuda_streams
Data transfer takes 14830.627930ms.
```
