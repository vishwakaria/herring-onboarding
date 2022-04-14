### Using MPI with GPU

Prompt:
1. Initialize MPI and assign each process a GPU.
2. Create a tensor in GPU memory, fill it with data, send it to CPU, all reduce across all processes on CPU, 
and send the all reduced tensor to GPU. Use the PyTorch C++ library to create and handle tensors.
3. Build with cmake. You will need to export Torch_DIR as the directory to torch, which will be something like 
`set(Torch_DIR /home/ec2-user/anaconda3/envs/pytorch_latest_p37/lib/python3.7/site-packages/torch/share/cmake/Torch/)`.

Notes:
1. Note that for A100, CUDA 11.0 is required. Torch CPP API for tensor allocation will hang silently with CUDA 10.
2. Make sure to indicate in the CMakeLists the correct location for CUDA 11:
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-11.0/)
3. Also, don’t forget to include the Torch and cuda libraries in the CMakeLists
`include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${TORCH_INCLUDE_DIRS})`
4. Using Ubuntu 18.04 seems to cause build issues with CUDA. Thus, Amazon Linux 2 is the preferred OS. 
My EC2 instance uses Ubuntu, thus I ended up using a dockerized environment with an Amazon DLC images.

References: 
1. Amazon DLC images: https://github.com/aws/deep-learning-containers/blob/master/available_images.md

Expected output:
```
Process 0 is running on device 0.
Tensor on process 0 is:  0
 0
 0
 0
 0
[ CUDAIntType{5} ]
Result is: 0
[ CUDAIntType{} ]

```