#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <math.h>
#include "buffer_ops.hpp"

int main() {
    // Initialize 1GB buffers on CPU and GPU
    int *in, *out, *d_in, *d_out;
    const size_t buffer_size = pow(2, 30);
    int length = buffer_size / sizeof(int);
    in = (int*)malloc(buffer_size);
    out = (int*)malloc(buffer_size);

    cudaMalloc((void**)&d_in, buffer_size);
    cudaMalloc((void**)&d_out, buffer_size);

    // Allocate CPU buffer with random data
    for (int i=0; i < length; i++) {
        in[i] = i%10;
    }

    //Use events to measure timing of each activity
    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    cudaEventRecord(startEvent, 0);

    //Move buffer to GPU
    cudaStream_t cudaStream;
    cudaStreamCreate(&cudaStream);
    cudaMemcpyAsync(d_in, in, buffer_size, cudaMemcpyHostToDevice, cudaStream);

    //Execute CUDA kernel to multiple each element by 2
    buffer_mul(d_in, d_out, length, 2, cudaStream);

    //Copy results back to CPU
    cudaMemcpyAsync(out, d_out, buffer_size, cudaMemcpyDeviceToHost, cudaStream);
    cudaStreamSynchronize(cudaStream);

    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    float time;
    cudaEventElapsedTime(&time, startEvent, stopEvent);
    printf("Approach 1 takes %fms.\n", time);

    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);

    cudaDeviceReset();
    return 0;
}


