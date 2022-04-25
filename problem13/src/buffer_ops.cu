#include "buffer_ops.hpp"

__global__ void buffer_mul_kernel(int *in, int *out, int length, int factor) {
    for (int i = 0; i < length; i++) {
        out[i] = factor * in[i];
    }
}

void buffer_mul(int *in, int *out, int length, int factor, cudaStream_t cudaStream) {
    buffer_mul_kernel<<<1,4>>>(in, out, length, factor);
}
