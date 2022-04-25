#pragma once

#include <cuda_runtime_api.h>

void buffer_mul(int *in, int *out, int length, int factor, cudaStream_t cudaStream);