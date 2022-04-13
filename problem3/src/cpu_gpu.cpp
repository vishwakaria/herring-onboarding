#include <stdio.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cstring>
#include <math.h>


int main() {
	//Allocate GPU memory and set it to constant value
	char *d_buffer;
	const char data = 'q';
	const size_t buffer_size = pow(2, 30);
	cudaMalloc((void**)&d_buffer, buffer_size);
	cudaMemset(d_buffer, data, buffer_size/sizeof(char));
    printf("Created a buffer of size 1 GB.\n");

	//Allocate CPU memory and copy GPU contents to CPU
	char *buffer = (char*)malloc(buffer_size);
	cudaEvent_t startEvent, stopEvent;
	cudaEventCreate(&startEvent);
	cudaEventCreate(&stopEvent);
	cudaEventRecord(startEvent, 0);

	cudaStream_t cudaStream;
	cudaStreamCreate(&cudaStream);
	cudaMemcpyAsync(buffer, d_buffer, buffer_size, cudaMemcpyDeviceToHost, cudaStream);
	cudaStreamSynchronize(cudaStream);
    printf("Copied buffer from GPU to CPU.\n");
	
	cudaEventRecord(stopEvent, 0);
	cudaEventSynchronize(stopEvent);
	float time;
	cudaEventElapsedTime(&time, startEvent, stopEvent);
	printf("Throughput of data transfer: %f\n", buffer_size * 1e-6 / time);

	cudaEventDestroy(startEvent);
	cudaEventDestroy(stopEvent);

	//Check if GPU and CPU buffers have the same content
    bool mismatch_detected = false;
	for (int i=0; i<(buffer_size/sizeof(char)); i++) {
		if ('q' != buffer[i]) {
			printf("Value mismatch at position %d\n", i);
            mismatch_detected = true;
		}
	}
    if (!mismatch_detected) {
        printf("GPU and CPU buffers have the same content.\n");
    }

	cudaDeviceReset();
	return 0;
}

