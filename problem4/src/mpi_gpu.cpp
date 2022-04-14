#include <stdio.h>
#include <mpi.h>
#include <cuda_runtime.h>
#include <ATen/ATen.h>
#include<iostream>
std::map<MPI_Datatype, at::ScalarType> mpiToATDtypeMap = {
        {MPI_INT, at::kInt}
};

int main() {
        //Initialize MPI and assign each process a GPU
        int provided;
        MPI_Init_thread(NULL, NULL, MPI_THREAD_MULTIPLE, &provided);
        int myRank;
        MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

        int deviceCount;
        cudaGetDeviceCount(&deviceCount);
        int deviceId = myRank % deviceCount;
        cudaSetDevice(deviceId);
        printf("Process %d is running on device %d.\n", myRank, deviceId);

        // create tensor in GPU memory
        int size = 5;
        at::ScalarType scalarType = mpiToATDtypeMap.at(MPI_INT);
        auto options =  at::TensorOptions().device(at::kCUDA, deviceId).dtype(scalarType);
        at::Tensor tensor = at::ones({size}, options) * myRank;
        std::cout<<"Tensor on process "<<myRank<<" is: "<<tensor<<std::endl;

        //Move tensor to CPU
        tensor = tensor.cpu();

        //All reduce across CPU
        int* buffer = tensor.data_ptr<int>();
        MPI_Allreduce(MPI_IN_PLACE, buffer, size, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        //Move tensor back to GPU
        tensor = tensor.cuda();

        if (myRank == 0) {
            int worldSize;
            MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
            std::cout<<"Result is: "<<tensor[0]<<std::endl;
        }

        MPI_Finalize();

        return 0;
}
