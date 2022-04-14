#include <mpi.h>
#include <cuda_runtime.h>
#include <ATen/ATen.h>
#include <iostream>
#include <nccl.h>

std::map<MPI_Datatype, at::ScalarType> mpiToATDtypeMap = {
        {MPI_INT, at::kInt}
};

int main() {
        //Initialize MPI and assign each process a GPU
        int provided;
        MPI_Init_thread(NULL, NULL, MPI_THREAD_MULTIPLE, &provided);
        int myRank, worldSize;
        MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
        MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
       
	    int deviceCount;
        cudaGetDeviceCount(&deviceCount);
        int deviceId = myRank % deviceCount;
        cudaSetDevice(deviceId);
	    std::cout<<"Process "<<myRank<<" is running on device "<<deviceId<<std::endl;
	
        //initialize NCCL
        ncclUniqueId id;
        ncclComm_t comm;
        if (myRank == 0) ncclGetUniqueId(&id);
        MPI_Bcast((void*)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
        ncclCommInitRank(&comm, worldSize, id, myRank);

        // create tensor in GPU memory 
        int size = 5;
        at::ScalarType scalarType = mpiToATDtypeMap.at(MPI_INT);
        auto options =  at::TensorOptions().device(at::kCUDA, deviceId).dtype(scalarType);
        at::Tensor tensor = at::ones({size}, options) * myRank;
        std::cout<<"Tensor on process "<<myRank<<" is: "<<tensor<<std::endl;

        //All reduce across GPU
        int* sendBuffer = tensor.data_ptr<int>();
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        ncclAllReduce((const void*) sendBuffer, (void*) sendBuffer, size, ncclInt, ncclSum, comm, stream);
    	cudaStreamSynchronize(stream);

        if (myRank == 0) {
                std::cout<<"Result is: "<<tensor[0]<<std::endl;
        }

        //Finalize NCCL and MPI
        ncclCommDestroy(comm);
        MPI_Finalize();

        return 0;
}
