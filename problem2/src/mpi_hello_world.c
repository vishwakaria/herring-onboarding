#include <stdio.h>
#include <mpi.h>

int main() {
	MPI_Init(NULL, NULL);
	
	// Print own rank
	int process_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);
	printf("My rank is %d\n", process_rank);

	// Print world size
	int world_size;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	printf("World size by rank %d is %d\n", process_rank, world_size);

	// Broadcast a value from rank 0 to other ranks
	int data;
	const int root = 0;
	if (process_rank == root) {
		data = 123;
	}
	MPI_Bcast(&data, 1, MPI_INT, root, MPI_COMM_WORLD);
	if (process_rank != root) {
		printf("Rank %d received data with value %d from root.\n", process_rank, data);
	}
	
	MPI_Finalize();
}
