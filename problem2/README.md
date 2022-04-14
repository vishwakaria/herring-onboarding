### MPI Hello World

Prompt:
1. Write mpi_hello_world.c that initializes MPI, prints its own rank, prints the world size, 
 and uses MPI’s broadcast function to broadcast a value from rank_0 to other ranks. 
 Print the broadcasted value from all ranks.
2. Build your code with cmake and run it. 

References:
1. Good tutorial if you are new to MPI: https://mpitutorial.com/tutorials/mpi-introduction/

Run instructions:
```
cd problem2
mkdir build && cd build
cmake ..
make 
mpirun -np 8 ./mpi_hello_world
```

Expected Output:
```
$ mpirun -np 2 ./mpi_hello_world
My rank is 0
World size by rank 0 is 2
My rank is 1
World size by rank 1 is 2
Rank 1 received data with value 123 from root.

```