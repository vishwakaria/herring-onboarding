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