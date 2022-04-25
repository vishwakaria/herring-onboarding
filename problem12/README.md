### Slurm

Prompt:
1. Create an AWS Parallel Cluster of P4D instances.
2. Log in to the controller node.
3. Write a simple python program to print hello world and the node ID from each node and run with srun

How to run?
```
$ sbatch hello.slurm
Submitted batch job 21
```
Output:
```
$vi slurm-21.out

('Hello! This is task number: ', '0')
('Hello! This is task number: ', '1')
```

References:
1. https://homeowmorphism.com/2017/04/18/Python-Slurm-Cluster-Five-Minutes
2. https://slurm.schedmd.com/sbatch.html