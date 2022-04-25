### Custom DistributedDataParallel class in PyTorch

Prompt:
1. Create a custom DistributedDataParallel class
2. Write a simple hook so that 0.1 is multiplied to every gradient. Optionally, use torch.distributed all_reduce to 
 communicate this number to every process.
    1. Note: PyTorch will accumulate gradients in param.grad after the hook. i.e. if you set param.grad = 0.1 * grad in the hook, 
   then the actual gradient that PyTorch will have after the hook is (1 + 0.1) * grad.
3. Register the hook for each parameter in DDP init
4. Add the forward method (see reference 2)
5. Use this custom DDP class to run some simple model training and print out the weight gradients in each iteration
6. You can run your training script in the DLAMI pytorch conda environment and 
`python -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 test.py`

Expected output:
```asm
(pytorch_p38) ubuntu@ip-10-0-0-32:~/src/onboarding/problem10$ torchrun --nproc_per_node=8 model.py 
WARNING:torch.distributed.run:
*****************************************
Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed. 
*****************************************
world size is: 8
my rank is: 0

Epoch 0

0/8      Weight Value: [[ 0. 10. 20. 30. 40. 50. 60. 70. 80. 90.]]

0/8      Weight Grad:  [[0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95]]

0/8      Weight New:   [[-5.000e-02  9.850e+00  1.975e+01  2.965e+01  3.955e+01  4.945e+01
   5.935e+01  6.925e+01  7.915e+01  8.905e+01]]

4/8      Weight Value: [[ 0. 10. 20. 30. 40. 50. 60. 70. 80. 90.]]
.
.
.
Epoch 9

7/8      Weight Value: [[-0.45  8.65 17.75 26.85 35.95 45.05 54.15 63.25 72.35 81.45]]

7/8      Weight Grad:  [[0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95]]

7/8      Weight New:   [[-0.5  8.5 17.5 26.5 35.5 44.5 53.5 62.5 71.5 80.5]]

7/8      result weights are:   [[-0.5  8.5 17.5 26.5 35.5 44.5 53.5 62.5 71.5 80.5]]

```