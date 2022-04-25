### SIMD

Prompt:
1. Create 5 arrays of 8 integers, then use AVX to vertically sum these arrays.
2. Then create 2 arrays of more than 8 integers, and use a for loop that increments by 8 (the number of 32-bit integers that can fit in a 256 bit register) and AVX to vertically sum the arrays 8 elements at a time.
3. Build with cmake, setting the target architecture to one that supports AVX intrinsics.

Expected Output:
```
$ ./simd_avx
PART 1: Vertically sum 5 arrays of 8 integers each using AVX
Arrays are:
3       6       7       5       3       5       6       2       
9       1       2       7       0       9       3       6       
0       6       2       6       1       8       7       9       
2       0       2       3       7       5       9       2       
2       8       9       7       3       6       1       2       
Result is: 
16      21      22      28      14      33      26      21  
    
PART 2: Vertically sum 2 arrays with more than 8 elements using AVX
Arrays are:
9       3       1       9       4       7       8       4       5       0       3       6       1       0       
6       3       2       0       6       1       5       5       4       7       6       5       6       9       
Sum calculated is: 
15      6       3       9       10      8       13      9       9       7       9       11      7       9      
```