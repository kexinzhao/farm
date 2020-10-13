# Kernel Design

<img src="/doc/images/register_layout.png" width="800" />

The 64-bit ARM architecture provides 32 128-bit SIMD (single instruction multiple data) registers. These SIMD registers (referred to as V0 ~ V31) are considered as vectors of elements of the same data type and the ARM NEON instructions perform the same operations in all lanes of the vectors. For instance, NEON instructions allow up to 16 x 8-bit, 8 x 16-bit, 4 x 32-bit integer operations. Our batch size 1 GEMM kernel is implemented to achieve effective data level parallelism using these registers and instructions.  

Specifically, for a matrix vector multiplication formulated as `lhs x rhs = res`, the `lhs` matrix of dimension `m x k` is treated as a collection of blocks with dimension `8 x k`. Our kernel sequentially calculates the product of each block and `rhs` of dimension `k x n` as follows:

- Each block of dimension `8 x k` is divided into mini-blocks of dimension `8 x 16` for serial processing. A mini-block of 8-bit integers are loaded from memory onto 16 128-bit SIMD registers (V0 ~ V15 in the figure above), where each register holds 8 consecutive matrix elements on the same row. Each register then expands its 8 x 8-bit integers to 8 x 16-bit integers and adds a 16-bit integer offset (obtained via the [quantization paradigm](https://github.com/google/gemmlowp/blob/master/doc/low-precision.md)) to each element. 

- Similarly, we use another two SIMD registers (V16 and V17) to hold the corresponding mini-block of dimension `16 x 1` in `rhs` using the 8 x 16-bit integer format.

- We use NEON instruction to multiply four pairs of 16-bit integers in parallel and add the products to the four 32-bit integer accumulators on a SIMD register, respectively. So an `8 x 16` mini-block takes 32 NEON instructions to compute and 8 SIMD registers (V18 ~ V25) to store the intermediate accumulation results.

- After processing all the `8 x 16` mini-blocks of a block, we compute the final matrix multiplication result as a vector of eight 8-bit integers (each integer is an element in `res`) and write it to memory. Each integer in the vector can be obtained from a corresponding SIMD register in V18 ~ V25 by firstly adding cross the four 32-bit accumulators on that register to get the final 32-bit output and then scaling it down to 8-bit based on the quantization paradigm. 

We also implemented kernels for batch sizes 2, 3, and 4 using similar design principles. However, the mini-block size needs to be adjusted accordingly to fit the budget of 32 SIMD registers. Consequently, we choose to use mini-blocks of dimension `8 x 8`, `4 x 16`, and `4 x 8` for kernels of batch size 2, 3, and 4, respectively. Moreover, we can also handle larger batch size GEMM by using a combination of batch size 1 ~ 4 kernels. For batch size `N`, we use `N / 4` kernels of batch size 4 followed by a kernel of batch size `N % 4`.
