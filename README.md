# farm

A self-contained library that performs low precision general matrix 
multiplication ("GEMM") optimized for small batch sizes on 64-bit ARM 
processors.

# Introduction

Farm is inspired by the
[gemmlowp](https://github.com/google/gemmlowp) library. It contains specialized 
ARM 64-bit assembly kernels for batch sizes 1 to 4. For higher batch sizes, it 
uses a combination of these assembly kernels. Please be aware that we have only 
tested these kernels for batch sizes up to 10 and most likely these kernels will
not be efficient for higher batch sizes.

The main motivation of creating this library is explained in
[fast-gemv.txt](https://github.com/google/gemmlowp/blob/master/todo/fast-gemv.txt).
Essentially, gemmlowp is not well optimized for small batch size GEMMs and
designing specialized ARM kernels could provide significant performance
improvement. This library is an essential component for the
[on-device automatic speech recognition](https://github.svail.baidu.com/baidu-research/ondevice-asr)
(WE SHOULD REPLACE THIS WITH REFERENCE TO OUR PAPER) and enabled the ASR model 
to run real time on ARM processors.

If you use the code in your research, please cite the following paper. (ADD 
CITATION HERE)

# Public Interface

farm's pubic interface is defined in [include/farm.h](include/farm.h) as:

```
template <MapOrder LhsOrder, MapOrder RhsOrder, MapOrder ResultOrder>
void Gemm(const MatrixMap<LhsOrder>& lhs,
          const MatrixMap<RhsOrder>& rhs,
          MatrixMap<ResultOrder>* res,
          int lhs_offset, int rhs_offset, int result_offset,
          int result_mult_int, int result_shift);
```

### Template Parameters


`LhsOrder`, `RhsOrder`, `ResultOrder`: the storage orders (row-major or
column-major) of the LHS, RHS, result matrices. At the moment, this must be 
RowMajor, ColMajor, and ColMajor, respectively.


### Function parameters

`lhs`, `rhs`, `res`: The LHS, RHS, and result operand matrices such that 
`res = lhs x rhs`. Note that these are `MatrixMap` objects, mapping external
buffers as matrices, not owning data. See [include/map.h](include/map.h) for
more details. The matrix elements must be contiguous in an external buffer
(row-major for LHS and column-major for RHS and result).

`lhs_offset`, `rhs_offset`, `result_offset`, `result_mult_int`, `result_shift`: 
Parameters of the low precision paradigm (adopted from gemmlowp, see
[quantization.md](https://github.com/google/gemmlowp/blob/master/doc/quantization.md)
and
[low-precision.md](https://github.com/google/gemmlowp/blob/master/doc/low-precision.md) 
). Details on how to
calculate these values are given in [doc/low-precision.pdf](doc/low-precision.pdf).

### Usage

The dimension of the matrix multiplication `res = lhs x rhs` can be described as
`(m, k, n)`, where `m` is the number of rows in `lhs`, `k` is the number of
columns in `lhs` and rows in `rhs`, and `n` is the number of columns in `rhs`.
If we refer to `uint8_t *ptr_lhs, *ptr_rhs, *ptr_res` as pointers to the first
element of `lhs`, `rhs`, and `res` matrices (stored in the external buffers),
respectively, then the three matrices are typically constructed using:

```
farm::MatrixMap<farm::MapOrder::RowMajor> uint8_lhs_matrix(ptr_lhs, m, k);
farm::MatrixMap<farm::MapOrder::ColMajor> uint8_rhs_matrix(ptr_rhs, k, n);
farm::MatrixMap<farm::MapOrder::ColMajor> uint8_res_matrix(ptr_res, m, n);
```

Then a typical call to `Gemm` will look like:

```
farm::Gemm(
    uint8_lhs_matrix, uint8_rhs_matrix, &uint8_res_matrix,
    lhs_offset, rhs_offset, res_offset, res_mult_int, res_shift);
```

# Compiling 

Simply use farm as a submodule and add [include/farm.h](include/farm.h) in your
source code. Then use the following compiling options:

```
c++ -O3 -o ./bin/a.out source.cc
```

You can benchmark the performance and bandwidth of the implemented kernels by:

```
cd farm/test
make gemm 
./bin/gemm_bench
``` 

You can also test the correctness of the implemented kernels by:

```
cd farm/test
make test
./bin/test_correctness
``` 

# Benchmark

Performance and bandwidth of farm on iPhone 7, iPhone 6, and Raspberry 
Pi 3 for batch-sizes up to 10 are provided in the following tables. 
For more details about the performance and comparisons with gemmlowp,
see [doc/performance-analysis.md](doc/performance-analysis.md).

## iPhone 7
| GEMM                | Application        | Results (ms) | GigaOps/s | Bandwidth(GB/s) |
|------------------------|--------------------|--------------|-----------|-----------|
| M=6144, N=1, K=320  | Speech Recognition | 0.18         | 21.59     | 10.83   |
| M=6144, N=2, K=320  | Speech Recognition | 0.28         | 28.07      | 7.06   |
| M=6144, N=3, K=320  | Speech Recognition | 0.40         | 29.59     | 4.98  |
| M=6144, N=4, K=320  | Speech Recognition | 0.50         | 31.29     | 3.96  |
| M=6144, N=5, K=320  | Speech Recognition | 0.69         | 28.44     | 2.89   |
| M=6144, N=6, K=320  | Speech Recognition | 0.78         | 30.19      | 2.57   |
| M=6144, N=7, K=320  | Speech Recognition | 0.90         | 30.50     | 2.23  |
| M=6144, N=8, K=320  | Speech Recognition | 1.01         | 31.25     | 2.00  |
| M=6144, N=9, K=320  | Speech Recognition | 1.19         | 29.83     | 1.71   |
| M=6144, N=10, K=320  | Speech Recognition | 1.28         | 30.80      | 1.59   |


## iPhone 6

| GEMM                 | Application        | Results (ms) | GigaOps/s | Bandwidth(GB/s) |
|------------------------|--------------------|--------------|-----------|-----------|
| M=6144, N=1, K=320  | Speech Recognition | 0.60         | 6.55     | 3.29   |
| M=6144, N=2, K=320  | Speech Recognition | 0.84         | 9.42      | 2.37   |
| M=6144, N=3, K=320  | Speech Recognition | 0.92         | 12.86     | 2.16  |
| M=6144, N=4, K=320  | Speech Recognition | 1.08         | 14.54     | 1.84  |
| M=6144, N=5, K=320  | Speech Recognition | 1.68         | 11.70     | 1.19   |
| M=6144, N=6, K=320  | Speech Recognition | 1.92         | 12.27      | 1.04   |
| M=6144, N=7, K=320  | Speech Recognition | 2.00         | 13.75     | 1.00  |
| M=6144, N=8, K=320  | Speech Recognition | 2.16         | 14.59     | 0.94  |
| M=6144, N=9, K=320  | Speech Recognition | 2.77         | 12.76     | 0.73   |
| M=6144, N=10, K=320  | Speech Recognition | 3.00         | 13.13      | 0.68   |


## Raspberry Pi 3

| GEMM                | Application        | Results (ms) | GigaOps/s | Bandwidth(GB/s) |
|------------------------|--------------------|--------------|-----------|-----------|
| M=6144, N=1, K=320  | Speech Recognition | 2.50         | 1.58     | 0.79   |
| M=6144, N=2, K=320  | Speech Recognition | 2.89         | 2.72      | 0.69   |
| M=6144, N=3, K=320  | Speech Recognition | 3.34         | 3.53     | 0.59  |
| M=6144, N=4, K=320  | Speech Recognition | 4.11         | 3.82     | 0.48  |
| M=6144, N=5, K=320  | Speech Recognition | 6.64         | 2.96     | 0.30   |
| M=6144, N=6, K=320  | Speech Recognition | 7.02         | 3.36      | 0.29   |
| M=6144, N=7, K=320  | Speech Recognition | 7.48         | 3.68     | 0.27  |
| M=6144, N=8, K=320  | Speech Recognition | 8.25         | 3.81     | 0.24  |
| M=6144, N=9, K=320  | Speech Recognition | 10.75         | 3.29     | 0.19   |
| M=6144, N=10, K=320  | Speech Recognition | 11.11         | 3.54      | 0.18   |



# Kernel Design

Check [doc/kernel-design.md](doc/kernel-design.md) if you are interested in the
details of our ARM assembly kernels.
