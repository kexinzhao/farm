# Performance and Bandwidth

<img src="/doc/images/ip7_kernel_perf_6144_320_bt10.png" width="420"/> <img src="/doc/images/ip7_kernel_bw_6144_320_bt10.png" width="420"/> 
<img src="/doc/images/ip6_kernel_perf_6144_320_bt10.png" width="420"/> <img src="/doc/images/ip6_kernel_bw_6144_320_bt10.png" width="420"/> 
<img src="/doc/images/pi_kernel_perf_6144_320_bt10.png" width="420"/> <img src="/doc/images/pi_kernel_bw_6144_320_bt10.png" width="420"/> 

Figures above show the comparison of our kernels and the gemmlowp library in terms of performance and bandwidth on iPhone 7, iPhone 6, and Raspberry Pi 3, respectively. Our kernels are significantly faster than the gemmlowp counterpart for batch size 1 ~ 4 and continue to perform better than gemmlowp for batch size 5 ~ 10.

# Roofline Model

We use [roofline model](https://en.wikipedia.org/wiki/Roofline_model) to analyze the performances of our kernels. The Roofline graphs below are based on the peak bandwidth and performance of iPhone 7, iPhone 6, and Raspberry Pi 3, respectively. Our kernels are represented as red dots pointed to by the corresponding batch size number on the Roofline graphs. The vertical and horizontal axis value of a red dot are the computational performance and arithmetic intensity of the corresponding kernel, respectively. The arithmetic intensity denotes the number of multiply and accumulate operations per byte of memory traffic and is calculated to be approximately `2 * N` for our batch size `N` kernel. 

We can see from these Roofline graphs that 1) batch size 1 and 2 kernels are bandwidth bound while larger batch size kernels are compute bound and 2) the performance of our batch size 1 kernel is close to the theoretical limit on iPhone 7. 

### iPhone 7

<img src="/doc/images/ip7_roofline_6144_320_bt10.png" width="600"/>

The peak bandwidth of iPhone 7 is approximately 12.5 GB/sec. The peak single-core performance of iPhone 7 is 56.16 Giga operations per second (GigaOps/s), which can be derived as follows. Our kernels primarily use a SIMD instruction to compute four integer multiplications and additions (a total of 8 operations). Because of the out-of-order execution and instruction pipelining capabilities of the A10 Hurricane core in iPhone 7, it can finish three such SIMD instructions per cycle using the three NEON arithmetic-logic units in parallel. Then the peak operations per second is calculated as 

```
8 operations per instruction * 3 instructions per cycle * 2.34 Giga cycles per second = 56.16 GigaOps/s 
```

### iPhone 6

<img src="/doc/images/ip6_roofline_6144_320_bt10.png" width="600"/>

The peak bandwidth of iPhone 6 is approximately 5.5 GB/sec. The peak single-core performance of iPhone 6 is calculated as 

```
8 operations per instruction * 2 instructions per cycle * 1.4 Giga cycles per second = 22.4 GigaOps/s
```

### Raspberry Pi 3

<img src="/doc/images/pi_roofline_6144_320_bt10.png" width="600"/> 

The peak bandwidth of Raspberry Pi 3 is approximately 1.8 GB/sec. The peak single-core performance of Raspberry Pi 3 is calculated as 

```
8 operations per instruction * 1 instruction per cycle * 1.2 Giga cycles per second = 9.6 GigaOps/s 
```
