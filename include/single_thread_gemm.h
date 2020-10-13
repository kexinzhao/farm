#pragma once

#include <cassert>
#include <iostream>
#include <limits>

#include "map.h"

namespace farm {

void gemm_1_kernel_run(const std::uint8_t* lhs_ptr, const std::uint8_t* rhs_ptr,
                       std::uint8_t* dst_ptr, std::size_t run_depth, 
                       std::int16_t* lhs_offset, std::int16_t* rhs_offset, 
                       std::int32_t res_offset, std::int32_t res_mul,
                       std::int32_t res_shift) {
#define FARM_LABEL_LOOP "1"
#define FARM_LABEL_AFTER_LOOP "2"
    asm volatile(
        // Load Rhs 16 x 1 block    
        "ld1 {v28.8b}, [%[rhs_ptr]], #8\n"
        "mov x0, %[lhs_ptr]\n"
        "ld1 {v29.8b}, [%[rhs_ptr]], #8\n"
        "mov x1, x0\n"

        // Load Lhs 8 x 16 block
        "ld1 {v2.8b}, [x1], #8\n"   
        "add x0, x0, %[run_depth]\n"   
        "ld1 {v10.8b}, [x1]\n"
        "mov x1, x0\n"
        "ld1 {v3.8b}, [x1], #8\n"
        "add x0, x0, %[run_depth]\n"
        "ld1 {v11.8b}, [x1]\n"
        "mov x1, x0\n"
        "ld1 {v4.8b}, [x1], #8\n"   
        "add x0, x0, %[run_depth]\n"
        "ld1 {v12.8b}, [x1]\n"   
        "mov x1, x0\n"
        "ld1 {v5.8b}, [x1], #8\n"   
        "add x0, x0, %[run_depth]\n"
        "ld1 {v13.8b}, [x1]\n"
        "mov x1, x0\n" 

        "ld1 {v6.8b}, [x1], #8\n"
        "add x0, x0, %[run_depth]\n"
        "ld1 {v14.8b}, [x1]\n"
        "mov x1, x0\n"
        "ld1 {v7.8b}, [x1], #8\n"
        "add x0, x0, %[run_depth]\n"
        "ld1 {v15.8b}, [x1]\n"
        "mov x1, x0\n"
        "ld1 {v8.8b}, [x1], #8\n"
        "add x0, x0, %[run_depth]\n"
        "ld1 {v16.8b}, [x1]\n"
        "mov x1, x0\n"
        "ld1 {v9.8b}, [x1], #8\n"
        "ld1r {v26.8h}, [%[rhs_offset]]\n"
        "ld1 {v17.8b}, [x1]\n"
        "ld1r {v27.8h}, [%[lhs_offset]]\n"

        // Clear accumulator registors
        "dup v18.4s, wzr\n"
        "mov x2, %[run_depth]\n"
        "dup v19.4s, wzr\n"
        "subs %[run_depth], %[run_depth], #16\n"   
        "dup v20.4s, wzr\n"
        "add %[lhs_ptr], %[lhs_ptr], #16\n"  
        "dup v21.4s, wzr\n"
        "dup v22.4s, wzr\n"
        "dup v23.4s, wzr\n"
        "dup v24.4s, wzr\n"
        "dup v25.4s, wzr\n"

        "beq " FARM_LABEL_AFTER_LOOP "f\n"

        FARM_LABEL_LOOP
        ":\n"

        // Inner Loop
        "uxtl v0.8h, v28.8b\n"
        "uxtl v2.8h, v2.8b\n"
        "uxtl v3.8h, v3.8b\n"
        "uxtl v4.8h, v4.8b\n"
        "uxtl v5.8h, v5.8b\n"
        "uxtl v6.8h, v6.8b\n"
        "uxtl v7.8h, v7.8b\n"
        "uxtl v8.8h, v8.8b\n"
        "uxtl v9.8h, v9.8b\n"
        "add v0.8h, v0.8h, v26.8h\n"
        "add v2.8h, v2.8h, v27.8h\n"
        "add v3.8h, v3.8h, v27.8h\n"
        "add v4.8h, v4.8h, v27.8h\n"
        "add v5.8h, v5.8h, v27.8h\n"
        "add v6.8h, v6.8h, v27.8h\n"
        "add v7.8h, v7.8h, v27.8h\n"
        "add v8.8h, v8.8h, v27.8h\n"
        "add v9.8h, v9.8h, v27.8h\n"

        "smlal v18.4s, v0.4h, v2.4h\n"
        "mov x0, %[lhs_ptr]\n"
        "smlal v19.4s, v0.4h, v3.4h\n"
        "mov x1, x0\n"
        "smlal v20.4s, v0.4h, v4.4h\n"
        "add x1, x1, #8\n"
        "smlal v21.4s, v0.4h, v5.4h\n"
        "ld1 {v28.8b}, [%[rhs_ptr]], #8\n"
        "smlal v22.4s, v0.4h, v6.4h\n"
        "smlal v23.4s, v0.4h, v7.4h\n"
        "smlal v24.4s, v0.4h, v8.4h\n"
        "smlal v25.4s, v0.4h, v9.4h\n"

        "smlal2 v18.4s, v0.8h, v2.8h\n"
        "ld1 {v2.8b}, [x0]\n"
        "add x0, x0, x2\n"
        "smlal2 v19.4s, v0.8h, v3.8h\n"
        "ld1 {v3.8b}, [x0]\n"
        "add x0, x0, x2\n"
        "smlal2 v20.4s, v0.8h, v4.8h\n"
        "ld1 {v4.8b}, [x0]\n"
        "add x0, x0, x2\n"
        "smlal2 v21.4s, v0.8h, v5.8h\n"
        "ld1 {v5.8b}, [x0]\n"
        "add x0, x0, x2\n"
        "smlal2 v22.4s, v0.8h, v6.8h\n"
        "ld1 {v6.8b}, [x0]\n"
        "add x0, x0, x2\n"
        "smlal2 v23.4s, v0.8h, v7.8h\n"
        "ld1 {v7.8b}, [x0]\n"
        "add x0, x0, x2\n"
        "smlal2 v24.4s, v0.8h, v8.8h\n"
        "ld1 {v8.8b}, [x0]\n"
        "add x0, x0, x2\n"
        "smlal2 v25.4s, v0.8h, v9.8h\n"
        "ld1 {v9.8b}, [x0]\n"

        // expand the inner block
        "uxtl v1.8h, v29.8b\n"
        "uxtl v10.8h, v10.8b\n"
        "uxtl v11.8h, v11.8b\n"
        "uxtl v12.8h, v12.8b\n"
        "uxtl v13.8h, v13.8b\n"
        "uxtl v14.8h, v14.8b\n"
        "uxtl v15.8h, v15.8b\n"
        "uxtl v16.8h, v16.8b\n"
        "uxtl v17.8h, v17.8b\n"
        "add v1.8h, v1.8h, v26.8h\n"
        "add v10.8h, v10.8h, v27.8h\n"
        "add v11.8h, v11.8h, v27.8h\n"
        "add v12.8h, v12.8h, v27.8h\n"
        "add v13.8h, v13.8h, v27.8h\n"
        "add v14.8h, v14.8h, v27.8h\n"
        "add v15.8h, v15.8h, v27.8h\n"
        "add v16.8h, v16.8h, v27.8h\n"
        "add v17.8h, v17.8h, v27.8h\n"

        "smlal v18.4s, v1.4h, v10.4h\n"
        "subs %[run_depth], %[run_depth], #16\n"   
        "smlal v19.4s, v1.4h, v11.4h\n"
        "add %[lhs_ptr], %[lhs_ptr], #16\n"  
        "smlal v20.4s, v1.4h, v12.4h\n"
        "ld1 {v29.8b}, [%[rhs_ptr]], #8\n"
        "smlal v21.4s, v1.4h, v13.4h\n"
        "smlal v22.4s, v1.4h, v14.4h\n"
        "smlal v23.4s, v1.4h, v15.4h\n"
        "smlal v24.4s, v1.4h, v16.4h\n"
        "smlal v25.4s, v1.4h, v17.4h\n"

        "smlal2 v18.4s, v1.8h, v10.8h\n"
        "ld1 {v10.8b}, [x1]\n"
        "add x1, x1, x2\n"
        "smlal2 v19.4s, v1.8h, v11.8h\n"
        "ld1 {v11.8b}, [x1]\n"
        "add x1, x1, x2\n"
        "smlal2 v20.4s, v1.8h, v12.8h\n"
        "ld1 {v12.8b}, [x1]\n"
        "add x1, x1, x2\n"
        "smlal2 v21.4s, v1.8h, v13.8h\n"
        "ld1 {v13.8b}, [x1]\n"
        "add x1, x1, x2\n"
        "smlal2 v22.4s, v1.8h, v14.8h\n"
        "ld1 {v14.8b}, [x1]\n"
        "add x1, x1, x2\n"
        "smlal2 v23.4s, v1.8h, v15.8h\n"
        "ld1 {v15.8b}, [x1]\n"
        "add x1, x1, x2\n"
        "smlal2 v24.4s, v1.8h, v16.8h\n"
        "ld1 {v16.8b}, [x1]\n"
        "add x1, x1, x2\n"
        "smlal2 v25.4s, v1.8h, v17.8h\n"
        "ld1 {v17.8b}, [x1]\n"

        "bne " FARM_LABEL_LOOP "b\n"

        FARM_LABEL_AFTER_LOOP
        ":\n"

        // Expand lhs/rhs to 16 bit
        "uxtl v0.8h, v28.8b\n"
        "add v0.8h, v0.8h, v26.8h\n"
        "uxtl v1.8h, v29.8b\n"
        "add v1.8h, v1.8h, v26.8h\n"
        "uxtl v2.8h, v2.8b\n"
        "add v2.8h, v2.8h, v27.8h\n"
        "uxtl v3.8h, v3.8b\n"
        "add v3.8h, v3.8h, v27.8h\n"
        "uxtl v4.8h, v4.8b\n"
        "add v4.8h, v4.8h, v27.8h\n"
        "uxtl v5.8h, v5.8b\n"
        "add v5.8h, v5.8h, v27.8h\n"
        "uxtl v6.8h, v6.8b\n"
        "add v6.8h, v6.8h, v27.8h\n"
        "uxtl v7.8h, v7.8b\n"
        "add v7.8h, v7.8h, v27.8h\n"
        "uxtl v8.8h, v8.8b\n"
        "add v8.8h, v8.8h, v27.8h\n"
        "uxtl v9.8h, v9.8b\n"
        "add v9.8h, v9.8h, v27.8h\n"
        "uxtl v10.8h, v10.8b\n"
        "add v10.8h, v10.8h, v27.8h\n"
        "uxtl v11.8h, v11.8b\n"
        "add v11.8h, v11.8h, v27.8h\n"
        "uxtl v12.8h, v12.8b\n"
        "add v12.8h, v12.8h, v27.8h\n"
        "uxtl v13.8h, v13.8b\n"
        "add v13.8h, v13.8h, v27.8h\n"
        "uxtl v14.8h, v14.8b\n"
        "add v14.8h, v14.8h, v27.8h\n"
        "uxtl v15.8h, v15.8b\n"
        "add v15.8h, v15.8h, v27.8h\n"
        "uxtl v16.8h, v16.8b\n"
        "add v16.8h, v16.8h, v27.8h\n"
        "uxtl v17.8h, v17.8b\n"
        "add v17.8h, v17.8h, v27.8h\n"

        // Multiply-accumulate
        "smlal v18.4s, v0.4h, v2.4h\n"
        "smlal v19.4s, v0.4h, v3.4h\n"
        "smlal v20.4s, v0.4h, v4.4h\n"
        "smlal v21.4s, v0.4h, v5.4h\n"
        "smlal v22.4s, v0.4h, v6.4h\n"
        "smlal v23.4s, v0.4h, v7.4h\n"
        "smlal v24.4s, v0.4h, v8.4h\n"
        "smlal v25.4s, v0.4h, v9.4h\n"

        "smlal2 v18.4s, v0.8h, v2.8h\n"
        "smlal2 v19.4s, v0.8h, v3.8h\n"
        "smlal2 v20.4s, v0.8h, v4.8h\n"
        "smlal2 v21.4s, v0.8h, v5.8h\n"
        "smlal2 v22.4s, v0.8h, v6.8h\n"
        "smlal2 v23.4s, v0.8h, v7.8h\n"
        "smlal2 v24.4s, v0.8h, v8.8h\n"
        "smlal2 v25.4s, v0.8h, v9.8h\n"

        "smlal v18.4s, v1.4h, v10.4h\n"
        "smlal v19.4s, v1.4h, v11.4h\n"
        "smlal v20.4s, v1.4h, v12.4h\n"
        "smlal v21.4s, v1.4h, v13.4h\n"
        "smlal v22.4s, v1.4h, v14.4h\n"
        "smlal v23.4s, v1.4h, v15.4h\n"
        "smlal v24.4s, v1.4h, v16.4h\n"
        "smlal v25.4s, v1.4h, v17.4h\n"

        "smlal2 v18.4s, v1.8h, v10.8h\n"
        "smlal2 v19.4s, v1.8h, v11.8h\n"
        "smlal2 v20.4s, v1.8h, v12.8h\n"
        "smlal2 v21.4s, v1.8h, v13.8h\n"
        "smlal2 v22.4s, v1.8h, v14.8h\n"
        "smlal2 v23.4s, v1.8h, v15.8h\n"
        "smlal2 v24.4s, v1.8h, v16.8h\n"
        "smlal2 v25.4s, v1.8h, v17.8h\n"

        // Add across vector
        "addv s18, v18.4s\n"
        "dup v0.4s, %w[res_offset]\n" 
        "addv s19, v19.4s\n"
        "dup v1.4s, %w[res_mul]\n"
        "addv s20, v20.4s\n"
        "dup v2.4s, %w[res_shift]\n"
        "addv s21, v21.4s\n"
        "mov v18.s[1], v19.s[0]\n"
        "addv s22, v22.4s\n"
        "mov v18.s[2], v20.s[0]\n"
        "addv s23, v23.4s\n"
        "mov v18.s[3], v21.s[0]\n"
        "addv s24, v24.4s\n"
        "add v18.4s, v18.4s, v0.4s\n"
        "addv s25, v25.4s\n"
        "mov v22.s[1], v23.s[0]\n"
        "mov x0, %[dst_ptr]\n"
        "mov v22.s[2], v24.s[0]\n"
        "mul v18.4s, v18.4s, v1.4s\n"
        "mov v22.s[3], v25.s[0]\n"
        "and v3.16b, v18.16b, v2.16b\n"
        "add v22.4s, v22.4s, v0.4s\n"
        "sshr v3.4s, v3.4s, #31\n"
        "mul v22.4s, v22.4s, v1.4s\n"
        "sqadd v18.4s, v18.4s, v3.4s\n"
        "and v4.16b, v22.16b, v2.16b\n"
        "srshl v18.4s, v18.4s, v2.4s\n"
        "sshr v4.4s, v4.4s, #31\n"
        "sqxtn v3.4h, v18.4s\n"
        "sqadd v22.4s, v22.4s, v4.4s\n"
        "mov v18.d[0], v3.d[0]\n"
        "mov v18.d[1], v3.d[0]\n"
        "srshl v22.4s, v22.4s, v2.4s\n"
        "sqxtun v18.8b, v18.8h\n"
        "sqxtn v4.4h, v22.4s\n"
        "mov v22.d[0], v4.d[0]\n"
        "mov v22.d[1], v4.d[0]\n"
        "sqxtun v22.8b, v22.8h\n" 

        // Store accumulators
        "st1 {v18.s}[0], [x0], #4\n"
        "st1 {v22.s}[0], [x0]\n"
#undef FARM_LABEL_LOOP
#undef FARM_LABEL_AFTER_LOOP
        : // outputs
        [lhs_ptr] "+r"(lhs_ptr), [rhs_ptr] "+r"(rhs_ptr),
        [dst_ptr] "+r"(dst_ptr),
        [run_depth] "+r"(run_depth)
        : // inputs
        [lhs_offset] "r"(lhs_offset),
        [rhs_offset] "r"(rhs_offset),
        [res_offset] "r"(res_offset),
        [res_mul] "r"(res_mul),
        [res_shift] "r"(res_shift)    
        : // clobbers
        "cc", "memory", "x0", "x1", "x2", "v0", "v1", "v2", "v3", "v4", "v5", "v6",
        "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16",
        "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26",
        "v27", "v28", "v29");
}

void gemm_2_kernel_run(const std::uint8_t* lhs_ptr, const std::uint8_t* rhs_ptr,
                       std::uint8_t* dst_ptr, std::size_t run_depth, std::size_t rows,
                       std::int16_t* lhs_offset, std::int16_t* rhs_offset,
                       std::int32_t res_offset, std::int32_t res_mul,
                       std::int32_t res_shift) {
#define FARM_LABEL_LOOP "1"
#define FARM_LABEL_AFTER_LOOP "2"
    asm volatile(
        // load lhs 8 x 8 block
        "mov x0, %[lhs_ptr]\n"
        "ld1 {v0.8b}, [x0]\n"
        "add x0, x0, %[run_depth]\n"
        "ld1 {v1.8b}, [x0]\n"
        "add x0, x0, %[run_depth]\n"
        "ld1 {v2.8b}, [x0]\n"
        "add x0, x0, %[run_depth]\n"
        "ld1 {v3.8b}, [x0]\n"
        "add x0, x0, %[run_depth]\n"
        "ld1 {v4.8b}, [x0]\n"
        "add x0, x0, %[run_depth]\n"
        "ld1 {v5.8b}, [x0]\n"
        "add x0, x0, %[run_depth]\n"
        "ld1 {v6.8b}, [x0]\n"
        "add x0, x0, %[run_depth]\n"
        "ld1 {v7.8b}, [x0]\n"

        // load rhs 8 x 2 block
        "mov x1, %[rhs_ptr]\n"
        "ld1 {v28.8b}, [x1]\n"
        "add x1, x1, %[run_depth]\n"
        "ld1 {v29.8b}, [x1]\n"

        // clear accumulator registors
        "dup v10.4s, wzr\n"
        "ld1r {v26.8h}, [%[lhs_offset]]\n"
        "dup v11.4s, wzr\n"
        "ld1r {v27.8h}, [%[rhs_offset]]\n"
        "dup v12.4s, wzr\n"
        "mov x2, %[run_depth]\n"
        "dup v13.4s, wzr\n"
        "subs %[run_depth], %[run_depth], #8\n"
        "dup v14.4s, wzr\n"
        "add %[lhs_ptr], %[lhs_ptr], #8\n"
        "dup v15.4s, wzr\n"
        "add %[rhs_ptr], %[rhs_ptr], #8\n"
        "dup v16.4s, wzr\n"
        "dup v17.4s, wzr\n"
        "dup v18.4s, wzr\n"
        "dup v19.4s, wzr\n"
        "dup v20.4s, wzr\n"
        "dup v21.4s, wzr\n"
        "dup v22.4s, wzr\n"
        "dup v23.4s, wzr\n"
        "dup v24.4s, wzr\n"
        "dup v25.4s, wzr\n"

        "beq " FARM_LABEL_AFTER_LOOP "f\n"

        FARM_LABEL_LOOP
        ":\n"

        // Inner Loop
        "uxtl v8.8h, v28.8b\n"
        "uxtl v9.8h, v29.8b\n"
        "uxtl v0.8h, v0.8b\n"
        "uxtl v1.8h, v1.8b\n"
        "uxtl v2.8h, v2.8b\n"
        "uxtl v3.8h, v3.8b\n"
        "uxtl v4.8h, v4.8b\n"
        "uxtl v5.8h, v5.8b\n"
        "uxtl v6.8h, v6.8b\n"
        "uxtl v7.8h, v7.8b\n"
        "add v8.8h, v8.8h, v27.8h\n"
        "add v9.8h, v9.8h, v27.8h\n"
        "add v0.8h, v0.8h, v26.8h\n"
        "add v1.8h, v1.8h, v26.8h\n"
        "add v2.8h, v2.8h, v26.8h\n"
        "add v3.8h, v3.8h, v26.8h\n"
        "add v4.8h, v4.8h, v26.8h\n"
        "add v5.8h, v5.8h, v26.8h\n"
        "add v6.8h, v6.8h, v26.8h\n"
        "add v7.8h, v7.8h, v26.8h\n"

        "smlal v10.4s, v0.4h, v8.4h\n"
        "mov x0, %[lhs_ptr]\n"
        "smlal v18.4s, v0.4h, v9.4h\n"
        "mov x1, %[rhs_ptr]\n"
        "smlal v11.4s, v1.4h, v8.4h\n"
        "subs %[run_depth], %[run_depth], #8\n"
        "smlal v19.4s, v1.4h, v9.4h\n"
        "ld1 {v28.8b}, [x1]\n"
        "smlal v12.4s, v2.4h, v8.4h\n"
        "add x1, x1, x2\n"
        "smlal v20.4s, v2.4h, v9.4h\n"
        "smlal v13.4s, v3.4h, v8.4h\n"
        "smlal v21.4s, v3.4h, v9.4h\n"

        "smlal2 v10.4s, v0.8h, v8.8h\n"
        "smlal2 v18.4s, v0.8h, v9.8h\n"
        "ld1 {v0.8b}, [x0]\n"
        "smlal2 v11.4s, v1.8h, v8.8h\n"
        "add x0, x0, x2\n"
        "smlal2 v19.4s, v1.8h, v9.8h\n"
        "ld1 {v1.8b}, [x0]\n"
        "smlal2 v12.4s, v2.8h, v8.8h\n"
        "add x0, x0, x2\n"
        "smlal2 v20.4s, v2.8h, v9.8h\n"
        "ld1 {v2.8b}, [x0]\n"
        "smlal2 v13.4s, v3.8h, v8.8h\n"
        "add x0, x0, x2\n"
        "smlal2 v21.4s, v3.8h, v9.8h\n"
        "ld1 {v3.8b}, [x0]\n"

        "smlal v14.4s, v4.4h, v8.4h\n"
        "add %[lhs_ptr], %[lhs_ptr], #8\n"
        "smlal v22.4s, v4.4h, v9.4h\n"    
        "add %[rhs_ptr], %[rhs_ptr], #8\n"
        "smlal v15.4s, v5.4h, v8.4h\n"
        "add x0, x0, x2\n"
        "smlal v23.4s, v5.4h, v9.4h\n"
        "ld1 {v29.8b}, [x1]\n"
        "smlal v16.4s, v6.4h, v8.4h\n"
        "smlal v24.4s, v6.4h, v9.4h\n"
        "smlal v17.4s, v7.4h, v8.4h\n"
        "smlal v25.4s, v7.4h, v9.4h\n"

        "smlal2 v14.4s, v4.8h, v8.8h\n"
        "smlal2 v22.4s, v4.8h, v9.8h\n"
        "ld1 {v4.8b}, [x0]\n"
        "smlal2 v15.4s, v5.8h, v8.8h\n"
        "add x0, x0, x2\n"
        "smlal2 v23.4s, v5.8h, v9.8h\n"
        "ld1 {v5.8b}, [x0]\n"
        "smlal2 v16.4s, v6.8h, v8.8h\n"
        "add x0, x0, x2\n"
        "smlal2 v24.4s, v6.8h, v9.8h\n"
        "ld1 {v6.8b}, [x0]\n"
        "smlal2 v17.4s, v7.8h, v8.8h\n"
        "add x0, x0, x2\n"
        "smlal2 v25.4s, v7.8h, v9.8h\n"
        "ld1 {v7.8b}, [x0]\n"

        "bne " FARM_LABEL_LOOP "b\n"

        FARM_LABEL_AFTER_LOOP
        ":\n"

        // Expand lhs/rhs to 16 bit and add offset
        "uxtl v8.8h, v28.8b\n"
        "add v8.8h, v8.8h, v27.8h\n"
        "uxtl v9.8h, v29.8b\n"
        "add v9.8h, v9.8h, v27.8h\n"
        "uxtl v0.8h, v0.8b\n"
        "add v0.8h, v0.8h, v26.8h\n"
        "uxtl v1.8h, v1.8b\n"
        "add v1.8h, v1.8h, v26.8h\n"
        "uxtl v2.8h, v2.8b\n"
        "add v2.8h, v2.8h, v26.8h\n"
        "uxtl v3.8h, v3.8b\n"
        "add v3.8h, v3.8h, v26.8h\n"
        "uxtl v4.8h, v4.8b\n"
        "add v4.8h, v4.8h, v26.8h\n"
        "uxtl v5.8h, v5.8b\n"
        "add v5.8h, v5.8h, v26.8h\n"
        "uxtl v6.8h, v6.8b\n"
        "add v6.8h, v6.8h, v26.8h\n"
        "uxtl v7.8h, v7.8b\n"
        "add v7.8h, v7.8h, v26.8h\n"

        "smlal v10.4s, v0.4h, v8.4h\n"
        "smlal v11.4s, v1.4h, v8.4h\n"
        "smlal v12.4s, v2.4h, v8.4h\n"
        "smlal v13.4s, v3.4h, v8.4h\n"
        "smlal v14.4s, v4.4h, v8.4h\n"
        "smlal v15.4s, v5.4h, v8.4h\n"
        "smlal v16.4s, v6.4h, v8.4h\n"
        "smlal v17.4s, v7.4h, v8.4h\n"

        "smlal2 v10.4s, v0.8h, v8.8h\n"
        "smlal2 v11.4s, v1.8h, v8.8h\n"
        "smlal2 v12.4s, v2.8h, v8.8h\n"
        "smlal2 v13.4s, v3.8h, v8.8h\n"
        "smlal2 v14.4s, v4.8h, v8.8h\n"
        "smlal2 v15.4s, v5.8h, v8.8h\n"
        "smlal2 v16.4s, v6.8h, v8.8h\n"
        "smlal2 v17.4s, v7.8h, v8.8h\n"

        "smlal v18.4s, v0.4h, v9.4h\n"
        "smlal v19.4s, v1.4h, v9.4h\n"
        "smlal v20.4s, v2.4h, v9.4h\n"
        "smlal v21.4s, v3.4h, v9.4h\n"
        "smlal v22.4s, v4.4h, v9.4h\n"
        "smlal v23.4s, v5.4h, v9.4h\n"
        "smlal v24.4s, v6.4h, v9.4h\n"
        "smlal v25.4s, v7.4h, v9.4h\n"

        "smlal2 v18.4s, v0.8h, v9.8h\n"
        "smlal2 v19.4s, v1.8h, v9.8h\n"
        "smlal2 v20.4s, v2.8h, v9.8h\n"
        "smlal2 v21.4s, v3.8h, v9.8h\n"
        "smlal2 v22.4s, v4.8h, v9.8h\n"
        "smlal2 v23.4s, v5.8h, v9.8h\n"
        "smlal2 v24.4s, v6.8h, v9.8h\n"
        "smlal2 v25.4s, v7.8h, v9.8h\n"

        // Add across vector
        "addv s10, v10.4s\n"
        "dup v0.4s, %w[res_offset]\n"
        "addv s11, v11.4s\n"
        "dup v1.4s, %w[res_mul]\n"
        "addv s12, v12.4s\n"
        "dup v2.4s, %w[res_shift]\n"
        "addv s13, v13.4s\n"
        "mov v10.s[1], v11.s[0]\n"
        "addv s14, v14.4s\n"
        "mov v10.s[2], v12.s[0]\n"
        "addv s15, v15.4s\n"
        "mov v10.s[3], v13.s[0]\n"
        "addv s16, v16.4s\n"
        "add v10.4s, v10.4s, v0.4s\n"
        "addv s17, v17.4s\n"
        "mov v14.s[1], v15.s[0]\n"
        "mov x0, %[dst_ptr]\n"
        "mov v14.s[2], v16.s[0]\n"
        "mul v10.4s, v10.4s, v1.4s\n"
        "mov v14.s[3], v17.s[0]\n" 
        "and v3.16b, v10.16b, v2.16b\n"
        "add v14.4s, v14.4s, v0.4s\n"
        "sshr v3.4s, v3.4s, #31\n"
        "mul v14.4s, v14.4s, v1.4s\n"
        "sqadd v10.4s, v10.4s, v3.4s\n"
        "and v4.16b, v14.16b, v2.16b\n"
        "srshl v10.4s, v10.4s, v2.4s\n"
        "sshr v4.4s, v4.4s, #31\n"
        "sqxtn v3.4h, v10.4s\n"
        "sqadd v14.4s, v14.4s, v4.4s\n"
        "mov v10.d[0], v3.d[0]\n"
        "mov v10.d[1], v3.d[0]\n"
        "srshl v14.4s, v14.4s, v2.4s\n"
        "sqxtun v10.8b, v10.8h\n"
        "sqxtn v4.4h, v14.4s\n"
        "mov x1, x0\n"
        "mov v14.d[0], v4.d[0]\n"
        "mov v14.d[1], v4.d[0]\n"
        "sqxtun v14.8b, v14.8h\n"
        "st1 {v10.s}[0], [x0], #4\n"
        "add x1, x1, %[rows]\n"
        "st1 {v14.s}[0], [x0]\n"

        "addv s18, v18.4s\n"
        "addv s19, v19.4s\n"
        "addv s20, v20.4s\n"
        "addv s21, v21.4s\n"
        "addv s22, v22.4s\n"
        "addv s23, v23.4s\n"
        "addv s24, v24.4s\n"
        "addv s25, v25.4s\n"
        "mov v18.s[1], v19.s[0]\n"
        "mov v18.s[2], v20.s[0]\n"
        "mov v18.s[3], v21.s[0]\n"
        "mov v22.s[1], v23.s[0]\n"
        "mov v22.s[2], v24.s[0]\n"
        "mov v22.s[3], v25.s[0]\n"
        "add v18.4s, v18.4s, v0.4s\n"
        "add v22.4s, v22.4s, v0.4s\n"
        "mul v18.4s, v18.4s, v1.4s\n"
        "mul v22.4s, v22.4s, v1.4s\n"
        "and v3.16b, v18.16b, v2.16b\n"
        "and v4.16b, v22.16b, v2.16b\n"
        "sshr v3.4s, v3.4s, #31\n"
        "sshr v4.4s, v4.4s, #31\n"
        "sqadd v18.4s, v18.4s, v3.4s\n"
        "sqadd v22.4s, v22.4s, v4.4s\n"
        "srshl v18.4s, v18.4s, v2.4s\n"
        "srshl v22.4s, v22.4s, v2.4s\n"
        "sqxtn v3.4h, v18.4s\n"
        "sqxtn v4.4h, v22.4s\n"
        "mov v18.d[0], v3.d[0]\n"
        "mov v18.d[1], v3.d[0]\n"
        "mov v22.d[0], v4.d[0]\n"
        "mov v22.d[1], v4.d[0]\n"
        "sqxtun v18.8b, v18.8h\n"
        "sqxtun v22.8b, v22.8h\n"
        "st1 {v18.s}[0], [x1], #4\n"
        "st1 {v22.s}[0], [x1]\n"
#undef FARM_LABEL_LOOP
#undef FARM_LABEL_AFTER_LOOP
        : // outputs
        [lhs_ptr] "+r"(lhs_ptr), [rhs_ptr] "+r"(rhs_ptr),
        [dst_ptr] "+r"(dst_ptr),
        [run_depth] "+r"(run_depth)
        : // inputs
        [rows] "r"(rows), // this is used when storing the result
        [lhs_offset] "r"(lhs_offset),
        [rhs_offset] "r"(rhs_offset),
        [res_offset] "r"(res_offset),
        [res_mul] "r"(res_mul),
        [res_shift] "r"(res_shift)    
        : // clobbers
        "cc", "memory", "x0", "x1", "x2", "v0", "v1", "v2", "v3", "v4", "v5", "v6",
        "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16",
        "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26",
        "v27", "v28", "v29");
}

void gemm_3_kernel_run(const std::uint8_t* lhs_ptr, const std::uint8_t* rhs_ptr,
                       std::uint8_t* dst_ptr, std::size_t run_depth, std::size_t rows,
                       std::int16_t* lhs_offset, std::int16_t* rhs_offset,
                       std::int32_t res_offset, std::int32_t res_mul,
                       std::int32_t res_shift) { 
#define FARM_LABEL_LOOP "1"
#define FARM_LABEL_AFTER_LOOP "2"
    asm volatile(
        // load lhs block
        "mov x0, %[lhs_ptr]\n"
        "mov x1, x0\n"
        "ld1 {v0.8b}, [x1], #8\n"
        "add x0, x0, %[run_depth]\n"
        "ld1 {v28.8b}, [x1]\n"
        "mov x1, x0\n"
        "ld1 {v1.8b}, [x1], #8\n"
        "add x0, x0, %[run_depth]\n"
        "ld1 {v29.8b}, [x1]\n"
        "mov x1, x0\n"
        "ld1 {v2.8b}, [x1], #8\n"
        "add x0, x0, %[run_depth]\n"
        "ld1 {v30.8b}, [x1]\n"
        "mov x1, x0\n"
        "ld1 {v3.8b}, [x1], #8\n"
        "mov x0, %[rhs_ptr]\n"
        "ld1 {v31.8b}, [x1]\n"
        "mov x1, x0\n"

        // load rhs block
        "ld1 {v8.8b}, [x1], #8\n"
        "add x0, x0, %[run_depth]\n"
        "ld1 {v9.8b}, [x1]\n"
        "mov x1, x0\n"
        "ld1 {v10.8b}, [x1], #8\n"
        "add x0, x0, %[run_depth]\n"
        "ld1 {v11.8b}, [x1]\n"
        "mov x1, x0\n"
        "ld1 {v12.8b}, [x1], #8\n"
        "ld1r {v26.8h}, [%[lhs_offset]]\n"
        "ld1 {v13.8b}, [x1]\n"
        "ld1r {v27.8h}, [%[rhs_offset]]\n"

        // clear accumulator
        "dup v14.4s, wzr\n"
        "mov x2, %[run_depth]\n"
        "dup v15.4s, wzr\n"
        "subs %[run_depth], %[run_depth], #16\n"
        "dup v16.4s, wzr\n"
        "add %[lhs_ptr], %[lhs_ptr], #16\n"
        "dup v17.4s, wzr\n"
        "add %[rhs_ptr], %[rhs_ptr], #16\n"
        "dup v18.4s, wzr\n"
        "dup v19.4s, wzr\n"
        "dup v20.4s, wzr\n"
        "dup v21.4s, wzr\n"
        "dup v22.4s, wzr\n"
        "dup v23.4s, wzr\n"
        "dup v24.4s, wzr\n"
        "dup v25.4s, wzr\n"

        "beq " FARM_LABEL_AFTER_LOOP "f\n"

        FARM_LABEL_LOOP
        ":\n"

        // Inner Loop first width
        "uxtl v0.8h, v0.8b\n"
        "uxtl v1.8h, v1.8b\n"
        "uxtl v2.8h, v2.8b\n"
        "uxtl v3.8h, v3.8b\n"
        "uxtl v8.8h, v8.8b\n"
        "uxtl v10.8h, v10.8b\n"
        "uxtl v12.8h, v12.8b\n"
        "add v0.8h, v0.8h, v26.8h\n"
        "add v1.8h, v1.8h, v26.8h\n"
        "add v2.8h, v2.8h, v26.8h\n"
        "add v3.8h, v3.8h, v26.8h\n"
        "add v8.8h, v8.8h, v27.8h\n"
        "add v10.8h, v10.8h, v27.8h\n"
        "add v12.8h, v12.8h, v27.8h\n"

        "smlal v14.4s, v0.4h, v8.4h\n"
        "mov x0, %[lhs_ptr]\n"
        "smlal v15.4s, v1.4h, v8.4h\n"
        "add %[lhs_ptr], %[lhs_ptr], #8\n"
        "smlal v16.4s, v2.4h, v8.4h\n"
        "mov x1, %[rhs_ptr]\n"
        "smlal v17.4s, v3.4h, v8.4h\n"
        "add %[rhs_ptr], %[rhs_ptr], #8\n"
        "smlal v18.4s, v0.4h, v10.4h\n"
        "subs %[run_depth], %[run_depth], #16\n"
        "smlal v19.4s, v1.4h, v10.4h\n"
        "smlal v20.4s, v2.4h, v10.4h\n"
        "smlal v21.4s, v3.4h, v10.4h\n"
        "smlal v22.4s, v0.4h, v12.4h\n"
        "smlal v23.4s, v1.4h, v12.4h\n"
        "smlal v24.4s, v2.4h, v12.4h\n"
        "smlal v25.4s, v3.4h, v12.4h\n"

        "smlal2 v14.4s, v0.8h, v8.8h\n"
        "smlal2 v15.4s, v1.8h, v8.8h\n"
        "smlal2 v16.4s, v2.8h, v8.8h\n"
        "smlal2 v17.4s, v3.8h, v8.8h\n"
        "ld1 {v8.8b}, [x1]\n"
        "smlal2 v18.4s, v0.8h, v10.8h\n"
        "add x1, x1, x2\n"
        "smlal2 v19.4s, v1.8h, v10.8h\n"
        "smlal2 v20.4s, v2.8h, v10.8h\n"
        "smlal2 v21.4s, v3.8h, v10.8h\n"
        "ld1 {v10.8b}, [x1]\n"
        "smlal2 v22.4s, v0.8h, v12.8h\n"
        "add x1, x1, x2\n"
        "smlal2 v23.4s, v1.8h, v12.8h\n"
        "smlal2 v24.4s, v2.8h, v12.8h\n"
        "smlal2 v25.4s, v3.8h, v12.8h\n"
        "ld1 {v12.8b}, [x1]\n"

        // Inner Loop second width
        "uxtl v4.8h, v28.8b\n"
        "ld1 {v0.8b}, [x0]\n"
        "uxtl v5.8h, v29.8b\n"
        "add x0, x0, x2\n"
        "uxtl v6.8h, v30.8b\n"
        "ld1 {v1.8b}, [x0]\n"
        "uxtl v7.8h, v31.8b\n"
        "add x0, x0, x2\n"
        "uxtl v9.8h, v9.8b\n"
        "ld1 {v2.8b}, [x0]\n"
        "uxtl v11.8h, v11.8b\n"
        "add x0, x0, x2\n"
        "uxtl v13.8h, v13.8b\n"
        "ld1 {v3.8b}, [x0]\n"
        "add v4.8h, v4.8h, v26.8h\n"
        "mov x0, %[lhs_ptr]\n"
        "add v5.8h, v5.8h, v26.8h\n"
        "add %[lhs_ptr], %[lhs_ptr], #8\n"
        "add v6.8h, v6.8h, v26.8h\n"
        "ld1 {v28.8b}, [x0]\n"
        "add v7.8h, v7.8h, v26.8h\n"
        "add x0, x0, x2\n"
        "add v9.8h, v9.8h, v27.8h\n"
        "ld1 {v29.8b}, [x0]\n"
        "add v11.8h, v11.8h, v27.8h\n"
        "add x0, x0, x2\n"
        "add v13.8h, v13.8h, v27.8h\n"
        "ld1 {v30.8b}, [x0]\n"

        "smlal v14.4s, v4.4h, v9.4h\n"
        "add x0, x0, x2\n"
        "smlal v15.4s, v5.4h, v9.4h\n"
        "ld1 {v31.8b}, [x0]\n"
        "smlal v16.4s, v6.4h, v9.4h\n"
        "mov x1, %[rhs_ptr]\n"
        "smlal v17.4s, v7.4h, v9.4h\n"
        "add %[rhs_ptr], %[rhs_ptr], #8\n"
        "smlal v18.4s, v4.4h, v11.4h\n"
        "smlal v19.4s, v5.4h, v11.4h\n"
        "smlal v20.4s, v6.4h, v11.4h\n"
        "smlal v21.4s, v7.4h, v11.4h\n"
        "smlal v22.4s, v4.4h, v13.4h\n"
        "smlal v23.4s, v5.4h, v13.4h\n"
        "smlal v24.4s, v6.4h, v13.4h\n"
        "smlal v25.4s, v7.4h, v13.4h\n"

        "smlal2 v14.4s, v4.8h, v9.8h\n"
        "smlal2 v15.4s, v5.8h, v9.8h\n"
        "smlal2 v16.4s, v6.8h, v9.8h\n"
        "smlal2 v17.4s, v7.8h, v9.8h\n"
        "ld1 {v9.8b}, [x1]\n"
        "smlal2 v18.4s, v4.8h, v11.8h\n"
        "add x1, x1, x2\n"
        "smlal2 v19.4s, v5.8h, v11.8h\n"
        "smlal2 v20.4s, v6.8h, v11.8h\n"
        "smlal2 v21.4s, v7.8h, v11.8h\n"
        "ld1 {v11.8b}, [x1]\n"
        "smlal2 v22.4s, v4.8h, v13.8h\n"
        "add x1, x1, x2\n"
        "smlal2 v23.4s, v5.8h, v13.8h\n"
        "smlal2 v24.4s, v6.8h, v13.8h\n"
        "smlal2 v25.4s, v7.8h, v13.8h\n"
        "ld1 {v13.8b}, [x1]\n"

        "bne " FARM_LABEL_LOOP "b\n"

        FARM_LABEL_AFTER_LOOP
        ":\n"
         
        // Expand lhs/rhs to 16 bit and add offset
        "uxtl v0.8h, v0.8b\n"
        "uxtl v1.8h, v1.8b\n"
        "uxtl v2.8h, v2.8b\n"
        "uxtl v3.8h, v3.8b\n"
        "uxtl v8.8h, v8.8b\n"
        "uxtl v10.8h, v10.8b\n"
        "uxtl v12.8h, v12.8b\n"
        "add v0.8h, v0.8h, v26.8h\n"
        "add v1.8h, v1.8h, v26.8h\n"
        "add v2.8h, v2.8h, v26.8h\n"
        "add v3.8h, v3.8h, v26.8h\n"
        "add v8.8h, v8.8h, v27.8h\n"
        "add v10.8h, v10.8h, v27.8h\n"
        "add v12.8h, v12.8h, v27.8h\n"

        "smlal v14.4s, v0.4h, v8.4h\n"
        "smlal v15.4s, v1.4h, v8.4h\n"
        "smlal v16.4s, v2.4h, v8.4h\n"
        "smlal v17.4s, v3.4h, v8.4h\n"
        "smlal v18.4s, v0.4h, v10.4h\n"
        "smlal v19.4s, v1.4h, v10.4h\n"
        "smlal v20.4s, v2.4h, v10.4h\n"
        "smlal v21.4s, v3.4h, v10.4h\n"
        "smlal v22.4s, v0.4h, v12.4h\n"
        "smlal v23.4s, v1.4h, v12.4h\n"
        "smlal v24.4s, v2.4h, v12.4h\n"
        "smlal v25.4s, v3.4h, v12.4h\n"

        "smlal2 v14.4s, v0.8h, v8.8h\n"
        "smlal2 v15.4s, v1.8h, v8.8h\n"
        "smlal2 v16.4s, v2.8h, v8.8h\n"
        "smlal2 v17.4s, v3.8h, v8.8h\n"
        "smlal2 v18.4s, v0.8h, v10.8h\n"
        "smlal2 v19.4s, v1.8h, v10.8h\n"
        "smlal2 v20.4s, v2.8h, v10.8h\n"
        "smlal2 v21.4s, v3.8h, v10.8h\n"
        "smlal2 v22.4s, v0.8h, v12.8h\n"
        "smlal2 v23.4s, v1.8h, v12.8h\n"
        "smlal2 v24.4s, v2.8h, v12.8h\n"
        "smlal2 v25.4s, v3.8h, v12.8h\n"

        "uxtl v4.8h, v28.8b\n"
        "uxtl v5.8h, v29.8b\n"
        "uxtl v6.8h, v30.8b\n"
        "uxtl v7.8h, v31.8b\n"
        "uxtl v9.8h, v9.8b\n"
        "uxtl v11.8h, v11.8b\n"
        "uxtl v13.8h, v13.8b\n"
        "add v4.8h, v4.8h, v26.8h\n"
        "add v5.8h, v5.8h, v26.8h\n"
        "add v6.8h, v6.8h, v26.8h\n"
        "add v7.8h, v7.8h, v26.8h\n"
        "add v9.8h, v9.8h, v27.8h\n"
        "add v11.8h, v11.8h, v27.8h\n"
        "add v13.8h, v13.8h, v27.8h\n"

        "smlal v14.4s, v4.4h, v9.4h\n"
        "smlal v15.4s, v5.4h, v9.4h\n"
        "smlal v16.4s, v6.4h, v9.4h\n"
        "smlal v17.4s, v7.4h, v9.4h\n"
        "smlal v18.4s, v4.4h, v11.4h\n"
        "smlal v19.4s, v5.4h, v11.4h\n"
        "smlal v20.4s, v6.4h, v11.4h\n"
        "smlal v21.4s, v7.4h, v11.4h\n"
        "smlal v22.4s, v4.4h, v13.4h\n"
        "smlal v23.4s, v5.4h, v13.4h\n"
        "smlal v24.4s, v6.4h, v13.4h\n"
        "smlal v25.4s, v7.4h, v13.4h\n"

        "smlal2 v14.4s, v4.8h, v9.8h\n"
        "smlal2 v15.4s, v5.8h, v9.8h\n"
        "smlal2 v16.4s, v6.8h, v9.8h\n"
        "smlal2 v17.4s, v7.8h, v9.8h\n"
        "smlal2 v18.4s, v4.8h, v11.8h\n"
        "smlal2 v19.4s, v5.8h, v11.8h\n"
        "smlal2 v20.4s, v6.8h, v11.8h\n"
        "smlal2 v21.4s, v7.8h, v11.8h\n"
        "smlal2 v22.4s, v4.8h, v13.8h\n"
        "smlal2 v23.4s, v5.8h, v13.8h\n"
        "smlal2 v24.4s, v6.8h, v13.8h\n"
        "smlal2 v25.4s, v7.8h, v13.8h\n"

        // Generate final output first column
        "addv s14, v14.4s\n"
        "dup v0.4s, %w[res_offset]\n"
        "addv s15, v15.4s\n"
        "dup v1.4s, %w[res_mul]\n"
        "addv s16, v16.4s\n"
        "dup v2.4s, %w[res_shift]\n"
        "addv s17, v17.4s\n"
        "mov v14.s[1], v15.s[0]\n"
        "mov v14.s[2], v16.s[0]\n"
        "mov v14.s[3], v17.s[0]\n"
        "add v14.4s, v14.4s, v0.4s\n"
        "mul v14.4s, v14.4s, v1.4s\n"
        "and v3.16b, v14.16b, v2.16b\n"
        "sshr v3.4s, v3.4s, #31\n"
        "sqadd v14.4s, v14.4s, v3.4s\n"
        "srshl v14.4s, v14.4s, v2.4s\n"
        "sqxtn v3.4h, v14.4s\n"
        "mov v14.d[0], v3.d[0]\n"
        "mov v14.d[1], v3.d[0]\n"
        "sqxtun v14.8b, v14.8h\n"
        "st1 {v14.s}[0], [%[dst_ptr]]\n"
        "add %[dst_ptr], %[dst_ptr], %[rows]\n"

        // Second column
        "addv s18, v18.4s\n"
        "addv s19, v19.4s\n"
        "addv s20, v20.4s\n"
        "addv s21, v21.4s\n"
        "mov v18.s[1], v19.s[0]\n"
        "mov v18.s[2], v20.s[0]\n"
        "mov v18.s[3], v21.s[0]\n"
        "add v18.4s, v18.4s, v0.4s\n"
        "mul v18.4s, v18.4s, v1.4s\n"
        "and v3.16b, v18.16b, v2.16b\n"
        "sshr v3.4s, v3.4s, #31\n"
        "sqadd v18.4s, v18.4s, v3.4s\n"
        "srshl v18.4s, v18.4s, v2.4s\n"
        "sqxtn v3.4h, v18.4s\n"
        "mov v18.d[0], v3.d[0]\n"
        "mov v18.d[1], v3.d[0]\n"
        "sqxtun v18.8b, v18.8h\n"
        "st1 {v18.s}[0], [%[dst_ptr]]\n"
        "add %[dst_ptr], %[dst_ptr], %[rows]\n"

        // Third column
        "addv s22, v22.4s\n"
        "addv s23, v23.4s\n"
        "addv s24, v24.4s\n"
        "addv s25, v25.4s\n"
        "mov v22.s[1], v23.s[0]\n"
        "mov v22.s[2], v24.s[0]\n"
        "mov v22.s[3], v25.s[0]\n"
        "add v22.4s, v22.4s, v0.4s\n"
        "mul v22.4s, v22.4s, v1.4s\n"
        "and v3.16b, v22.16b, v2.16b\n"
        "sshr v3.4s, v3.4s, #31\n"
        "sqadd v22.4s, v22.4s, v3.4s\n"
        "srshl v22.4s, v22.4s, v2.4s\n"
        "sqxtn v3.4h, v22.4s\n"
        "mov v22.d[0], v3.d[0]\n"
        "mov v22.d[1], v3.d[0]\n"
        "sqxtun v22.8b, v22.8h\n"
        "st1 {v22.s}[0], [%[dst_ptr]]\n"
#undef FARM_LABEL_LOOP
#undef FARM_LABEL_AFTER_LOOP
        : // outputs
        [lhs_ptr] "+r"(lhs_ptr), [rhs_ptr] "+r"(rhs_ptr),
        [dst_ptr] "+r"(dst_ptr),
        [run_depth] "+r"(run_depth)
        : // inputs
        [rows] "r"(rows), // this is used when storing the result
        [lhs_offset] "r"(lhs_offset),
        [rhs_offset] "r"(rhs_offset),
        [res_offset] "r"(res_offset),
        [res_mul] "r"(res_mul),
        [res_shift] "r"(res_shift)    
        : // clobbers
        "cc", "memory", "x0", "x1", "x2", "v0", "v1", "v2", "v3", "v4", "v5", "v6",
        "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16",
        "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26",
        "v27", "v28", "v29", "v30", "v31");
}        


void gemm_4_kernel_run(const std::uint8_t* lhs_ptr, const std::uint8_t* rhs_ptr,
                       std::uint8_t* dst_ptr, std::size_t run_depth, std::size_t rows,
                       std::int16_t* lhs_offset, std::int16_t* rhs_offset,
                       std::int32_t res_offset, std::int32_t res_mul,
                       std::int32_t res_shift) { 
#define FARM_LABEL_LOOP "1"
#define FARM_LABEL_AFTER_LOOP "2"
    asm volatile(
    	// load lhs 4 x 8 block
    	"mov x0, %[lhs_ptr]\n"
    	"ld1 {v0.8b}, [x0]\n"
    	"add x0, x0, %[run_depth]\n"
    	"ld1 {v1.8b}, [x0]\n"
    	"add x0, x0, %[run_depth]\n"
    	"ld1 {v2.8b}, [x0]\n"
    	"add x0, x0, %[run_depth]\n"
    	"ld1 {v3.8b}, [x0]\n"

        // load rhs 8 x 4 block
        "mov x1, %[rhs_ptr]\n"
        "ld1 {v26.8b}, [x1]\n"
        "add x1, x1, %[run_depth]\n"
        "ld1 {v27.8b}, [x1]\n"
        "add x1, x1, %[run_depth]\n"
        "ld1 {v28.8b}, [x1]\n"
        "add x1, x1, %[run_depth]\n"
        "ld1 {v29.8b}, [x1]\n"

        // clear accumulator registors
        "dup v8.4s, wzr\n"
        "ld1r {v24.8h}, [%[lhs_offset]]\n"
        "dup v9.4s, wzr\n"
        "ld1r {v25.8h}, [%[rhs_offset]]\n"
        "dup v10.4s, wzr\n"
        "mov x2, %[run_depth]\n"
        "dup v11.4s, wzr\n"
        "subs %[run_depth], %[run_depth], #8\n"
        "dup v12.4s, wzr\n"
        "add %[lhs_ptr], %[lhs_ptr], #8\n"
        "dup v13.4s, wzr\n"
        "add %[rhs_ptr], %[rhs_ptr], #8\n"        
        "dup v14.4s, wzr\n"
        "dup v15.4s, wzr\n"
        "dup v16.4s, wzr\n"
        "dup v17.4s, wzr\n"
        "dup v18.4s, wzr\n"
        "dup v19.4s, wzr\n"
        "dup v20.4s, wzr\n"
        "dup v21.4s, wzr\n"
        "dup v22.4s, wzr\n"
        "dup v23.4s, wzr\n"

        "beq " FARM_LABEL_AFTER_LOOP "f\n"

        FARM_LABEL_LOOP
        ":\n"

        // Inner Loop
        "uxtl v4.8h, v26.8b\n"
        "uxtl v5.8h, v27.8b\n"
        "uxtl v6.8h, v28.8b\n"
        "uxtl v7.8h, v29.8b\n"
        "uxtl v0.8h, v0.8b\n"
        "uxtl v1.8h, v1.8b\n"
        "uxtl v2.8h, v2.8b\n"
        "uxtl v3.8h, v3.8b\n"
        "add v4.8h, v4.8h, v25.8h\n"
        "add v5.8h, v5.8h, v25.8h\n"
        "add v6.8h, v6.8h, v25.8h\n"
        "add v7.8h, v7.8h, v25.8h\n"     
        "add v0.8h, v0.8h, v24.8h\n"
        "add v1.8h, v1.8h, v24.8h\n"
        "add v2.8h, v2.8h, v24.8h\n"
        "add v3.8h, v3.8h, v24.8h\n"   

        "smlal v8.4s, v0.4h, v4.4h\n"
        "mov x0, %[lhs_ptr]\n"
        "smlal v12.4s, v0.4h, v5.4h\n"
        "mov x1, %[rhs_ptr]\n"
        "smlal v16.4s, v0.4h, v6.4h\n"
        "subs %[run_depth], %[run_depth], #8\n"
        "smlal v20.4s, v0.4h, v7.4h\n"
        "ld1 {v26.8b}, [x1]\n"
        "smlal v9.4s, v1.4h, v4.4h\n"
        "add x1, x1, x2\n"
        "smlal v13.4s, v1.4h, v5.4h\n"
        "ld1 {v27.8b}, [x1]\n"
        "smlal v17.4s, v1.4h, v6.4h\n"
        "add x1, x1, x2\n"
        "smlal v21.4s, v1.4h, v7.4h\n"
        "ld1 {v28.8b}, [x1]\n"
        "smlal v10.4s, v2.4h, v4.4h\n"
        "add x1, x1, x2\n"
        "smlal v14.4s, v2.4h, v5.4h\n"
        "ld1 {v29.8b}, [x1]\n"
        "smlal v18.4s, v2.4h, v6.4h\n"
        "add %[lhs_ptr], %[lhs_ptr], #8\n"
        "smlal v22.4s, v2.4h, v7.4h\n"
        "add %[rhs_ptr], %[rhs_ptr], #8\n"
        "smlal v11.4s, v3.4h, v4.4h\n"
        "smlal v15.4s, v3.4h, v5.4h\n"
        "smlal v19.4s, v3.4h, v6.4h\n"
        "smlal v23.4s, v3.4h, v7.4h\n"

        "smlal2 v8.4s, v0.8h, v4.8h\n"
        "smlal2 v12.4s, v0.8h, v5.8h\n"
        "smlal2 v16.4s, v0.8h, v6.8h\n"
        "smlal2 v20.4s, v0.8h, v7.8h\n"
        "ld1 {v0.8b}, [x0]\n"
        "smlal2 v9.4s, v1.8h, v4.8h\n"
        "add x0, x0, x2\n"
        "smlal2 v13.4s, v1.8h, v5.8h\n"
        "smlal2 v17.4s, v1.8h, v6.8h\n"
        "smlal2 v21.4s, v1.8h, v7.8h\n"
        "ld1 {v1.8b}, [x0]\n"
        "smlal2 v10.4s, v2.8h, v4.8h\n"
        "add x0, x0, x2\n"
        "smlal2 v14.4s, v2.8h, v5.8h\n"
        "smlal2 v18.4s, v2.8h, v6.8h\n"
        "smlal2 v22.4s, v2.8h, v7.8h\n"
        "ld1 {v2.8b}, [x0]\n"
        "smlal2 v11.4s, v3.8h, v4.8h\n"
        "add x0, x0, x2\n"
        "smlal2 v15.4s, v3.8h, v5.8h\n"
        "smlal2 v19.4s, v3.8h, v6.8h\n"
        "smlal2 v23.4s, v3.8h, v7.8h\n"        
        "ld1 {v3.8b}, [x0]\n"

        "bne " FARM_LABEL_LOOP "b\n"

        FARM_LABEL_AFTER_LOOP
        ":\n"

        // Expand lhs/rhs to 16 bit and add offset
        "uxtl v4.8h, v26.8b\n"
        "uxtl v5.8h, v27.8b\n"
        "uxtl v6.8h, v28.8b\n"
        "uxtl v7.8h, v29.8b\n"
        "uxtl v0.8h, v0.8b\n"
        "uxtl v1.8h, v1.8b\n"
        "uxtl v2.8h, v2.8b\n"
        "uxtl v3.8h, v3.8b\n"
        "add v4.8h, v4.8h, v25.8h\n"
        "add v5.8h, v5.8h, v25.8h\n"
        "add v6.8h, v6.8h, v25.8h\n"
        "add v7.8h, v7.8h, v25.8h\n"     
        "add v0.8h, v0.8h, v24.8h\n"
        "add v1.8h, v1.8h, v24.8h\n"
        "add v2.8h, v2.8h, v24.8h\n"
        "add v3.8h, v3.8h, v24.8h\n"   

        "smlal v8.4s, v0.4h, v4.4h\n"
        "smlal v12.4s, v0.4h, v5.4h\n"
        "smlal v16.4s, v0.4h, v6.4h\n"
        "smlal v20.4s, v0.4h, v7.4h\n"
        "smlal v9.4s, v1.4h, v4.4h\n"
        "smlal v13.4s, v1.4h, v5.4h\n"
        "smlal v17.4s, v1.4h, v6.4h\n"
        "smlal v21.4s, v1.4h, v7.4h\n"
        "smlal v10.4s, v2.4h, v4.4h\n"
        "smlal v14.4s, v2.4h, v5.4h\n"
        "smlal v18.4s, v2.4h, v6.4h\n"
        "smlal v22.4s, v2.4h, v7.4h\n"
        "smlal v11.4s, v3.4h, v4.4h\n"
        "smlal v15.4s, v3.4h, v5.4h\n"
        "smlal v19.4s, v3.4h, v6.4h\n"
        "smlal v23.4s, v3.4h, v7.4h\n"

        "smlal2 v8.4s, v0.8h, v4.8h\n"
        "smlal2 v12.4s, v0.8h, v5.8h\n"
        "smlal2 v16.4s, v0.8h, v6.8h\n"
        "smlal2 v20.4s, v0.8h, v7.8h\n"
        "smlal2 v9.4s, v1.8h, v4.8h\n"
        "smlal2 v13.4s, v1.8h, v5.8h\n"
        "smlal2 v17.4s, v1.8h, v6.8h\n"
        "smlal2 v21.4s, v1.8h, v7.8h\n"
        "smlal2 v10.4s, v2.8h, v4.8h\n"
        "smlal2 v14.4s, v2.8h, v5.8h\n"
        "smlal2 v18.4s, v2.8h, v6.8h\n"
        "smlal2 v22.4s, v2.8h, v7.8h\n"
        "smlal2 v11.4s, v3.8h, v4.8h\n"
        "smlal2 v15.4s, v3.8h, v5.8h\n"
        "smlal2 v19.4s, v3.8h, v6.8h\n"
        "smlal2 v23.4s, v3.8h, v7.8h\n"     

        // Add across vector
        "addv s8, v8.4s\n"
        "dup v0.4s, %w[res_offset]\n"
        "addv s9, v9.4s\n"
        "dup v1.4s, %w[res_mul]\n"
        "addv s10, v10.4s\n"
        "dup v2.4s, %w[res_shift]\n"
        "addv s11, v11.4s\n"
        "mov v8.s[1], v9.s[0]\n"    
        "mov v8.s[2], v10.s[0]\n"
        "mov v8.s[3], v11.s[0]\n"
        "add v8.4s, v8.4s, v0.4s\n"
        "mul v8.4s, v8.4s, v1.4s\n"
        "and v3.16b, v8.16b, v2.16b\n"
        "sshr v3.4s, v3.4s, #31\n"
        "sqadd v8.4s, v8.4s, v3.4s\n"
        "srshl v8.4s, v8.4s, v2.4s\n"
        "sqxtn v3.4h, v8.4s\n"
        "mov v8.d[0], v3.d[0]\n"
        "mov v8.d[1], v3.d[0]\n"
        "sqxtun v8.8b, v8.8h\n"
        "st1 {v8.s}[0], [%[dst_ptr]]\n"
        "add %[dst_ptr], %[dst_ptr], %[rows]\n"

        "addv s12, v12.4s\n"
        "addv s13, v13.4s\n"
        "addv s14, v14.4s\n"
        "addv s15, v15.4s\n"
        "mov v12.s[1], v13.s[0]\n"
        "mov v12.s[2], v14.s[0]\n"
        "mov v12.s[3], v15.s[0]\n"
        "add v12.4s, v12.4s, v0.4s\n"
        "mul v12.4s, v12.4s, v1.4s\n"
        "and v3.16b, v12.16b, v2.16b\n"
        "sshr v3.4s, v3.4s, #31\n"
        "sqadd v12.4s, v12.4s, v3.4s\n"
        "srshl v12.4s, v12.4s, v2.4s\n"
        "sqxtn v3.4h, v12.4s\n"
        "mov v12.d[0], v3.d[0]\n"
        "mov v12.d[1], v3.d[0]\n"
        "sqxtun v12.8b, v12.8h\n"
        "st1 {v12.s}[0], [%[dst_ptr]]\n"
        "add %[dst_ptr], %[dst_ptr], %[rows]\n"

        "addv s16, v16.4s\n"
        "addv s17, v17.4s\n"
        "addv s18, v18.4s\n"
        "addv s19, v19.4s\n"
        "mov v16.s[1], v17.s[0]\n"
        "mov v16.s[2], v18.s[0]\n"
        "mov v16.s[3], v19.s[0]\n"
        "add v16.4s, v16.4s, v0.4s\n"
        "mul v16.4s, v16.4s, v1.4s\n"
        "and v3.16b, v16.16b, v2.16b\n"
        "sshr v3.4s, v3.4s, #31\n"
        "sqadd v16.4s, v16.4s, v3.4s\n"
        "srshl v16.4s, v16.4s, v2.4s\n"
        "sqxtn v3.4h, v16.4s\n"
        "mov v16.d[0], v3.d[0]\n"
        "mov v16.d[1], v3.d[0]\n"
        "sqxtun v16.8b, v16.8h\n"
        "st1 {v16.s}[0], [%[dst_ptr]]\n"
        "add %[dst_ptr], %[dst_ptr], %[rows]\n"

        "addv s20, v20.4s\n"
        "addv s21, v21.4s\n"
        "addv s22, v22.4s\n"
        "addv s23, v23.4s\n"
        "mov v20.s[1], v21.s[0]\n"
        "mov v20.s[2], v22.s[0]\n"
        "mov v20.s[3], v23.s[0]\n"
        "add v20.4s, v20.4s, v0.4s\n"
        "mul v20.4s, v20.4s, v1.4s\n"
        "and v3.16b, v20.16b, v2.16b\n"
        "sshr v3.4s, v3.4s, #31\n"
        "sqadd v20.4s, v20.4s, v3.4s\n"
        "srshl v20.4s, v20.4s, v2.4s\n"
        "sqxtn v3.4h, v20.4s\n"
        "mov v20.d[0], v3.d[0]\n"
        "mov v20.d[1], v3.d[0]\n"
        "sqxtun v20.8b, v20.8h\n"
        "st1 {v20.s}[0], [%[dst_ptr]]\n"       
#undef FARM_LABEL_LOOP
#undef FARM_LABEL_AFTER_LOOP
        : // outputs
        [lhs_ptr] "+r"(lhs_ptr), [rhs_ptr] "+r"(rhs_ptr),
        [dst_ptr] "+r"(dst_ptr),
        [run_depth] "+r"(run_depth)
        : // inputs
        [rows] "r"(rows), // this is used when storing the result
        [lhs_offset] "r"(lhs_offset),
        [rhs_offset] "r"(rhs_offset),
        [res_offset] "r"(res_offset),
        [res_mul] "r"(res_mul),
        [res_shift] "r"(res_shift)    
        : // clobbers
        "cc", "memory", "x0", "x1", "x2", "v0", "v1", "v2", "v3", "v4", "v5", "v6",
        "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16",
        "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26",
        "v27", "v28", "v29");
}  


void gevv_kernel_run(const std::uint8_t* lhs_ptr, const std::uint8_t* rhs_ptr,
                     std::uint8_t* dst_ptr, std::size_t run_depth,
                     std::int16_t* lhs_offset, std::int16_t* rhs_offset,
                     std::int32_t res_offset, std::int32_t res_mul,
                     std::int32_t res_shift) {
#define FARM_LABEL_LOOP "1"
#define FARM_LABEL_AFTER_LOOP "2"
    asm volatile(
        "ld1 {v0.8b}, [%[lhs_ptr]], #8\n"
        "subs %[run_depth], %[run_depth], #8\n"
        "ld1 {v1.8b}, [%[rhs_ptr]], #8\n"
        "ld1r {v3.8h}, [%[lhs_offset]]\n"
        "ld1r {v4.8h}, [%[rhs_offset]]\n"
        "dup v2.4s, wzr\n"

        "beq " FARM_LABEL_AFTER_LOOP "f\n"

        FARM_LABEL_LOOP
        ":\n"

        // Inner Loop
        "uxtl v0.8h, v0.8b\n"
        "uxtl v1.8h, v1.8b\n"
        "add v0.8h, v0.8h, v3.8h\n"
        "add v1.8h, v1.8h, v4.8h\n"

        "smlal v2.4s, v0.4h, v1.4h\n"
        "smlal2 v2.4s, v0.8h, v1.8h\n"
        "ld1 {v0.8b}, [%[lhs_ptr]], #8\n"
        "ld1 {v1.8b}, [%[rhs_ptr]], #8\n"

        "subs %[run_depth], %[run_depth], #8\n"

        "bne " FARM_LABEL_LOOP "b\n"

        FARM_LABEL_AFTER_LOOP
        ":\n"

        // Last Iter
        "uxtl v0.8h, v0.8b\n"
        "uxtl v1.8h, v1.8b\n"
        "add v0.8h, v0.8h, v3.8h\n"
        "add v1.8h, v1.8h, v4.8h\n"

        "smlal v2.4s, v0.4h, v1.4h\n"
        "dup v3.4s, %w[res_offset]\n"
        "dup v4.4s, %w[res_mul]\n"
        "dup v5.4s, %w[res_shift]\n"
        "smlal2 v2.4s, v0.8h, v1.8h\n"

        "addv s2, v2.4s\n"
        "add v2.4s, v2.4s, v3.4s\n"
        "mul v2.4s, v2.4s, v4.4s\n"
        "and v0.16b, v2.16b, v5.16b\n"
        "sshr v0.4s, v0.4s, #31\n"
        "sqadd v2.4s, v2.4s, v0.4s\n"
        "srshl v2.4s, v2.4s, v5.4s\n"
        "sqxtn v0.4h, v2.4s\n"
        "mov v2.d[0], v0.d[0]\n"
        "mov v2.d[1], v0.d[0]\n"
        "sqxtun v2.8b, v2.8h\n"

        // Store accumulators
        "st1 {v2.b}[0], [%[dst_ptr]]\n"
#undef FARM_LABEL_LOOP
#undef FARM_LABEL_AFTER_LOOP
        : // outputs
        [lhs_ptr] "+r"(lhs_ptr), [rhs_ptr] "+r"(rhs_ptr),
        [dst_ptr] "+r"(dst_ptr),
        [run_depth] "+r"(run_depth)
        : // inputs
        [lhs_offset] "r"(lhs_offset),
        [rhs_offset] "r"(rhs_offset),
        [res_offset] "r"(res_offset),
        [res_mul] "r"(res_mul),
        [res_shift] "r"(res_shift)
        : // clobbers
        "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5");
}

void gemm_1_run(const MatrixMap<MapOrder::RowMajor>& lhs,
                const MatrixMap<MapOrder::ColMajor>& rhs,
                MatrixMap<MapOrder::ColMajor>* result, int col_idx,
                int16_t lhs_offset_16, int16_t rhs_offset_16, int32_t result_offset,
                int32_t result_mult_int, int32_t result_shift) {
    int rows = lhs.rows();
    int depth = lhs.cols();
    int step = 8;
    int blk_depth = 16;

    assert(depth % blk_depth == 0);

    int num_bottom_rows = rows % step;
    int clean_rows = rows - num_bottom_rows;

    for (int r = 0; r < clean_rows; r += step) {
        gemm_1_kernel_run(lhs.data() + r * depth, rhs.data() + col_idx * depth, 
                          result->data() + col_idx * rows + r, depth, 
                          &lhs_offset_16, &rhs_offset_16, result_offset,
                          result_mult_int, -result_shift);
    }    

    for (int r = 0; r < num_bottom_rows; ++r) {
        gevv_kernel_run(lhs.data() + (clean_rows + r) * depth, 
                        rhs.data() + col_idx * depth, 
                        result->data() + (clean_rows + r) + col_idx * rows,
                        depth, &lhs_offset_16, &rhs_offset_16, result_offset,
                        result_mult_int, -result_shift);
    }    

}

void gemm_2_run(const MatrixMap<MapOrder::RowMajor>& lhs,
                const MatrixMap<MapOrder::ColMajor>& rhs,
                MatrixMap<MapOrder::ColMajor>* result, int col_idx,
                int16_t lhs_offset_16, int16_t rhs_offset_16, int32_t result_offset,
                int32_t result_mult_int, int32_t result_shift) {
    int rows = lhs.rows();
    int depth = lhs.cols();
    int step = 8;
    int blk_depth = 8;

    assert(depth % blk_depth == 0);

    int num_bottom_rows = rows % step;
    int clean_rows = rows - num_bottom_rows;

    for (int r = 0; r < clean_rows; r += step) {
        gemm_2_kernel_run(lhs.data() + r * depth, rhs.data() + col_idx * depth, 
                          result->data() + col_idx * rows + r,
                          depth, rows, &lhs_offset_16, &rhs_offset_16, 
                          result_offset, result_mult_int, -result_shift);
    }     

    for (int r = 0; r < num_bottom_rows; ++r) {
        for (int c = 0; c < 2; ++c) {
            gevv_kernel_run(lhs.data() + (clean_rows + r) * depth, 
                            rhs.data() + (col_idx + c) * depth, 
                            result->data() + (clean_rows + r) + (col_idx + c) * rows,
                            depth, &lhs_offset_16, &rhs_offset_16, result_offset,
                            result_mult_int, -result_shift);
        }
    }    
}

void gemm_3_run(const MatrixMap<MapOrder::RowMajor>& lhs,
                const MatrixMap<MapOrder::ColMajor>& rhs,
                MatrixMap<MapOrder::ColMajor>* result, int col_idx,
                int16_t lhs_offset_16, int16_t rhs_offset_16, int32_t result_offset,
                int32_t result_mult_int, int32_t result_shift) {
    int rows = lhs.rows();
    int depth = lhs.cols();
    int step = 4;
    int blk_depth = 16;

    assert(depth % blk_depth == 0);

    int num_bottom_rows = rows % step;
    int clean_rows = rows - num_bottom_rows;

    for (int r = 0; r < clean_rows; r += step) {
        gemm_3_kernel_run(lhs.data() + r * depth, rhs.data() + col_idx * depth, 
                          result->data() + col_idx * rows + r, 
                          depth, rows, &lhs_offset_16, &rhs_offset_16, 
                          result_offset, result_mult_int, -result_shift);
    }         

    for (int r = 0; r < num_bottom_rows; ++r) {
        for (int c = 0; c < 3; ++c) {
            gevv_kernel_run(lhs.data() + (clean_rows + r) * depth, 
                            rhs.data() + (col_idx + c) * depth, 
                            result->data() + (clean_rows + r) + (col_idx + c) * rows,
                            depth, &lhs_offset_16, &rhs_offset_16, result_offset,
                            result_mult_int, -result_shift);
        }
    }
}

void gemm_4_run(const MatrixMap<MapOrder::RowMajor>& lhs,
                const MatrixMap<MapOrder::ColMajor>& rhs,
                MatrixMap<MapOrder::ColMajor>* result, int col_idx,
                int16_t lhs_offset_16, int16_t rhs_offset_16, int32_t result_offset,
                int32_t result_mult_int, int32_t result_shift) {
    int rows = lhs.rows();
    int depth = lhs.cols();
    int step = 4;
    int blk_depth = 8;

    assert(depth % blk_depth == 0);

    int num_bottom_rows = rows % step;
    int clean_rows = rows - num_bottom_rows;

    for (int r = 0; r < clean_rows; r += step) {
        gemm_4_kernel_run(lhs.data() + r * depth, rhs.data() + col_idx * depth, 
                          result->data() + col_idx * rows + r, 
                          depth, rows, &lhs_offset_16, &rhs_offset_16, 
                          result_offset, result_mult_int, -result_shift);
    }         

    for (int r = 0; r < num_bottom_rows; ++r) {
        for (int c = 0; c < 4; ++c) {
            gevv_kernel_run(lhs.data() + (clean_rows + r) * depth, 
                            rhs.data() + (col_idx + c) * depth, 
                            result->data() + (clean_rows + r) + (col_idx + c) * rows,
                            depth, &lhs_offset_16, &rhs_offset_16, result_offset,
                            result_mult_int, -result_shift);
        }
    }
}

template <MapOrder LhsOrder, MapOrder RhsOrder, MapOrder ResultOrder>
void SingleThreadGemm(const MatrixMap<LhsOrder>& lhs,
                      const MatrixMap<RhsOrder>& rhs,
                      MatrixMap<ResultOrder>* result,
                      int32_t lhs_offset, int32_t rhs_offset, int32_t result_offset,
                      int32_t result_mult_int, int32_t result_shift) {
    std::cout << "General version of SingleThreadGemm not supported!" << std::endl;
}

template <>
void SingleThreadGemm(const MatrixMap<MapOrder::RowMajor>& lhs,
                      const MatrixMap<MapOrder::ColMajor>& rhs,
                      MatrixMap<MapOrder::ColMajor>* result, 
                      int32_t lhs_offset, int32_t rhs_offset, int32_t result_offset,
                      int32_t result_mult_int, int32_t result_shift) {
    static constexpr int32_t min_int16 = int32_t(std::numeric_limits<int16_t>::min());
    static constexpr int32_t max_int16 = int32_t(std::numeric_limits<int16_t>::max());

    assert(lhs_offset >= min_int16 && lhs_offset <= max_int16);
    assert(rhs_offset >= min_int16 && rhs_offset <= max_int16);

    assert(lhs.cols() == rhs.rows() && 
           lhs.rows() == result->rows() && 
           rhs.cols() == result->cols());

    int16_t lhs_offset_16 = static_cast<int16_t>(lhs_offset);
    int16_t rhs_offset_16 = static_cast<int16_t>(rhs_offset);

    // The number of iterations to use call batch size 4 kernel.
    int num_of_iter = rhs.cols() / 4;
    // The remaining cols are assgined to a batch size 1 ~ 3 kernel.
    int cols_left = rhs.cols() % 4;

    for (int i = 0; i < num_of_iter; ++i) {
        gemm_4_run(lhs, rhs, result, i * 4, lhs_offset_16, rhs_offset_16,
                   result_offset, result_mult_int, result_shift);
    }

    switch (cols_left) {
        case 0:
            break;
        case 1:
            gemm_1_run(lhs, rhs, result, num_of_iter * 4, lhs_offset_16, rhs_offset_16,
                       result_offset, result_mult_int, result_shift);
            break;
        case 2:
            gemm_2_run(lhs, rhs, result, num_of_iter * 4, lhs_offset_16, rhs_offset_16,
                       result_offset, result_mult_int, result_shift);
            break;
        case 3:
            gemm_3_run(lhs, rhs, result, num_of_iter * 4, lhs_offset_16, rhs_offset_16,
                       result_offset, result_mult_int, result_shift);
            break;
        default:
            assert(false);
    }
}

}  // namespace farm
