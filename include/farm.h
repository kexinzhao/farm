// farm.h: the main public interface header of gemmlowp

#pragma once

#include <iostream>

#include "map.h"

#ifdef __aarch64__
#define FARM_ARM_64
#endif

#if (defined __ARM_NEON) || (defined __ARM_NEON__)
#define FARM_NEON
#endif

#if defined(FARM_NEON) && defined(FARM_ARM_64)
#define FARM_NEON_64
#endif

#ifdef FARM_NEON_64
#include "single_thread_gemm.h"
#endif

namespace farm {

template <MapOrder LhsOrder, MapOrder RhsOrder, MapOrder ResultOrder>
void Gemm(const MatrixMap<LhsOrder>& lhs,
          const MatrixMap<RhsOrder>& rhs,
          MatrixMap<ResultOrder>* res, 
          int lhs_offset, int rhs_offset, int result_offset,
          int result_mult_int, int result_shift) {
    if (LhsOrder == MapOrder::RowMajor &&
        RhsOrder == MapOrder::ColMajor &&
        ResultOrder == MapOrder::ColMajor) {
#ifdef FARM_NEON_64
        SingleThreadGemm(lhs, rhs, res, lhs_offset, rhs_offset,
                         result_offset, result_mult_int, result_shift);
#else
#warning "64-bit ARM is not enabled, bypassing SingleThreadGemm!"
#endif
        return;
    }  

    std::cout << "SingleThreadGemm is not used!" << std::endl;
}

}  // namespace farm
