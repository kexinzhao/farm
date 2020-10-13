#include <cmath>
#include <iostream>
#include <tuple>
#include <vector>

#include "../include/farm.h"
#include "test.h"

template <bool a_t, bool b_t>
bool test_correctness(int m, int n, int k) {
    typedef farm::MapOrder Order;

    static const Order LhsOrder = a_t ? Order::RowMajor : Order::ColMajor;
    static const Order RhsOrder = b_t ? Order::RowMajor : Order::ColMajor;

    farm::Matrix<LhsOrder> lhs(m, k);
    farm::Matrix<RhsOrder> rhs(k, n);
    // result_gemm is the gemm output of the farm library.
    farm::Matrix<Order::ColMajor> result_gemm(m, n);
    // result_uint8 is the correct gemm output calculated using for loop.
    farm::Matrix<Order::ColMajor> result_uint8(m, n);

    farm::MakeRandom<typename farm::OperandRange<0, 255>>(&lhs);
    farm::MakeRandom<typename farm::OperandRange<0, 255>>(&rhs);

    farm::MakeConstant(&result_gemm, 0);
    farm::MakeConstant(&result_uint8, 0);
    
    int lhs_offset = -128;
    int rhs_offset = 0;
    int res_offset = 32768;
    int res_mul = 1;
    int res_shift = 8;

    farm::Gemm(
        lhs.map(),
        rhs.map(),
        &(result_gemm.map()),
        lhs_offset,
        rhs_offset,
        res_offset,
        res_mul,
        res_shift);

    // Calculate result_uint8 as the correct gemm output
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            int temp = 0;
            for (int t = 0; t < k; t++) {
                temp += (static_cast<int32_t>(lhs(i, t)) + lhs_offset) * 
                        (static_cast<int32_t>(rhs(t, j)) + rhs_offset);
            }
            
            double res = static_cast<double>((temp + res_offset) * res_mul) / 
                         static_cast<double>(pow(2.0, res_shift));

            if (res < 0.0) {
                result_uint8(i, j) = 0;
            } else if (res > 255.0) {
                result_uint8(i, j) = 255;
            } else {
                result_uint8(i, j) = static_cast<std::uint8_t>(std::round(res));
            }
            
            if (result_gemm(i, j) != result_uint8(i, j)) {
                std::cout << "result (" << m << ", "
                          << n << ", " << k << ") " 
                          << "row " << i << " col " << j << ":" << std::endl;
                std::cout << "ARM kernel result is " 
                          << static_cast<int>(result_gemm(i, j)) 
                          << std::endl;
                std::cout << "The correct result is " 
                          << static_cast<int>(result_uint8(i, j))
                          << std::endl << std::endl;
                return false;
            }
        }
    }

    std::cout << "Implementation for (" 
              << m << ", "
              << n << ", "
              << k << ")"
              << " is correct!"
              << std::endl;

    return true;
}

bool test_correctness_helper(int m, int n, int k, bool a_t, bool b_t) {
#define HANDLE_MATRIX_ORDER(ta, tb)                  \
    if (a_t == ta && b_t == tb) {                    \
        return test_correctness<ta, tb>(m, n, k);    \
    }

    HANDLE_MATRIX_ORDER(false, false)
    HANDLE_MATRIX_ORDER(false, true)
    HANDLE_MATRIX_ORDER(true, false)
    HANDLE_MATRIX_ORDER(true, true)

#undef HANDLE_MATRIX_ORDER
    return false;
}  

int main(int argc, char** argv) {
    // false = col-major, true = row-major
    // m = rows, n = cols, k = depth
    std::vector<std::tuple<int, int, int, bool, bool>> problems = {
        std::make_tuple(320, 1, 320, true, false),
        std::make_tuple(640, 2, 640, true, false),
        std::make_tuple(659, 3, 640, true, false),
        std::make_tuple(1024, 4, 1024, true, false),
        std::make_tuple(1039, 5, 1024, true, false),
        std::make_tuple(1280, 6, 1280, true, false),
        std::make_tuple(1919, 7, 1600, true, false),
        std::make_tuple(1865, 8, 1920, true, false),
        std::make_tuple(2048, 9, 2560, true, false),
        std::make_tuple(2057, 10, 2560, true, false),
        std::make_tuple(2136, 11, 2560, true, false),
        std::make_tuple(2259, 12, 2560, true, false),
        std::make_tuple(2478, 13, 2560, true, false),
        std::make_tuple(2566, 14, 2560, true, false),
        std::make_tuple(2600, 15, 2560, true, false),
        std::make_tuple(2701, 16, 2560, true, false),
        std::make_tuple(2899, 17, 2560, true, false),
        std::make_tuple(3027, 18, 4096, true, false),
        std::make_tuple(4096, 19, 4096, true, false),
        std::make_tuple(4111, 20, 4096, true, false)
    };

    for (const auto &problem : problems) {
        int m, n, k;
        bool a_t, b_t;
        std::tie(m, n, k, a_t, b_t) = problem;

        if (!test_correctness_helper(m, n, k, a_t, b_t)) {
            return -1; 
        }
    }

    return 0;
}
