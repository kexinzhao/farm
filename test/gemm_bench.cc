#include <chrono>
#include <iomanip>
#include <iostream>
#include <vector>

#include "../include/farm.h"
#include "test.h"

template <bool a_t, bool b_t>
double time_gemm(int m, int n, int k) {
    typedef farm::MapOrder Order;
    
    static const Order LhsOrder = a_t ? Order::RowMajor : Order::ColMajor;
    static const Order RhsOrder = b_t ? Order::RowMajor : Order::ColMajor;
    
    farm::Matrix<LhsOrder> lhs(m, k);
    farm::Matrix<RhsOrder> rhs(k, n);
    farm::Matrix<Order::ColMajor> result(m, n);
    
    farm::MakeRandom<typename farm::OperandRange<0, 255>>(&lhs);
    farm::MakeRandom<typename farm::OperandRange<0, 255>>(&rhs);
    
    int lhs_offset = -128;
    int rhs_offset = 0;
    int res_offset = 32768;
    int res_mul = 1;
    int res_shift = 8;
    
    // warm up
    farm::Gemm(
        lhs.map(),
        rhs.map(),
        &(result.map()),
        lhs_offset,
        rhs_offset,
        res_offset,
        res_mul,
        res_shift);
    
    int numRepeats = std::min(100., std::max(std::ceil(1e12 / (m * k * n)), 10.));
    auto start = std::chrono::steady_clock::now();
    
    for (int i = 0; i < numRepeats; ++i) {
        farm::Gemm(
            lhs.map(),
            rhs.map(),
            &(result.map()),
            lhs_offset,
            rhs_offset,
            res_offset,
            res_mul,
            res_shift);
    }
    
    auto end = std::chrono::steady_clock::now();
    
    return std::chrono::duration<double, std::milli>(end - start).count() / numRepeats;
}


double time_gemm_helper(int m, int n, int k, bool a_t, bool b_t) {
#define HANDLE_MATRIX_ORDER(ta, tb)            \
    if (a_t == ta && b_t == tb) {              \
        return time_gemm<ta, tb>(m, n, k);     \
    }
    
    HANDLE_MATRIX_ORDER(false, false)
    HANDLE_MATRIX_ORDER(false, true)
    HANDLE_MATRIX_ORDER(true, false)
    HANDLE_MATRIX_ORDER(true, true)
    
#undef HANDLE_MATRIX_ORDER
    return 0;
}

int main(int argc, char** argv) {
    // false = col-major, true = row-major
    // m = rows, n = cols, k = depth
    std::vector<std::tuple<int, int, int, bool, bool>> problems = {
        std::make_tuple(6144, 1, 320, true, false),
        std::make_tuple(6144, 2, 320, true, false),
        std::make_tuple(6144, 3, 320, true, false),
        std::make_tuple(6144, 4, 320, true, false),
        std::make_tuple(6144, 5, 320, true, false),
        std::make_tuple(6144, 6, 320, true, false),
        std::make_tuple(6144, 7, 320, true, false),
        std::make_tuple(6144, 8, 320, true, false),
        std::make_tuple(6144, 9, 320, true, false),
        std::make_tuple(6144, 10, 320, true, false)
    };
    
    std::cout << std::setw(30) << "Times" << std::endl;
    std::cout << std::setfill('-') << std::setw(88) << "-" << std::endl;
    std::cout << std::setfill(' ');
    std::cout << "    m       n      k      a_t    b_t    time (msec)     GOP/s        MB/s " << std::endl;
    
    for (const auto &problem : problems) {
        int m, n, k;
        bool a_t, b_t;
        std::tie(m, n, k, a_t, b_t) = problem;
        
        double time = time_gemm_helper(m, n, k, a_t, b_t);
        double gops = 1e-6 * 2 * m * n * k / time;
        double mbs = 1e-3 * (m * k + k * n + m * n) / time; 
        
        std::cout << std::setw(7) << m;
        std::cout << std::setw(7) << n;
        std::cout << std::setw(7) << k;
        std::cout << std::setw(7) << (a_t ? "t" : "n");
        std::cout << std::setw(7) << (b_t ? "t" : "n");
        std::cout << std::setw(13) << std::setprecision(6) << time;
        std::cout << std::setw(13) << std::setprecision(6) << gops; 
        std::cout << std::setw(13) << std::setprecision(6) << mbs;
        std::cout << std::endl;
    }
    
    return 0;
}
