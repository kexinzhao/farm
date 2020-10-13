// test.h: shared testing helpers.
#pragma once

#include <cstring>
#include <iostream>
#include <random>
#include <vector>

#include "../include/farm.h"

namespace farm {
    // farm itself doesn't have a Matrix class, only a MatrixMap class,
    // since it only maps existing data. In tests though, we need to
    // create our own matrices.

    // The range of allowed values for an operand.
    template <int tMinValue, int tMaxValue>
    struct OperandRange {
        static const int kMinValue = tMinValue;
        static const int kMaxValue = tMaxValue;
        static_assert(0 <= kMinValue, "");
        static_assert(kMinValue < kMaxValue, "");
        static_assert(kMaxValue <= 255, "");
    };

    template <MapOrder tOrder>
    class Matrix : public MatrixMap<tOrder> {
    public:
        typedef MatrixMap<tOrder> Map;
        static const MapOrder Order = tOrder;
        using Map::kOrder;
        using Map::rows_;
        using Map::cols_;
        using Map::stride_;
        using Map::data_;
        
    public:
        Matrix() : Map(nullptr, 0, 0, 0) {}
        
        Matrix(int rows, int cols) : Map(nullptr, 0, 0, 0) { Resize(rows, cols); }
        
        Matrix(const Matrix& other) : Map(nullptr, 0, 0, 0) { *this = other; }
        
        Matrix& operator=(const Matrix& other) {
            Resize(other.rows_, other.cols_);
            std::memcpy(data_, other.data_, size() * sizeof(std::uint8_t));
            return *this;
        }
        
        friend bool operator==(const Matrix& a, const Matrix& b) {
            return a.rows_ == b.rows_ && a.cols_ == b.cols_ &&
            !std::memcmp(a.data_, b.data_, a.size());
        }
        
        void Resize(int rows, int cols) {
            rows_ = rows;
            cols_ = cols;
            stride_ = kOrder == MapOrder::ColMajor ? rows : cols;
            storage.resize(size());
            data_ = storage.data();
        }
        
        int size() const { return rows_ * cols_; }
        
        Map& map() { return *static_cast<Map*>(this); }
                
    protected:
        std::vector<std::uint8_t> storage;
    };
    
    std::mt19937& RandomEngine() {
        static std::mt19937 engine;
        return engine;
    }
    
    int Random() {
        std::uniform_int_distribution<int> dist(0, std::numeric_limits<int>::max());
        return dist(RandomEngine());
    }
    
    template <typename OperandRange, typename MatrixType>
    void MakeRandom(MatrixType* m) {
        std::uniform_int_distribution<std::uint8_t> dist(OperandRange::kMinValue,
                                                    OperandRange::kMaxValue);
        for (int c = 0; c < m->cols(); c++) {
            for (int r = 0; r < m->rows(); r++) {
                (*m)(r, c) = dist(RandomEngine());
            }
        }
    }
    
    template <typename MatrixType>
    void MakeConstant(MatrixType* m, std::uint8_t val) {
        for (int c = 0; c < m->cols(); c++) {
            for (int r = 0; r < m->rows(); r++) {
                (*m)(r, c) = val;
            }
        }
    }
    
    template <typename MatrixType>
    void MakeZero(MatrixType* m) {
        MakeConstant(m, 0);
    }
    
}  // namespace farm
