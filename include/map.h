// map.h: a minimalist view-existing-buffer-as-a-matrix class,
// which is how farm interfaces with external matrix data.

#pragma once

#include <cassert>

namespace farm {

// The two storage orders allowed to map buffers as matrices: ColMajor
// means column-major, RowMajor means row-major.
enum class MapOrder { ColMajor, RowMajor };

// A MatrixMap is a view of an existing buffer as a matrix. It does not own
// the buffer.
template <MapOrder tOrder>
class MatrixMap {
public:
    static const MapOrder kOrder = tOrder;

protected:
    std::uint8_t* data_;  // not owned.
    int rows_, cols_, stride_;

public:
    MatrixMap() : data_(nullptr), rows_(0), cols_(0), stride_(0) {}
    MatrixMap(std::uint8_t* data, int rows, int cols)
        : data_(data),
          rows_(rows),
          cols_(cols),
          stride_(kOrder == MapOrder::ColMajor ? rows : cols) {}
    MatrixMap(std::uint8_t* data, int rows, int cols, int stride)
        : data_(data), rows_(rows), cols_(cols), stride_(stride) {}
    MatrixMap(const MatrixMap& other)
        : data_(other.data_),
          rows_(other.rows_),
          cols_(other.cols_),
          stride_(other.stride_) {}

    int rows() const { return rows_; }
    int cols() const { return cols_; }
    int stride() const { return stride_; }
    int rows_stride() const { return kOrder == MapOrder::ColMajor ? 1 : stride_; }
    int cols_stride() const { return kOrder == MapOrder::RowMajor ? 1 : stride_; }
    std::uint8_t* data() const { return data_; }
    std::uint8_t* data(int row, int col) const {
        return data_ + row * rows_stride() + col * cols_stride();
    }
    std::uint8_t& operator()(int row, int col) const { return *data(row, col); }

    MatrixMap block(int start_row, int start_col, int block_rows,
                    int block_cols) const {
        assert(start_row >= 0);
        assert(start_row + block_rows <= rows_);
        assert(start_col >= 0);
        assert(start_col + block_cols <= cols_);

        return MatrixMap(data(start_row, start_col), block_rows, block_cols,
                         stride_);
    }
};	
    
}  // namespace farm
