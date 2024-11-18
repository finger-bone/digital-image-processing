#ifndef IMAGE_PROCESSING_LINALG_HXX
#define IMAGE_PROCESSING_LINALG_HXX

namespace Linalg {

#include <stdexcept>
#include <thread>
#include <vector>

template <typename T> struct Matrix {
  std::vector<std::vector<T>> data;
  int rows;
  int cols;

  Matrix(int r, int c) : rows(r), cols(c) {
    data.resize(rows, std::vector<T>(cols));
  }

  Matrix(std::initializer_list<std::initializer_list<T>> data) {
    this->rows = data.size();
    this->cols = data.begin()->size();
    this->data = std::vector<std::vector<T>>(rows, std::vector<T>(cols));
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        this->data[i][j] = (*data.begin())[j];
      }
      data.begin();
    }
  }

  const T &operator[](std::pair<int, int> index) const {
    if (index.first >= rows || index.second >= cols || index.first < 0 ||
        index.second < 0) {
      throw std::out_of_range("Matrix index out of bounds");
    }
    return data[index.first][index.second];
  }

  Matrix<T> operator+(const Matrix<T> &other) {
    if (rows != other.rows || cols != other.cols) {
      throw std::invalid_argument("Matrix dimensions must match for addition");
    }
    Matrix<T> result(rows, cols);
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        result.data[i][j] = data[i][j] + other.data[i][j];
      }
    }
    return result;
  }

  Matrix<T> operator*(const Matrix<T> &other) {
    if (cols != other.rows) {
      throw std::invalid_argument(
          "Matrix dimensions incompatible for multiplication");
    }
    Matrix<T> result(rows, other.cols);
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < other.cols; j++) {
        result.data[i][j] = T();
        for (int k = 0; k < cols; k++) {
          result.data[i][j] += data[i][k] * other.data[k][j];
        }
      }
    }
    return result;
  }

  Matrix<T> transpose() {
    Matrix<T> result(cols, rows);
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        result.data[j][i] = data[i][j];
      }
    }
    return result;
  }

  Matrix<T> map(T (*func)(T)) {
    Matrix<T> result(rows, cols);
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        result.data[i][j] = func(data[i][j]);
      }
    }
    return result;
  }
};

} // namespace Linalg

#endif
