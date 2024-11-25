#ifndef IMAGE_PROCESSING_LINALG_HXX
#define IMAGE_PROCESSING_LINALG_HXX

#include <cmath>
#include <numeric>
#include <stdexcept>
#include <thread>
#include <tuple>
#include <vector>

namespace Linalg {

template <typename T> struct Matrix {
  std::vector<std::vector<T>> data;
  int rows;
  int cols;

  std::vector<T> &operator[](int index) {
    if (index < 0 || index >= rows) {
      throw std::out_of_range("Row index out of bounds");
    }
    return data[index];
  }

  const std::vector<T> &operator[](int index) const {
    if (index < 0 || index >= rows) {
      throw std::out_of_range("Row index out of bounds");
    }
    return data[index];
  }

  Matrix(int r, int c) : rows(r), cols(c) {
    data.resize(rows, std::vector<T>(cols));
  }

  Matrix(std::initializer_list<std::initializer_list<T>> data) {
    this->rows = data.size();
    this->cols = data.begin()->size();
    this->data = std::vector<std::vector<T>>(rows, std::vector<T>(cols));
    auto row_it = data.begin();
    for (int i = 0; i < rows; i++, row_it++) {
      auto col_it = row_it->begin();
      for (int j = 0; j < cols; j++, col_it++) {
        this->data[i][j] = *col_it;
      }
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

  std::tuple<Matrix<T>, Matrix<T>, Matrix<T>> svd() {
    // Ensure the matrix is non-empty
    if (rows == 0 || cols == 0) {
      throw std::invalid_argument("Matrix is empty, cannot compute SVD.");
    }

    // Step 1: Initialize U, Σ, V
    Matrix<T> U = *this;         // Copy of A (will be modified)
    Matrix<T> Sigma(rows, cols); // Diagonal matrix for singular values
    Matrix<T> V(cols, cols);     // Initialize V as identity matrix
    for (int i = 0; i < cols; i++) {
      for (int j = 0; j < cols; j++) {
        V.data[i][j] = (i == j) ? 1 : 0;
      }
    }

    // Step 2: Jacobi method for bidiagonalization
    const T tolerance = 1e-10; // Convergence threshold
    bool converged = false;
    while (!converged) {
      converged = true;
      for (int i = 0; i < cols; i++) {
        for (int j = i + 1; j < cols; j++) {
          // Compute Jacobi rotation for columns i and j
          T a = 0, b = 0, c = 0;
          for (int k = 0; k < rows; k++) {
            a += U.data[k][i] * U.data[k][i];
            b += U.data[k][j] * U.data[k][j];
            c += U.data[k][i] * U.data[k][j];
          }
          if (std::fabs(c) > tolerance) {
            converged = false;
            T tau = (b - a) / (2 * c);
            T t = (tau > 0 ? 1 : -1) /
                  (std::fabs(tau) + std::sqrt(1 + tau * tau));
            T cos_theta = 1 / std::sqrt(1 + t * t);
            T sin_theta = t * cos_theta;

            // Apply rotation to U
            for (int k = 0; k < rows; k++) {
              T u_ki = U.data[k][i];
              T u_kj = U.data[k][j];
              U.data[k][i] = cos_theta * u_ki - sin_theta * u_kj;
              U.data[k][j] = sin_theta * u_ki + cos_theta * u_kj;
            }

            // Apply rotation to V
            for (int k = 0; k < cols; k++) {
              T v_ki = V.data[k][i];
              T v_kj = V.data[k][j];
              V.data[k][i] = cos_theta * v_ki - sin_theta * v_kj;
              V.data[k][j] = sin_theta * v_ki + cos_theta * v_kj;
            }
          }
        }
      }
    }

    // Step 3: Extract singular values into Σ
    for (int i = 0; i < std::min(rows, cols); i++) {
      T norm = 0;
      for (int k = 0; k < rows; k++) {
        norm += U.data[k][i] * U.data[k][i];
      }
      Sigma.data[i][i] = std::sqrt(norm);
      if (Sigma.data[i][i] > tolerance) {
        for (int k = 0; k < rows; k++) {
          U.data[k][i] /= Sigma.data[i][i];
        }
      }
    }

    // Step 4: Return U, Σ, V^T
    return {U, Sigma, V.transpose()};
  }

  Matrix<T> pinv() {
    // Step 1: Check if the matrix is empty
    if (rows == 0 || cols == 0) {
      throw std::invalid_argument(
          "Matrix is empty, cannot compute pseudoinverse.");
    }

    // Step 2: Compute SVD (Placeholder for SVD calculation)
    // You would need to implement or use an external library for SVD.
    // Here, assume `svd` returns U, Sigma, and V^T matrices.
    auto [U, Sigma, VT] = this->svd();

    // Step 3: Compute Sigma^+
    Matrix<T> SigmaP(cols, rows); // Transposed dimensions
    T tolerance = 1e-10;          // Threshold for singular values
    for (int i = 0; i < Sigma.rows; i++) {
      for (int j = 0; j < Sigma.cols; j++) {
        if (i == j && std::abs(Sigma[i][j]) > tolerance) {
          SigmaP[j][i] = 1 / Sigma[i][j];
        } else {
          SigmaP[j][i] = 0;
        }
      }
    }

    // Step 4: Compute A^+ = V * Sigma^+ * U^T
    Matrix<T> result = VT.transpose() * SigmaP * U.transpose();
    return result;
  }

  void pretty_print(std::ostream &os) {
    // Step 1: Determine the maximum width of any number in the matrix
    int max_width = 0;
    for (const auto &row : data) {
      for (const auto &value : row) {
        std::ostringstream ss;
        ss << value;
        max_width = std::max(max_width, static_cast<int>(ss.str().length()));
      }
    }

    // Step 2: Print the matrix with proper padding
    for (const auto &row : data) {
      os << "| ";
      for (const auto &value : row) {
        os << std::setw(max_width) << value << " ";
      }
      os << "|" << std::endl;
    }
  }
};

Matrix<double> eye(int n) {
  Matrix<double> result(n, n);
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      result[i][j] = (i == j) ? 1 : 0;
    }
  }
  return result;
}

struct LinearTransformMatrix {
  Matrix<double> data;
  LinearTransformMatrix() : data(eye(3)) {}

  LinearTransformMatrix &translate(double x, double y) {
    this->data = Matrix<double>({{1, 0, x}, {0, 1, y}, {0, 0, 1}}) * this->data;
    return *this;
  }

  LinearTransformMatrix &rotate(double theta) {
    this->data = Matrix<double>({{cos(theta), -sin(theta), 0},
                                 {sin(theta), cos(theta), 0},
                                 {0, 0, 1}}) *
                 this->data;
    return *this;
  }

  LinearTransformMatrix &scale(double x, double y) {
    this->data = Matrix<double>({{x, 0, 0}, {0, y, 0}, {0, 0, 1}}) * this->data;
    return *this;
  }

  LinearTransformMatrix &shear(double x, double y) {
    this->data = Matrix<double>({{1, x, 0}, {y, 1, 0}, {0, 0, 1}}) * this->data;
    return *this;
  }

  LinearTransformMatrix &perspective(double x, double y) {
    this->data = Matrix<double>({{1, 0, 0}, {0, 1, 0}, {x, y, 1}}) * this->data;
    return *this;
  }

  Matrix<double> take() { return this->data; }
};

} // namespace Linalg

#endif
