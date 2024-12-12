#ifndef DIGITAL_IMAGE_PROCESSING_FREQUENCY_HXX
#define DIGITAL_IMAGE_PROCESSING_FREQUENCY_HXX

#include "bmp_image.hxx"
#include "plot.hxx"
#include <algorithm>
#include <cmath>
#include <tuple>
#include <vector>

namespace Frequency {

struct Complex {
  double real;
  double imag;

  Complex(double r = 0, double i = 0) : real(r), imag(i) {}

  Complex operator+(const Complex &other) const {
    return {real + other.real, imag + other.imag};
  }

  Complex operator-(const Complex &other) const {
    return {real - other.real, imag - other.imag};
  }

  Complex operator*(const Complex &other) const {
    return {real * other.real - imag * other.imag,
            real * other.imag + imag * other.real};
  }

  double magnitude() const { return sqrt(real * real + imag * imag); }
  double phase() const { return atan2(imag, real); }
};

using ComplexMatrix = std::vector<std::vector<Complex>>;
using ComplexVector = std::vector<Complex>;
using RealMatrix = std::vector<std::vector<double>>;

void fft_1d(ComplexVector &vec, bool inverse = false) {
  int n = vec.size();
  if (n <= 1)
    return;

  ComplexVector even(n / 2), odd(n / 2);
  for (int i = 0; i < n; ++i) {
    if (i % 2 == 0)
      even[i / 2] = vec[i];
    else
      odd[i / 2] = vec[i];
  }

  fft_1d(even, inverse);
  fft_1d(odd, inverse);

  double angle = (inverse ? 2 : -2) * M_PI / n;
  Complex w_n{cos(angle), sin(angle)}, w{1, 0};

  for (int i = 0; i < n / 2; ++i) {
    Complex t = w * odd[i];
    vec[i] = even[i] + t;
    vec[i + n / 2] = even[i] - t;
    w = w * w_n;
  }
}

void fft_2d(ComplexMatrix &mat, bool inverse = false) {
  int n = mat.size();
  int m = mat[0].size();

  // Transform rows
  for (int i = 0; i < n; ++i) {
    fft_1d(mat[i], inverse);
  }

  // Transform columns
  for (int j = 0; j < m; ++j) {
    ComplexVector column(n);
    for (int i = 0; i < n; ++i) {
      column[i] = mat[i][j];
    }
    fft_1d(column, inverse);
    for (int i = 0; i < n; ++i) {
      mat[i][j] = column[i];
    }
  }

  // Normalize if inverse
  if (inverse) {
    double scale = 1.0 / (n * m);
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < m; ++j) {
        mat[i][j].real *= scale;
        mat[i][j].imag *= scale;
      }
    }
  }
}

ComplexMatrix fft(const RealMatrix &matrix) {
  int n = matrix.size();
  int m = matrix[0].size();

  ComplexMatrix result(n, std::vector<Complex>(m));
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < m; ++j) {
      result[i][j] = {matrix[i][j], 0};
    }
  }
  fft_2d(result);
  return result;
}

RealMatrix ifft(const ComplexMatrix &matrix) {
  int n = matrix.size();
  int m = matrix[0].size();

  ComplexMatrix result = matrix;
  fft_2d(result, true);

  RealMatrix res(n, std::vector<double>(m));
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < m; ++j) {
      res[i][j] = result[i][j].real;
    }
  }
  return res;
}

void cutoff_freq(ComplexMatrix &matrix, double cutoff, bool remove_high = false,
                 double value = 0) {
  for (int i = 0; i < matrix.size(); i++) {
    for (int j = 0; j < matrix[0].size(); j++) {
      if (remove_high) {
        if (i > cutoff && j > cutoff) {
          matrix[i][j] = {value, 0};
        }
      } else {
        if (i < cutoff && j < cutoff) {
          matrix[i][j] = {value, 0};
        }
      }
    }
  }
}

BmpImage::BmpImage plot(const RealMatrix &matrix) {
  auto res = Plot::generate_blank_canvas(matrix[0].size(), matrix.size());
  res.image.data.foreach ([&](BmpImage::BmpPixel &p, size_t idx) {
    int x = idx % matrix[0].size();
    int y = idx / matrix[0].size();
    uint8_t val = static_cast<uint8_t>(matrix[y][x]);
    p = {val, val, val, 255};
  });
  return res;
}

std::tuple<RealMatrix, RealMatrix>
polar_transform(const ComplexMatrix &matrix) {
  int n = matrix.size();
  int m = matrix[0].size();

  RealMatrix magnitudes(n, std::vector<double>(m));
  RealMatrix phases(n, std::vector<double>(m));

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      magnitudes[i][j] = matrix[i][j].magnitude();
      phases[i][j] = matrix[i][j].phase();
    }
  }
  return std::make_tuple(magnitudes, phases);
}

int next_power_of_two(int n) {
  int power = 1;
  while (power < n) {
    power *= 2;
  }
  return power;
}

void pad(RealMatrix &matrix) {
  int n = matrix.size();
  int m = matrix[0].size();
  int new_n = next_power_of_two(n);
  int new_m = next_power_of_two(m);
  RealMatrix padded(new_n, std::vector<double>(new_m, 0));
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      padded[i][j] = matrix[i][j];
    }
  }
  matrix = padded;
}

} // namespace Frequency

#endif