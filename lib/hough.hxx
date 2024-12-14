#ifndef IMAGE_PROCESSING_HOUGH_HXX
#define IMAGE_PROCESSING_HOUGH_HXX

#include "bmp_image.hxx"
#include "plot.hxx"
#include <algorithm>
#include <cmath>
#include <vector>

namespace Hough {

using RealMatrix = std::vector<std::vector<double>>;

struct HoughLineParam {
  int theta_steps = 1440; // 0 到 2π 的角度分辨率
  int rho_steps = 1024;   // ρ 的分辨率
  double rho_max = -1;    // 默认自动计算
};

RealMatrix hough_linear_transform(const RealMatrix &matrix,
                                  HoughLineParam &param, bool rect_mode = false,
                                  double rect_tolerant = 0.05) {
  auto &[theta_steps, rho_steps, rho_max] = param;

  if (rho_max == -1) { // 自动计算 rho_max
    rho_max = std::sqrt(matrix.size() * matrix.size() +
                        matrix[0].size() * matrix[0].size());
  }

  RealMatrix result(theta_steps, std::vector<double>(rho_steps, 0));
  for (int y = 0; y < matrix.size(); ++y) {
    for (int x = 0; x < matrix[0].size(); ++x) {
      if (matrix[y][x] > 1e-5) {
        for (int t_i = 0; t_i < theta_steps; ++t_i) {
          if (rect_mode) {
            double theta = t_i * 2 * M_PI / theta_steps;
            if (abs(theta) > rect_tolerant &&
                abs(theta - M_PI) > rect_tolerant &&
                abs(theta + M_PI) > rect_tolerant &&
                abs(theta - M_PI_2) > rect_tolerant &&
                abs(theta + M_PI_2) > rect_tolerant &&
                abs(theta - M_PI_2 * 3) > rect_tolerant &&
                abs(theta + M_PI_2 * 3) > rect_tolerant) {
              continue;
            }
          }
          double theta = t_i * 2 * M_PI / theta_steps;
          double rho = x * std::cos(theta) + y * std::sin(theta);
          int r_i =
              static_cast<int>((rho + rho_max) * rho_steps / (2 * rho_max));
          if (r_i >= 0 && r_i < rho_steps) {
            result[t_i][r_i] += matrix[y][x];
          }
        }
      }
    }
  }
  return result;
}

// 提取直线：从霍夫空间中找到阈值以上的直线
std::vector<std::tuple<double, double>> get_lines(const RealMatrix &matrix,
                                                  HoughLineParam &param,
                                                  double threshold = -1,
                                                  double auto_ratio = 0.5) {
  auto &[theta_steps, rho_steps, rho_max] = param;

  // 自动计算阈值
  if (threshold < 0) {
    std::vector<double> threshold_values;
    for (int i = 0; i < matrix.size(); ++i) {
      for (int j = 0; j < matrix[0].size(); ++j) {
        threshold_values.push_back(matrix[i][j]);
      }
    }
    std::sort(threshold_values.begin(), threshold_values.end(),
              std::greater<double>());
    threshold = threshold_values[0] * auto_ratio;
  }

  std::vector<std::tuple<double, double>> lines;
  for (int t_i = 0; t_i < theta_steps; ++t_i) {
    for (int r_i = 0; r_i < rho_steps; ++r_i) {
      if (matrix[t_i][r_i] > threshold) {
        // 峰值检测
        bool is_peak = true;
        for (int dt = -1; dt <= 1; ++dt) {
          for (int dr = -1; dr <= 1; ++dr) {
            if (dt == 0 && dr == 0)
              continue;
            int nt = (t_i + dt + theta_steps) % theta_steps;
            int nr = r_i + dr;
            if (nr >= 0 && nr < rho_steps &&
                matrix[nt][nr] > matrix[t_i][r_i]) {
              is_peak = false;
              break;
            }
          }
          if (!is_peak)
            break;
        }

        if (is_peak) {
          double theta = t_i * 2 * M_PI / theta_steps;
          double rho = r_i * (2 * rho_max) / rho_steps - rho_max;
          lines.emplace_back(theta, rho);
        }
      }
    }
  }
  return lines;
}

double calculate_distance(int t1, int r1, int t2, int r2, int theta_steps,
                          int rho_steps) {
  return sqrt((t1 - t2) * (t1 - t2) + (r1 - r2) * (r1 - r2));
}

// BFS to find nearby points within spread distance
std::vector<std::tuple<int, int>> bfs(const RealMatrix &matrix, int start_t,
                                      int start_r, double threshold, int spread,
                                      int theta_steps, int rho_steps) {
  std::vector<std::tuple<int, int>> group;
  std::queue<std::tuple<int, int>> queue;
  std::vector<std::vector<bool>> visited(theta_steps,
                                         std::vector<bool>(rho_steps, false));

  queue.push({start_t, start_r});
  visited[start_t][start_r] = true;

  while (!queue.empty()) {
    auto [t, r] = queue.front();
    queue.pop();
    group.push_back({t, r});

    // Explore neighbors within spread range
    for (int dt = -spread; dt <= spread; ++dt) {
      for (int dr = -spread; dr <= spread; ++dr) {
        if (dt == 0 && dr == 0)
          continue;

        int nt = (t + dt + theta_steps) % theta_steps;
        int nr = r + dr;
        if (nr >= 0 && nr < rho_steps && !visited[nt][nr] &&
            matrix[nt][nr] > threshold) {
          visited[nt][nr] = true;
          queue.push({nt, nr});
        }
      }
    }
  }

  return group;
}

std::vector<std::tuple<double, double>>
get_lines_bfs(const RealMatrix &matrix, HoughLineParam &param, int spread = 1,
              double threshold = -1, double auto_ratio = 0.5,
              bool rect_mode = false, double rect_tolerant = 0.05) {
  auto &[theta_steps, rho_steps, rho_max] = param;

  // Automatically calculate threshold if it's not provided
  if (threshold < 0) {
    std::vector<double> threshold_values;
    for (int i = 0; i < matrix.size(); ++i) {
      for (int j = 0; j < matrix[0].size(); ++j) {
        threshold_values.push_back(matrix[i][j]);
      }
    }
    std::sort(threshold_values.begin(), threshold_values.end(),
              std::greater<double>());
    threshold = threshold_values[0] * auto_ratio;
  }

  std::vector<std::tuple<double, double>> lines;
  std::vector<std::vector<bool>> visited(theta_steps,
                                         std::vector<bool>(rho_steps, false));

  // Iterate through the Hough space to find points above threshold
  for (int t_i = 0; t_i < theta_steps; ++t_i) {
    for (int r_i = 0; r_i < rho_steps; ++r_i) {
      double this_th = threshold;
      if (rect_mode) {
        double theta = t_i * 2 * M_PI / theta_steps;
        if (abs(theta) < rect_tolerant || abs(theta - M_PI) < rect_tolerant ||
            abs(theta + M_PI) < rect_tolerant ||
            abs(theta - 2 * M_PI) < rect_tolerant) {
          this_th /= 2;
        }
      }
      if (matrix[t_i][r_i] > this_th && !visited[t_i][r_i]) {
        // BFS to collect points in the neighborhood
        auto group =
            bfs(matrix, t_i, r_i, this_th, spread, theta_steps, rho_steps);

        // Calculate the average theta and rho for the group
        double sum_theta = 0;
        double sum_rho = 0;
        for (const auto &[t, r] : group) {
          sum_theta += t * 2 * M_PI / theta_steps;
          sum_rho += r * (2 * rho_max) / rho_steps - rho_max;
          visited[t][r] = true; // Mark these points as visited
        }

        if (!group.empty()) {
          double avg_theta = sum_theta / group.size();
          double avg_rho = sum_rho / group.size();
          lines.emplace_back(avg_theta, avg_rho);
        }
      }
    }
  }

  return lines;
}

// 在图像上绘制直线
BmpImage::BmpImage
draw_lines(const std::vector<std::tuple<double, double>> &lines,
           BmpImage::BmpImage &image,
           BmpImage::BmpPixel color = {255, 0, 0, 255}) {
  int width = image.header.infoHeader.width;
  int height = image.header.infoHeader.height;

  for (const auto &line : lines) {
    double theta = std::get<0>(line);
    double rho = std::get<1>(line);

    std::vector<std::pair<int, int>> points;

    // 与左右边界的交点
    if (std::abs(std::sin(theta)) > 1e-6) {
      int x1 = 0;
      int y1 = static_cast<int>(rho / std::sin(theta));
      if (y1 >= 0 && y1 < height)
        points.emplace_back(x1, y1);

      int x2 = width - 1;
      int y2 = static_cast<int>((rho - x2 * std::cos(theta)) / std::sin(theta));
      if (y2 >= 0 && y2 < height)
        points.emplace_back(x2, y2);
    }

    // 与上下边界的交点
    if (std::abs(std::cos(theta)) > 1e-6) {
      int y1 = 0;
      int x1 = static_cast<int>(rho / std::cos(theta));
      if (x1 >= 0 && x1 < width)
        points.emplace_back(x1, y1);

      int y2 = height - 1;
      int x2 = static_cast<int>((rho - y2 * std::sin(theta)) / std::cos(theta));
      if (x2 >= 0 && x2 < width)
        points.emplace_back(x2, y2);
    }

    // 如果找到两个端点，则绘制线段
    if (points.size() == 2) {
      Plot::draw_line(image, points[0].first, points[0].second, points[1].first,
                      points[1].second, color);
    }
  }
  return image;
}

BmpImage::BmpImage plot(const RealMatrix &matrix) {
  double max_val = 0;
  for (int i = 0; i < matrix.size(); i++) {
    for (int j = 0; j < matrix[0].size(); j++) {
      max_val = std::max(max_val, matrix[i][j]);
    }
  }
  auto res = Plot::generate_blank_canvas(matrix[0].size(), matrix.size());

  res.image.data.foreach ([&](BmpImage::BmpPixel &p, size_t idx) {
    int x = idx % matrix[0].size();
    int y = idx / matrix[0].size();
    uint8_t val = static_cast<uint8_t>(matrix[y][x] / max_val * 255);
    p = {val, val, val, 255};
  });
  return res;
}

// 定义一个Point类型来存储坐标
using Point = std::tuple<int, int>;

Point intersect(std::tuple<double, double> line1,
                std::tuple<double, double> line2) {
  double theta1 = std::get<0>(line1);
  double rho1 = std::get<1>(line1);
  double theta2 = std::get<0>(line2);
  double rho2 = std::get<1>(line2);
  double a = std::sin(theta1);
  double b = std::cos(theta1);
  double c = std::sin(theta2);
  double d = std::cos(theta2);
  double denominator = a * d - b * c;
  double x = (d * rho1 - b * rho2) / denominator;
  double y = (a * rho2 - c * rho1) / denominator;
  return {static_cast<int>(x), static_cast<int>(y)};
}

std::vector<Point> all_intersects(std::vector<std::tuple<double, double>> lines,
                                  double parallel_tolerance = 1.0) {
  std::vector<Point> intersects;
  for (int i = 0; i < lines.size(); i++) {
    for (int j = i + 1; j < lines.size(); j++) {
      double theta1 = std::get<0>(lines[i]);
      double rho1 = std::get<1>(lines[i]);
      double theta2 = std::get<0>(lines[j]);
      double rho2 = std::get<1>(lines[j]);
      if (std::abs(theta1 - theta2) < parallel_tolerance ||
          std::abs(theta1 - theta2 + M_PI) < parallel_tolerance ||
          std::abs(theta1 - theta2 - M_PI) < parallel_tolerance) {
        continue;
      }
      Point intersect_point = intersect(lines[i], lines[j]);
      if (std::get<0>(intersect_point) < 0 ||
          std::get<1>(intersect_point) < 0) {
        continue;
      }
      intersects.push_back(intersect_point);
    }
  }
  return intersects;
}

int cross(const Point &o, const Point &a, const Point &b) {
  return (std::get<0>(a) - std::get<0>(o)) * (std::get<1>(b) - std::get<1>(o)) -
         (std::get<0>(b) - std::get<0>(o)) * (std::get<1>(a) - std::get<1>(o));
}

// 计算凸包，使用Andrew算法
std::vector<Point> hull(std::vector<Point> points) {
  if (points.size() <= 1)
    return points;

  // 排序点集，按x坐标升序，若x相同则按y坐标升序
  std::sort(points.begin(), points.end());

  std::vector<Point> lower, upper;

  // 构建下半部分
  for (const auto &p : points) {
    while (lower.size() >= 2 &&
           cross(lower[lower.size() - 2], lower.back(), p) <= 0) {
      lower.pop_back();
    }
    lower.push_back(p);
  }

  // 构建上半部分
  for (auto it = points.rbegin(); it != points.rend(); ++it) {
    while (upper.size() >= 2 &&
           cross(upper[upper.size() - 2], upper.back(), *it) <= 0) {
      upper.pop_back();
    }
    upper.push_back(*it);
  }

  // 去掉重复的点（上下链表的最后一个点是重复的）
  lower.pop_back();
  upper.pop_back();

  // 合并下半部分和上半部分
  lower.insert(lower.end(), upper.begin(), upper.end());

  return lower;
}

} // namespace Hough
#endif