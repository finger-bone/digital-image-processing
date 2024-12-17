#ifndef IMAGE_PROCESSING_SEGMENTATION_HXX
#define IMAGE_PROCESSING_SEGMENTATION_HXX

#include "bmp_image.hxx"
#include <algorithm>
#include <functional>
#include <iterator>
#include <memory>
#include <queue>
#include <set>
#include <tuple>
#include <unordered_set>
#include <vector>

namespace Segmentation {

namespace SegmentationByThreshold {

BmpImage::BmpImage
segment_by_threshold(BmpImage::BmpImage &img_src, int threshold,
                     BmpImage::BmpPixel left_color = {0, 0, 0, 255},
                     BmpImage::BmpPixel right_color = {255, 255, 255, 255}) {
  auto img = img_src;
  img.image.data.foreach ([&](BmpImage::BmpPixel &pxl, size_t idx) {
    if (pxl.gray() < threshold) {
      pxl = left_color;
    } else {
      pxl = right_color;
    }
  });
  return img;
}

int auto_find_threshold_by_iteration(BmpImage::BmpImage &img_src,
                                     int max_iterations = 1000,
                                     double eps = 2) {
  auto img = img_src;
  int threshold = 128; // Initial threshold
  double left_mean = 0;
  double right_mean = 0;
  int left_count = 0;
  int right_count = 0;
  int iterations = 0;
  while (iterations < max_iterations) {
    left_mean = 0;
    right_mean = 0;
    left_count = 0;
    right_count = 0;
    img.image.data.foreach_sync([&](BmpImage::BmpPixel &pxl, size_t idx) {
      if (pxl.gray() < threshold) {
        left_mean += pxl.gray();
        left_count++;
      } else {
        right_mean += pxl.gray();
        right_count++;
      }
    });
    if (left_count == 0 || right_count == 0) {
      break;
    }
    left_mean /= left_count;
    right_mean /= right_count;
    if (left_mean == threshold || right_mean == threshold) {
      break;
    }
    if (threshold - left_mean < eps && right_mean - threshold < eps) {
      break;
    }
    threshold = (left_mean + right_mean) / 2;
    iterations++;
  }
  return threshold;
}

int auto_find_threshold_by_otsu(BmpImage::BmpImage &img_src) {
  auto img = img_src;
  int threshold = 0;
  double max_variance = 0;
  for (int i = 0; i <= 256; i++) {
    int left_count = 0;
    int right_count = 0;
    int left_sum = 0;
    int right_sum = 0;
    img.image.data.foreach_sync([&](BmpImage::BmpPixel &pxl, size_t idx) {
      if (pxl.gray() < i) {
        left_sum += pxl.gray();
        left_count++;
      } else {
        right_sum += pxl.gray();
        right_count++;
      }
    });
    if (left_count == 0 || right_count == 0) {
      continue;
    }
    double left_mean = static_cast<double>(left_sum) / left_count;
    double right_mean = static_cast<double>(right_sum) / right_count;
    double variance = left_count * right_count * (left_mean - right_mean) *
                      (left_mean - right_mean);
    if (variance > max_variance) {
      max_variance = variance;
      threshold = i;
    }
  }
  return threshold;
}
} // namespace SegmentationByThreshold

namespace SegmentationByGrowth {

using Point = std::tuple<int, int>;

std::set<Point>
grow_region(BmpImage::BmpImage &img_src, const std::set<Point> &seeds,
            std::function<bool(Point, const BmpImage::BmpImage &,
                               const std::set<Point> &)>
                validate,
            bool eight_direction = true) {
  auto img = img_src;
  std::set<Point> region = seeds;

  bool last_round_any_points_grown = true;
  const std::vector<Point> directions = {{0, -1},  {0, 1},  {-1, 0}, {1, 0},
                                         {-1, -1}, {-1, 1}, {1, -1}, {1, 1}};
  int max_iterations = 1280;
  int cnt = 0;
  while (last_round_any_points_grown) {
    if (cnt++ > max_iterations) {
      break;
    }
    last_round_any_points_grown = false;
    std::set<Point> next_try_points;

    for (const auto &p : region) {
      auto px = std::get<0>(p);
      auto py = std::get<1>(p);

      for (size_t i = 0; i < (eight_direction ? 8 : 4); ++i) {
        int nx = px + std::get<0>(directions[i]);
        int ny = py + std::get<1>(directions[i]);

        if (nx >= 0 && nx < img.image.size.width && ny >= 0 &&
            ny < img.image.size.height) {
          next_try_points.insert({nx, ny});
        }
      }
    }

    std::set<Point> next_points;
    std::set_difference(next_try_points.begin(), next_try_points.end(),
                        region.begin(), region.end(),
                        std::inserter(next_points, next_points.begin()));

    for (const auto &point : next_points) {
      if (validate(point, img, region)) {
        region.insert(point);
        last_round_any_points_grown = true;
      }
    }
  }

  return region;
}

std::vector<std::set<Point>>
border_trace(BmpImage::BmpImage &img_src,
             BmpImage::BmpPixel bg_color = {0, 0, 0, 255},
             BmpImage::BmpPixel fg_color = {255, 255, 255, 255},
             double color_tolerance = 8.0) {
  int width = img_src.image.size.width;
  int height = img_src.image.size.height;

  // Directions for the 8 neighbors (dx, dy)
  std::vector<Point> directions = {{-1, -1}, {-1, 0}, {-1, 1}, {0, -1},
                                   {0, 1},   {1, -1}, {1, 0},  {1, 1}};
  std::vector<Point> four_directions = {{-1, 0}, {1, 0}, {1, 0}, {-1, 0}};

  // Record which pixels are boundary pixels
  std::vector<std::vector<char>> is_border(height,
                                           std::vector<char>(width, false));

  // Identify the boundary pixels by comparing each pixel with its neighbors
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      // Skip pixels that are part of the background
      if (img_src.image.data.data[i * width + j].diff(bg_color) <
          color_tolerance) {
        continue;
      }

      // Check the neighbors
      bool is_border_pixel = false;
      for (const auto &[dx, dy] : directions) {
        int nx = j + dx;
        int ny = i + dy;

        // Ensure the neighbor is within bounds
        if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
          // Check if the neighbor is background color
          if (img_src.image.data.data[ny * width + nx].diff(bg_color) <
              color_tolerance) {
            is_border_pixel = true;
            break;
          }
        }
      }

      // If a border pixel is found, mark the pixel
      if (is_border_pixel) {
        is_border[i][j] = true;
      }
    }
  }

  // Store the boundaries
  std::vector<std::set<Point>> borders;
  // Track whether a pixel has been visited
  std::vector<std::vector<char>> visited(height,
                                         std::vector<char>(width, false));

  // Helper function to find the next unvisited boundary pixel
  auto find_next_border = [&]() {
    int x = -1, y = -1;
    // Scan the entire image to find an unvisited boundary pixel
    for (int i = 0; i < height; i++) {
      for (int j = 0; j < width; j++) {
        if (!visited[i][j] && is_border[i][j]) {
          x = j;
          y = i;
          break;
        }
      }
      if (x != -1 && y != -1) {
        break;
      }
    }

    if (x == -1 || y == -1) {
      return false; // No unvisited boundary pixel found
    }

    std::set<Point> border;
    border.insert({x, y});
    visited[y][x] = true; // Mark the starting point as visited

    // Use DFS to trace the boundary
    while (true) {
      bool found = false;
      // Check 8 neighboring directions
      for (const auto [dx, dy] : directions) {
        int nx = x + dx;
        int ny = y + dy;

        if (nx < 0 || nx >= width || ny < 0 || ny >= height) {
          continue;
        }
        if (is_border[ny][nx] && !visited[ny][nx]) {
          // If an unvisited boundary neighbor is found, move to it
          x = nx;
          y = ny;
          border.insert({x, y});
          visited[y][x] = true;
          found = true;
          break;
        }
      }

      if (!found) {
        break; // If no neighbor is found, exit the loop
      }
    }
    if (border.size() > 2) {
      borders.push_back(border);
    }
    // Add the completed border to the list of borders
    return true;
  };

  // Find all boundaries in the image
  while (find_next_border()) {
  }

  return borders;
}
std::vector<std::set<Point>>
split_region(BmpImage::BmpImage &img_src,
             BmpImage::BmpPixel bg_color = {0, 0, 0, 255},
             BmpImage::BmpPixel fg_color = {255, 255, 255, 255},
             double color_tolerance = 8.0) {

  int width = img_src.image.size.width;
  int height = img_src.image.size.height;

  std::vector<std::vector<int>> labels(height, std::vector<int>(width, 0));
  std::map<int, int> parent;
  int current_label = 0;

  const std::vector<Point> directions = {{0, -1}, {-1, -1}, {-1, 0}, {-1, 1}};
  auto find = [&](int label) -> int {
    while (parent[label] != label) {
      parent[label] = parent[parent[label]];
      label = parent[label];
    }
    return label;
  };

  auto unite = [&](int label1, int label2) {
    int root1 = find(label1);
    int root2 = find(label2);
    if (root1 != root2) {
      parent[root2] = root1;
    }
  };

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      BmpImage::BmpPixel &pixel = img_src.image[y * width + x];
      if (pixel.diff(bg_color) < color_tolerance) {
        continue;
      }

      std::set<int> neighbor_labels;

      for (const auto &[dx, dy] : directions) {
        int nx = x + dx;
        int ny = y + dy;

        if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
          int neighbor_label = labels[ny][nx];
          if (neighbor_label > 0) {
            neighbor_labels.insert(neighbor_label);
          }
        }
      }

      if (neighbor_labels.empty()) {
        current_label++;
        labels[y][x] = current_label;
        parent[current_label] = current_label;
      } else {
        int min_label =
            *std::min_element(neighbor_labels.begin(), neighbor_labels.end());
        labels[y][x] = min_label;

        for (int label : neighbor_labels) {
          unite(label, min_label);
        }
      }
    }
  }

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      int label = labels[y][x];
      if (label > 0) {
        labels[y][x] = find(label);
      }
    }
  }

  std::map<int, std::set<Point>> regions;
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      int label = labels[y][x];
      if (label > 0) {
        regions[label].insert({x, y});
      }
    }
  }

  std::vector<std::set<Point>> result;
  for (const auto &[label, points] : regions) {
    if (!points.empty()) {
      result.push_back(points);
    }
  }

  return result;
}

std::vector<std::set<Point>>
get_borders(BmpImage::BmpImage &img_src,
            BmpImage::BmpPixel bg_color = {0, 0, 0, 255},
            BmpImage::BmpPixel fg_color = {255, 255, 255, 255},
            double color_tolerance = 8.0) {
  std::vector<std::set<Point>> regions =
      split_region(img_src, bg_color, fg_color, color_tolerance);
  const std::vector<Point> directions = {{0, -1}, {0, 1}, {-1, 0}, {1, 0}};

  std::vector<std::set<Point>> borders;
  for (const auto &region : regions) {
    std::set<Point> border_points;

    for (const auto &[x, y] : region) {
      bool is_border = false;
      for (const auto &[dx, dy] : directions) {
        int nx = x + dx;
        int ny = y + dy;

        Point neighbor = {nx, ny};
        if (region.find(neighbor) == region.end()) {
          is_border = true;
          break;
        }
      }

      if (is_border) {
        border_points.insert(std::make_tuple(x, y));
      }
    }

    borders.push_back(border_points);
  }

  return borders;
}
} // namespace SegmentationByGrowth

namespace SegmentationByQuadTree {
struct Box {
  int l;
  int r;
  int t;
  int b;
  bool is_adjacent(const Box &other) const {
    // Check horizontal adjacency
    if (r == other.l || l == other.r) {
      return true;
    }
    // Check vertical adjacency
    if (b == other.t || t == other.b) {
      return true;
    }
    return false;
  }
};

struct QuadTreeNode {
  std::array<std::shared_ptr<QuadTreeNode>, 4> children;
  Box box;
  bool is_leaf = false;
};

using HomogeneousFunction =
    std::function<bool(const BmpImage::BmpImage &, const std::vector<Box> &)>;

std::shared_ptr<QuadTreeNode>
build_quad_tree(const BmpImage::BmpImage &img_src,
                HomogeneousFunction homogenous_function, Box box) {
  auto node = std::make_shared<QuadTreeNode>();
  node->box = box;
  node->is_leaf = false;
  if (homogenous_function(img_src, {node->box})) {
    node->is_leaf = true;
    return node;
  } else {
    auto mid_x = (box.l + box.r) / 2;
    auto mid_y = (box.t + box.b) / 2;
    node->children[0] = build_quad_tree(img_src, homogenous_function,
                                        {box.l, mid_x, box.t, mid_y});
    node->children[1] = build_quad_tree(img_src, homogenous_function,
                                        {mid_x, box.r, box.t, mid_y});
    node->children[2] = build_quad_tree(img_src, homogenous_function,
                                        {box.l, mid_x, mid_y, box.b});
    node->children[3] = build_quad_tree(img_src, homogenous_function,
                                        {mid_x, box.r, mid_y, box.b});
    return node;
  }
}

std::vector<Box> get_leaf_boxes(std::shared_ptr<QuadTreeNode> node) {
  std::vector<Box> boxes;
  if (node->is_leaf) {
    boxes.push_back(node->box);
  } else {
    for (auto &child : node->children) {
      auto child_boxes = get_leaf_boxes(child);
      boxes.insert(boxes.end(), child_boxes.begin(), child_boxes.end());
    }
  }
  return boxes;
}
} // namespace SegmentationByQuadTree
} // namespace Segmentation

#endif