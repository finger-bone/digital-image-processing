#ifndef IMAGE_PROCESSING_BAR_PLOT_HXX
#define IMAGE_PROCESSING_BAR_PLOT_HXX

#include "bmp_image.hxx"
#include "numeric_array.hxx"

#include <set>
#include <tuple>
#include <vector>

namespace Plot {
BmpImage::BmpImage generate_blank_canvas(int width, int height,
                                         BmpImage::BmpPixel color = {
                                             255, 255, 255, 255}) {
  BmpImage::BmpHeader header{.fileHeader =
                                 {
                                     .fileType = 0x4D42,
                                 },
                             .infoHeader = {
                                 .headerSize = 40,
                                 .width = width,
                                 .height = height,
                                 .planes = 1,
                                 .bitsPerPixel = 24,
                             }};
  return BmpImage::BmpImage{
      .header = header,
      .image = {.size =
                    {
                        .width = width,
                        .height = height,
                    },
                .data = NumericArray::NumericArray<BmpImage::BmpPixel>{
                    std::vector<BmpImage::BmpPixel>(width * height, color),
                }}};
}

void draw_line(BmpImage::BmpImage &image, int x1, int y1, int x2, int y2,
               BmpImage::BmpPixel color = {255, 0, 0, 255}) {
  // x and y are from bottom left corner
  int dx = x2 - x1;
  int dy = y2 - y1;
  int steps = std::max(std::abs(dx), std::abs(dy));
  double x_inc = dx / (double)steps;
  double y_inc = dy / (double)steps;
  for (int i = 0; i < steps; i++) {
    int x = x1 + i * x_inc;
    int y = y1 + i * y_inc;
    if (x < 0 || x >= image.image.size.width || y < 0 ||
        y >= image.image.size.height) {
      continue;
    }
    image.image.data.data[y * image.image.size.width + x] = color;
  }
}

void draw_points(BmpImage::BmpImage &image,
                 const std::set<std::tuple<int, int>> &points,
                 BmpImage::BmpPixel color = {255, 0, 0, 255}) {
  image.image.data.foreach ([&](BmpImage::BmpPixel &p, size_t idx) {
    int x = idx % image.image.size.width;
    int y = idx / image.image.size.width;
    if (points.find({x, y}) != points.end()) {
      p = color;
    }
  });
}

void draw_a_point(BmpImage::BmpImage &image, int x, int y,
                  BmpImage::BmpPixel color = {255, 255, 0, 255}) {
  image.image.data.data[y * image.image.size.width + x] = color;
}

void draw_box(BmpImage::BmpImage &image, int l, int r, int t, int b,
              BmpImage::BmpPixel color = {255, 0, 0, 255}) {
  draw_line(image, l, t, l, b, color);
  draw_line(image, r, t, r, b, color);
  draw_line(image, l, t, r, t, color);
  draw_line(image, l, b, r, b, color);
}

void bar_plot(BmpImage::BmpImage &image, std::vector<int> values, int chunks,
              BmpImage::BmpPixel color = {0, 0, 0, 255}) {
  int width = image.image.size.width;
  int height = image.image.size.height;
  int bar_width = width / chunks;
  int max_value = *std::max_element(values.begin(), values.end());
  int chunk_size = values.size() / chunks;
  std::vector<int> bar_heights(chunks);
  for (int i = 0; i < chunks; i++) {
    int sum = 0;
    for (int j = 0; j < chunk_size; j++) {
      sum += values[i * chunk_size + j];
    }
    bar_heights[i] = height * sum / chunk_size / max_value;
  }

  image.image.data.foreach ([&](BmpImage::BmpPixel &p, size_t idx) {
    int x = idx % width;
    int y = idx / width;
    // find the correct bar of x
    int bar_idx = x / bar_width;
    int bar_height = bar_heights[bar_idx];
    if (bar_idx < 0 || bar_idx >= chunks) {
      return;
    }
    if (y <= bar_height) {
      p = color;
      return;
    } else {
      return;
    }
  });
}

BmpImage::BmpImage generate_gray_scale_histogram(BmpImage::BmpImage &image,
                                                 int width = 256,
                                                 int height = 256,
                                                 int chunks = 256) {
  BmpImage::BmpImage plot = generate_blank_canvas(width, height);
  std::vector<int> values(256);
  image.image.data.foreach_sync([&](BmpImage::BmpPixel p, size_t idx) {
    int gray_value = p.gray();
    values[gray_value]++;
  });
  bar_plot(plot, values, chunks);
  return plot;
}
} // namespace Plot

#endif