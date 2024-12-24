#include "lib/bmp_image.hxx"
#include "lib/convolution.hxx"
#include "lib/frequency.hxx"
#include "lib/hough.hxx"
#include "lib/linalg.hxx"
#include "lib/linear_transform.hxx"
#include "lib/numeric_array.hxx"
#include "lib/plot.hxx"
#include "lib/segmentation.hxx"
#include "lib/terminal_print.hxx"

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <thread>
#include <vector>

#define RESET "\033[0m"
#define BOLD "\033[1m"
#define RED "\033[31m"
#define GREEN "\033[32m"
#define YELLOW "\033[33m"
#define BLUE "\033[34m"
#define CYAN "\033[36m"

std::vector<BmpImage::BmpPixel> random_colors = std::vector<BmpImage::BmpPixel>{
    BmpImage::BmpPixel{255, 0, 0, 255},
    BmpImage::BmpPixel{0, 255, 0, 255},
    BmpImage::BmpPixel{0, 0, 255, 255},
    BmpImage::BmpPixel{255, 255, 0, 255},
    BmpImage::BmpPixel{255, 0, 255, 255},
    BmpImage::BmpPixel{0, 255, 255, 255},
    BmpImage::BmpPixel{64, 128, 128, 255},
    BmpImage::BmpPixel{128, 64, 128, 255},
    BmpImage::BmpPixel{128, 128, 64, 255},
    BmpImage::BmpPixel{128, 64, 64, 255},
    BmpImage::BmpPixel{64, 128, 64, 255},
    BmpImage::BmpPixel{64, 64, 128, 255},
    BmpImage::BmpPixel{255, 128, 0, 255},
    BmpImage::BmpPixel{128, 255, 0, 255},
    BmpImage::BmpPixel{128, 0, 255, 255},
    BmpImage::BmpPixel{255, 0, 128, 255},
    BmpImage::BmpPixel{0, 255, 128, 255},
    BmpImage::BmpPixel{0, 128, 255, 255},
    BmpImage::BmpPixel{32, 64, 72, 255},
    BmpImage::BmpPixel{64, 32, 72, 255},
    BmpImage::BmpPixel{72, 32, 64, 255},
    BmpImage::BmpPixel{72, 64, 32, 255},
    BmpImage::BmpPixel{64, 72, 32, 255},
    BmpImage::BmpPixel{32, 72, 64, 255},
};

void task1(std::string path) {
  std::ifstream in_file(path, std::ios::binary);
  auto raw_img = BmpImage::read_bmp(in_file);
  // copy the img
  auto gray_img = raw_img;
  gray_img.image.data.foreach ([](BmpImage::BmpPixel &pixel) {
    auto gray = static_cast<uint8_t>(pixel.gray());
    pixel = BmpImage::BmpPixel{
        .red = gray,
        .green = gray,
        .blue = gray,
        .alpha = pixel.alpha,
    };
  });
  gray_img.change_to_eight_bit();

  std::ofstream gray_img_file("output/gray_img.bmp", std::ios::binary);
  BmpImage::write_bmp(gray_img_file, gray_img);
  // invert the image
  auto inverted_grey_img = gray_img;
  inverted_grey_img.image.data.foreach ([](BmpImage::BmpPixel &pixel) {
    pixel = BmpImage::BmpPixel{
        .red = static_cast<uint8_t>(255 - pixel.red),
        .green = static_cast<uint8_t>(255 - pixel.green),
        .blue = static_cast<uint8_t>(255 - pixel.blue),
        .alpha = pixel.alpha,
    };
  });
  inverted_grey_img.change_to_eight_bit();

  std::ofstream inverted_img_file("output/inverted_img.bmp", std::ios::binary);
  BmpImage::write_bmp(inverted_img_file, inverted_grey_img);

  // split the RGB channels
  auto r_img = raw_img;
  r_img.image.data.foreach ([](BmpImage::BmpPixel &pixel) {
    pixel = BmpImage::BmpPixel{
        .red = pixel.red,
        .green = pixel.red,
        .blue = pixel.red,
        .alpha = pixel.alpha,
    };
  });
  r_img.change_to_eight_bit();

  std::ofstream r_img_file("output/r_img.bmp", std::ios::binary);
  BmpImage::write_bmp(r_img_file, r_img);
  auto g_img = raw_img;
  g_img.image.data.foreach ([](BmpImage::BmpPixel &pixel) {
    pixel = BmpImage::BmpPixel{
        .red = pixel.green,
        .green = pixel.green,
        .blue = pixel.green,
        .alpha = pixel.alpha,
    };
  });
  g_img.change_to_eight_bit();

  std::ofstream g_img_file("output/g_img.bmp", std::ios::binary);
  BmpImage::write_bmp(g_img_file, g_img);
  auto b_img = raw_img;
  b_img.image.data.foreach ([](BmpImage::BmpPixel &pixel) {
    pixel = BmpImage::BmpPixel{
        .red = pixel.blue,
        .green = pixel.blue,
        .blue = pixel.blue,
        .alpha = pixel.alpha,
    };
  });
  b_img.change_to_eight_bit();

  std::ofstream b_img_file("output/b_img.bmp", std::ios::binary);
  BmpImage::write_bmp(b_img_file, b_img);
}

void task2(std::string path) {
  std::ifstream in_file(path, std::ios::binary);
  auto raw_img = BmpImage::read_bmp(in_file);

  auto hist = Plot::generate_gray_scale_histogram(raw_img);

  std::ofstream hist_file("output/hist_before.bmp", std::ios::binary);
  BmpImage::write_bmp(hist_file, hist);
  auto balanced_img = BmpImage::gray_balanced_image(raw_img);

  std::ofstream balanced_img_file("output/balanced_img.bmp", std::ios::binary);
  BmpImage::write_bmp(balanced_img_file, balanced_img);
  auto balanced_hist = Plot::generate_gray_scale_histogram(balanced_img);

  std::ofstream balanced_hist_file("output/hist_after.bmp", std::ios::binary);
  BmpImage::write_bmp(balanced_hist_file, balanced_hist);
}

void task3(std::string path) {
  std::ifstream in_file(path, std::ios::binary);
  auto raw_img = BmpImage::read_bmp(in_file);

  auto value = 1.0 / 25;
  auto avg_filtered_image = Convolution::apply_kernel(
      raw_img, {
                   {value, value, value, value, value},
                   {value, value, value, value, value},
                   {value, value, value, value, value},
                   {value, value, value, value, value},
                   {value, value, value, value, value},
               });

  std::ofstream avg_filtered_file("output/avg_filtered.bmp", std::ios::binary);
  BmpImage::write_bmp(avg_filtered_file, avg_filtered_image);
  auto mid_filtered_image =
      Convolution::apply_mid_value_kernel(raw_img, 5, (5 * 5) / 2);

  std::ofstream mid_filtered_file("output/mid_filtered.bmp", std::ios::binary);
  BmpImage::write_bmp(mid_filtered_file, mid_filtered_image);
}

void task3_with_parameters(std::string path, int kernel_size) {
  std::ifstream in_file(path, std::ios::binary);
  auto raw_img = BmpImage::read_bmp(in_file);

  auto value = 1.0 / kernel_size;
  std::vector<std::vector<double>> kernel(
      kernel_size, std::vector<double>(kernel_size, value));

  auto avg_filtered_image = Convolution::apply_kernel(raw_img, kernel);

  std::ofstream avg_filtered_file("output/avg_filtered.bmp", std::ios::binary);
  BmpImage::write_bmp(avg_filtered_file, avg_filtered_image);
  auto mid_filtered_image = Convolution::apply_mid_value_kernel(
      raw_img, kernel_size, (kernel_size * kernel_size) / 2);

  std::ofstream mid_filtered_file("output/mid_filtered.bmp", std::ios::binary);
}

void task4(std::string path) {
  std::ifstream in_file(path, std::ios::binary);
  auto raw_img = BmpImage::read_bmp(in_file);

  raw_img.change_to_twenty_four_bit();
  auto scaled_img = LinearTransform::linear_transform(
      raw_img, Linalg::LinearTransformMatrix().scale(0.5, 0.5).take());

  std::ofstream scaled_img_file("output/scaled_img.bmp", std::ios::binary);
  BmpImage::write_bmp(scaled_img_file, scaled_img);

  auto rotated_img = LinearTransform::linear_transform(
      raw_img, Linalg::LinearTransformMatrix().rotate(3.14 / 4).take());
  std::ofstream rotated_img_file("output/rotated_img.bmp", std::ios::binary);
  BmpImage::write_bmp(rotated_img_file, rotated_img);

  auto translated_img = LinearTransform::linear_transform(
      raw_img, Linalg::LinearTransformMatrix().translate(100, 100).take());
  std::ofstream translated_img_file("output/translated_img.bmp",
                                    std::ios::binary);
  BmpImage::write_bmp(translated_img_file, translated_img);

  auto flipped_image = LinearTransform::linear_transform(
      raw_img, Linalg::LinearTransformMatrix()
                   .translate(-raw_img.header.infoHeader.width, 0)
                   .scale(-1, 1)
                   .take());
  std::ofstream flipped_img_file("output/flipped_img.bmp", std::ios::binary);
  BmpImage::write_bmp(flipped_img_file, flipped_image);

  auto half_height = static_cast<double>(raw_img.header.infoHeader.height) / 2;
  auto perspective_img = LinearTransform::linear_transform(
      raw_img,
      Linalg::LinearTransformMatrix()
          .perspective_by_points(
              {std::make_tuple(0., 0.),
               std::make_tuple(0., raw_img.header.infoHeader.height),
               std::make_tuple(582., 582.),
               std::make_tuple(582., raw_img.header.infoHeader.height - 582.)},
              {std::make_tuple(0., 0.),
               std::make_tuple(0., raw_img.header.infoHeader.height),
               std::make_tuple(raw_img.header.infoHeader.width,
                               raw_img.header.infoHeader.height),
               std::make_tuple(raw_img.header.infoHeader.width, 0.)})
          .take());
  std::ofstream perspective_img_file("output/perspective_img.bmp",
                                     std::ios::binary);
  BmpImage::write_bmp(perspective_img_file, perspective_img);

  auto combined =
      LinearTransform::linear_transform(raw_img, Linalg::LinearTransformMatrix()
                                                     .scale(0.5, 0.5)
                                                     .translate(10, -10)
                                                     .rotate(3.14 / 4)
                                                     .take());
  std::ofstream combined_img_file("output/combined_img.bmp", std::ios::binary);
  BmpImage::write_bmp(combined_img_file, combined);
}

void task4_with_parameters(std::string path, double scale, double translate_x,
                           double translate_y, double rotate) {
  std::ifstream in_file(path, std::ios::binary);
  auto raw_img = BmpImage::read_bmp(in_file);

  raw_img.change_to_twenty_four_bit();
  auto scaled_img = LinearTransform::linear_transform(
      raw_img, Linalg::LinearTransformMatrix().scale(scale, scale).take());
  std::ofstream scaled_img_file("output/scaled_img.bmp", std::ios::binary);
  BmpImage::write_bmp(scaled_img_file, scaled_img);

  auto rotated_img = LinearTransform::linear_transform(
      raw_img, Linalg::LinearTransformMatrix().rotate(rotate).take());
  std::ofstream rotated_img_file("output/rotated_img.bmp", std::ios::binary);

  auto translated_img = LinearTransform::linear_transform(
      raw_img, Linalg::LinearTransformMatrix()
                   .translate(translate_x, translate_y)
                   .take());
  std::ofstream translated_img_file("output/translated_img.bmp",
                                    std::ios::binary);
  BmpImage::write_bmp(translated_img_file, translated_img);

  auto flipped_image = LinearTransform::linear_transform(
      raw_img, Linalg::LinearTransformMatrix()
                   .translate(-raw_img.header.infoHeader.width, 0)
                   .scale(-1, 1)
                   .take());
  std::ofstream flipped_img_file("output/flipped_img.bmp", std::ios::binary);
  BmpImage::write_bmp(flipped_img_file, flipped_image);
}

void task5(std::string path) {
  std::ifstream in_file(path, std::ios::binary);
  auto raw_img = BmpImage::read_bmp(in_file);

  auto segmented_img =
      Segmentation::SegmentationByThreshold::segment_by_threshold(raw_img, 128);

  std::ofstream segmented_img_file("output/segmented.bmp", std::ios::binary);
  BmpImage::write_bmp(segmented_img_file, segmented_img);
  BmpImage::BmpImage segmented_img_histogram =
      Plot::generate_gray_scale_histogram(raw_img);
  Plot::draw_line(segmented_img_histogram, 128, 256, 128, 0);
  std::ofstream segmented_img_histogram_file("output/segmented_histogram.bmp",
                                             std::ios::binary);
  BmpImage::write_bmp(segmented_img_histogram_file, segmented_img_histogram);
  auto th_by_iteration =
      Segmentation::SegmentationByThreshold::auto_find_threshold_by_iteration(
          raw_img);
  auto segmented_img_by_iteration =
      Segmentation::SegmentationByThreshold::segment_by_threshold(
          raw_img, th_by_iteration);

  std::ofstream segmented_img_by_iteration_file(
      "output/segmented_img_by_iteration.bmp", std::ios::binary);
  BmpImage::write_bmp(segmented_img_by_iteration_file,
                      segmented_img_by_iteration);
  auto segmented_by_iteration_histogram =
      Plot::generate_gray_scale_histogram(raw_img);
  Plot::draw_line(segmented_by_iteration_histogram, th_by_iteration, 256,
                  th_by_iteration, 0);
  std::ofstream segmented_by_iteration_histogram_file(
      "output/segmented_by_iteration_histogram.bmp", std::ios::binary);
  BmpImage::write_bmp(segmented_by_iteration_histogram_file,
                      segmented_by_iteration_histogram);
  auto th_by_otsu =
      Segmentation::SegmentationByThreshold::auto_find_threshold_by_otsu(
          raw_img);
  auto segmented_img_by_otsu =
      Segmentation::SegmentationByThreshold::segment_by_threshold(raw_img,
                                                                  th_by_otsu);

  std::ofstream segmented_img_by_otsu_file("output/segmented_img_by_otsu.bmp",
                                           std::ios::binary);
  BmpImage::write_bmp(segmented_img_by_otsu_file, segmented_img_by_otsu);
  auto segmented_by_otsu_histogram =
      Plot::generate_gray_scale_histogram(raw_img);
  Plot::draw_line(segmented_by_otsu_histogram, th_by_otsu, 256, th_by_otsu, 0);
  std::ofstream segmented_by_otsu_histogram_file(
      "output/segmented_by_otsu_histogram.bmp", std::ios::binary);
  BmpImage::write_bmp(segmented_by_otsu_histogram_file,
                      segmented_by_otsu_histogram);
}

void task5_with_parameters(std::string path, int threshold) {
  task5(path);
  std::ifstream in_file(path, std::ios::binary);
  auto raw_img = BmpImage::read_bmp(in_file);

  auto segmented_img =
      Segmentation::SegmentationByThreshold::segment_by_threshold(raw_img,
                                                                  threshold);

  std::ofstream segmented_img_file("output/segmented.bmp", std::ios::binary);
  BmpImage::write_bmp(segmented_img_file, segmented_img);
}

void task6(std::string path) {
  std::ifstream in_file(path, std::ios::binary);
  auto raw_img = BmpImage::read_bmp(in_file);
  raw_img.change_to_twenty_four_bit();

  auto seed_segmented_img = raw_img;

  auto seeds = std::set<std::tuple<int, int>>({
      {0, 0},
      {seed_segmented_img.header.infoHeader.width - 1, 0},
      {0, seed_segmented_img.header.infoHeader.height - 1},
      {seed_segmented_img.header.infoHeader.width - 1,
       seed_segmented_img.header.infoHeader.height - 1},
  });
  int min_gray = 0;
  int max_gray = 0;
  bool first = true;
  auto validate = [&min_gray, &max_gray,
                   &first](std::tuple<int, int> next_point,
                           const BmpImage::BmpImage &img,
                           const std::set<std::tuple<int, int>> &region) {
    if (first) {
      first = false;
      min_gray = img.image.data
                     .data[std::get<1>(next_point) * img.image.size.width +
                           std::get<0>(next_point)]
                     .gray();
      max_gray = min_gray;
      return true;
    }
    auto x = std::get<0>(next_point);
    auto y = std::get<1>(next_point);
    auto gray = img.image.data.data[y * img.image.size.width + x].gray();
    auto next_min_gray = std::min(min_gray, static_cast<int>(gray));
    auto next_max_gray = std::max(max_gray, static_cast<int>(gray));
    if (next_max_gray - next_min_gray < 64) {
      min_gray = next_min_gray;
      max_gray = next_max_gray;
      return true;
    } else {
      return false;
    }
  };
  auto res = Segmentation::SegmentationByGrowth::grow_region(
      seed_segmented_img, seeds, validate, false);
  Plot::draw_points(seed_segmented_img, res, {128, 0, 0, 255});
  Plot::draw_points(seed_segmented_img, seeds, {255, 128, 128, 255});
  auto seed2 = std::set<std::tuple<int, int>>({
      {static_cast<int>(seed_segmented_img.header.infoHeader.width * 0.4),
       static_cast<int>(seed_segmented_img.header.infoHeader.height * 0.6)},
      {static_cast<int>(seed_segmented_img.header.infoHeader.width * 0.7),
       static_cast<int>(seed_segmented_img.header.infoHeader.height * 0.6)},
      {static_cast<int>(seed_segmented_img.header.infoHeader.width * 0.4),
       static_cast<int>(seed_segmented_img.header.infoHeader.height * 0.4)},
      {static_cast<int>(seed_segmented_img.header.infoHeader.width * 0.7),
       static_cast<int>(seed_segmented_img.header.infoHeader.height * 0.4)},
  });
  min_gray = 0;
  max_gray = 0;
  first = true;
  auto res2 = Segmentation::SegmentationByGrowth::grow_region(
      seed_segmented_img, seed2, validate, false);
  Plot::draw_points(seed_segmented_img, res2, {0, 128, 0, 255});
  Plot::draw_points(seed_segmented_img, seed2, {128, 255, 128, 255});

  std::ofstream seed_segmented_img_file("output/seed_segmented_img.bmp",
                                        std::ios::binary);
  BmpImage::write_bmp(seed_segmented_img_file, seed_segmented_img);

  Segmentation::SegmentationByQuadTree::HomogeneousFunction func =
      [](const BmpImage::BmpImage &img,
         std::vector<Segmentation::SegmentationByQuadTree::Box> boxes) {
        auto is_homogeneous = true;

        for (auto box : boxes) {
          auto [l, r, t, b] = box;
          auto width = r - l;
          auto height = b - t;
          if (width <= 8 || height <= 8) {
            continue;
          }

          double sum = 0.0;
          double sum_squared = 0.0;
          int count = 0;

          for (int y = t; y < b; ++y) {
            for (int x = l; x < r; ++x) {
              auto gray =
                  img.image.data.data[y * img.image.size.width + x].gray();
              sum += gray;
              sum_squared += gray * gray;
              ++count;
            }
          }

          if (count > 0) {
            double mean = sum / count;
            double variance = (sum_squared / count) - (mean * mean);

            if (variance > 64.0) {
              is_homogeneous = false;
              break;
            }
          }
        }

        return is_homogeneous;
      };

  auto quad_tree_segmented_img = raw_img;
  auto quad_tree = Segmentation::SegmentationByQuadTree::build_quad_tree(
      quad_tree_segmented_img, func,
      {0, seed_segmented_img.header.infoHeader.width - 1, 0,
       seed_segmented_img.header.infoHeader.height - 1});
  auto leaf_boxes =
      Segmentation::SegmentationByQuadTree::get_leaf_boxes(quad_tree);
  for (auto box : leaf_boxes) {
    Plot::draw_box(quad_tree_segmented_img, box.l, box.r, box.t, box.b);
  }

  std::ofstream quad_tree_segmented_img_before_merge_file(
      "output/quad_tree_segmented_img.bmp", std::ios::binary);
  BmpImage::write_bmp(quad_tree_segmented_img_before_merge_file,
                      quad_tree_segmented_img);
}

void task7(std::string path) {
  std::ifstream in_file(path, std::ios::binary);
  auto raw_img = BmpImage::read_bmp(in_file);

  auto sobel_filtered_image =
      Convolution::apply_kernel(raw_img, {{1, 2, 1}, {0, 0, 0}, {-1, -2, -1}});

  std::ofstream sobel_filtered_file("output/sobel_filtered.bmp",
                                    std::ios::binary);
  BmpImage::write_bmp(sobel_filtered_file, sobel_filtered_image);

  auto otsu_sobel_filtered_image = sobel_filtered_image;

  auto th_by_otsu_sobel_filtered_image =
      Segmentation::SegmentationByThreshold::auto_find_threshold_by_otsu(
          otsu_sobel_filtered_image);

  auto segmented_by_otsu_sobel_filtered_image =
      Segmentation::SegmentationByThreshold::segment_by_threshold(
          otsu_sobel_filtered_image, th_by_otsu_sobel_filtered_image);

  std::ofstream segmented_by_otsu_sobel_filtered_file(
      "output/segmented_by_otsu_sobel_filtered.bmp", std::ios::binary);
  BmpImage::write_bmp(segmented_by_otsu_sobel_filtered_file,
                      segmented_by_otsu_sobel_filtered_image);

  // Prewitt

  auto prewitt_filtered_image =
      Convolution::apply_kernel(raw_img, {{1, 1, 1}, {0, 0, 0}, {-1, -1, -1}});

  std::ofstream prewitt_filtered_file("output/prewitt_filtered.bmp",
                                      std::ios::binary);
  BmpImage::write_bmp(prewitt_filtered_file, prewitt_filtered_image);

  auto th_by_otsu_prewitt_filtered_image =
      Segmentation::SegmentationByThreshold::auto_find_threshold_by_otsu(
          prewitt_filtered_image);

  auto segmented_by_otsu_prewitt_filtered_image =
      Segmentation::SegmentationByThreshold::segment_by_threshold(
          prewitt_filtered_image, th_by_otsu_prewitt_filtered_image);

  std::ofstream segmented_by_otsu_prewitt_filtered_file(
      "output/segmented_by_otsu_prewitt_filtered.bmp", std::ios::binary);
  BmpImage::write_bmp(segmented_by_otsu_prewitt_filtered_file,
                      segmented_by_otsu_prewitt_filtered_image);

  // LOG
  auto log_filtered_image =
      Convolution::apply_kernel(raw_img, {{0, 0, -1, 0, 0},
                                          {0, -1, -2, -1, 0},
                                          {-1, -2, 16, -2, -1},
                                          {0, -1, -2, -1, 0},
                                          {0, 0, -1, 0, 0}});

  std::ofstream log_filtered_file("output/log_filtered.bmp", std::ios::binary);
  BmpImage::write_bmp(log_filtered_file, log_filtered_image);

  auto th_by_otsu_log_filtered_image =
      Segmentation::SegmentationByThreshold::auto_find_threshold_by_otsu(
          log_filtered_image);

  auto segmented_by_otsu_log_filtered_image =
      Segmentation::SegmentationByThreshold::segment_by_threshold(
          log_filtered_image, th_by_otsu_log_filtered_image);

  std::ofstream segmented_by_otsu_log_filtered_file(
      "output/segmented_by_otsu_log_filtered.bmp", std::ios::binary);
  BmpImage::write_bmp(segmented_by_otsu_log_filtered_file,
                      segmented_by_otsu_log_filtered_image);
}

void task8(std::string path) {
  std::ifstream in_file(path, std::ios::binary);
  auto raw_img = BmpImage::read_bmp(in_file);

  auto hough_data = raw_img.get_channel([&](BmpImage::BmpPixel pixel) {
    return static_cast<double>(pixel.gray()) / 256;
  });

  auto hough_param = Hough::HoughLineParam{};
  auto hough_transformed = Hough::hough_linear_transform(
      hough_data.interpret(raw_img.header.infoHeader.height,
                           raw_img.header.infoHeader.width),
      hough_param);
  auto img = Hough::plot(hough_transformed);
  img.regenerate_header();

  std::ofstream hough_file("output/hough.bmp", std::ios::binary);
  BmpImage::write_bmp(hough_file, img);

  auto lines = Hough::get_lines(hough_transformed, hough_param);

  Hough::draw_lines(lines, raw_img);

  std::ofstream lines_file("output/lines.bmp", std::ios::binary);
  BmpImage::write_bmp(lines_file, raw_img);
}

void task9(std::string path) {
  std::ifstream in_file(path, std::ios::binary);
  auto raw_img = BmpImage::read_bmp(in_file);

  raw_img =
      Segmentation::SegmentationByThreshold::segment_by_threshold(raw_img, 64);

  std::ofstream segmented_img_file("output/segmented.bmp", std::ios::binary);
  BmpImage::write_bmp(segmented_img_file, raw_img);

  auto split = Segmentation::SegmentationByGrowth::split_region(raw_img);
  int c = 0;
  for (auto group : split) {
    c = (c + 1) % random_colors.size();
    Plot::draw_points(raw_img, group, random_colors[c]);
  }

  std::ofstream split_file("output/split.bmp", std::ios::binary);
  BmpImage::write_bmp(split_file, raw_img);
}

void task10(std::string path) {
  std::ifstream in_file(path, std::ios::binary);
  auto raw_img = BmpImage::read_bmp(in_file);

  raw_img =
      Segmentation::SegmentationByThreshold::segment_by_threshold(raw_img, 64);

  std::ofstream segmented_img_file("output/segmented.bmp", std::ios::binary);
  BmpImage::write_bmp(segmented_img_file, raw_img);

  auto split = Segmentation::SegmentationByGrowth::get_borders(
      raw_img, {255, 255, 255, 255}, {0, 0, 0, 255});
  auto canvas = Plot::generate_blank_canvas(raw_img.header.infoHeader.width,
                                            raw_img.header.infoHeader.height);
  for (auto group : split) {
    Plot::draw_points(canvas, group, {0, 0, 0, 255});
  }

  canvas.regenerate_header();
  std::ofstream split_file("output/split.bmp", std::ios::binary);
  BmpImage::write_bmp(split_file, canvas);
}

void task12(std::string path) {
  std::ifstream in_file(path, std::ios::binary);
  auto raw_img = BmpImage::read_bmp(in_file);

  auto scale_channel = raw_img.get_channel([&](BmpImage::BmpPixel pixel) {
    return std::clamp<double>(
        static_cast<double>(pixel.blue - (pixel.red + pixel.green) / 2), 0,
        256);
  });
  auto scaled_img = raw_img;
  scaled_img.image.data.foreach ([&](BmpImage::BmpPixel &pixel, size_t idx) {
    pixel = BmpImage::BmpPixel(pixel.red * scale_channel.data[idx] / 256,
                               pixel.green * scale_channel.data[idx] / 256,
                               pixel.blue * scale_channel.data[idx] / 256,
                               pixel.alpha);
    pixel = BmpImage::BmpPixel(pixel.gray(), pixel.gray(), pixel.gray(),
                               pixel.alpha);
  });

  std::ofstream scale_channel_file("output/scaled_img.bmp", std::ios::binary);
  BmpImage::write_bmp(scale_channel_file, scaled_img);

  // LOG
  auto mid_filtered_image =
      Convolution::apply_mid_value_kernel(scaled_img, 5, 1);
  auto log_filtered_image =
      Convolution::apply_kernel(mid_filtered_image, {{0, 0, -1, 0, 0},
                                                     {0, -1, -2, -1, 0},
                                                     {-1, -2, 16, -2, -1},
                                                     {0, -1, -2, -1, 0},
                                                     {0, 0, -1, 0, 0}});

  std::ofstream log_filtered_file("output/log_filtered.bmp", std::ios::binary);
  BmpImage::write_bmp(log_filtered_file, log_filtered_image);
  auto th_by_otsu_log_filtered_image =
      Segmentation::SegmentationByThreshold::auto_find_threshold_by_otsu(
          log_filtered_image);

  auto segmented_by_otsu_log_filtered_image =
      Segmentation::SegmentationByThreshold::segment_by_threshold(
          log_filtered_image, th_by_otsu_log_filtered_image);

  std::ofstream segmented_by_otsu_log_filtered_image_file(
      "output/segmented_by_otsu_log_filtered_image.bmp", std::ios::binary);
  BmpImage::write_bmp(segmented_by_otsu_log_filtered_image_file,
                      segmented_by_otsu_log_filtered_image);

  // Hough

  auto hough_data = segmented_by_otsu_log_filtered_image.get_channel(
      [](BmpImage::BmpPixel pixel) { return pixel.gray(); });

  auto hough_param = Hough::HoughLineParam{
      .theta_steps = 360,
  };
  auto hough_transformed = Hough::hough_linear_transform(
      hough_data.interpret(scaled_img.header.infoHeader.height,
                           scaled_img.header.infoHeader.width),
      hough_param, true);
  auto img = Hough::plot(hough_transformed);
  img.regenerate_header();

  std::ofstream hough_file("output/hough.bmp", std::ios::binary);
  BmpImage::write_bmp(hough_file, img);

  auto raw_lines =
      Hough::get_lines_bfs(hough_transformed, hough_param, 1, -1, 0.2, true);
  Hough::draw_lines(raw_lines, segmented_by_otsu_log_filtered_image);

  std::ofstream hough_lines_file("output/hough_lines.bmp", std::ios::binary);
  BmpImage::write_bmp(hough_lines_file, segmented_by_otsu_log_filtered_image);

  auto intersects = Hough::all_intersects(raw_lines);
  auto closing_hull = Hough::hull(intersects);

  auto hull_image = raw_img;
  for (int i = 0; i < closing_hull.size(); i++) {
    auto [x, y] = closing_hull[i];
    auto [x2, y2] = closing_hull[(i + 1) % closing_hull.size()];
    Plot::draw_line(hull_image, y, x, y2, x2);
  }

  std::ofstream closing_hull_file("output/closing_hull.bmp", std::ios::binary);
  BmpImage::write_bmp(closing_hull_file, hull_image);

  auto boxed_area_r = -hull_image.header.infoHeader.width;
  auto boxed_area_l = hull_image.header.infoHeader.width;
  auto boxed_area_t = -hull_image.header.infoHeader.height;
  auto boxed_area_b = hull_image.header.infoHeader.height;
  for (auto [y, x] : closing_hull) {
    boxed_area_r = std::max(boxed_area_r, x);
    boxed_area_l = std::min(boxed_area_l, x);
    boxed_area_t = std::max(boxed_area_t, y);
    boxed_area_b = std::min(boxed_area_b, y);
  }

  auto boxed_area_only = Plot::generate_blank_canvas(
      boxed_area_r - boxed_area_l, boxed_area_t - boxed_area_b);
  for (int i = boxed_area_l; i < boxed_area_r; i++) {
    for (int j = boxed_area_b; j < boxed_area_t; j++) {
      boxed_area_only.image.data
          .data[(j - boxed_area_b) * (boxed_area_r - boxed_area_l) +
                (i - boxed_area_l)] =
          raw_img.image.data.data[j * raw_img.header.infoHeader.width + i];
    }
  }

  boxed_area_only.image.data.foreach ([&](BmpImage::BmpPixel &pxl, size_t idx) {
    auto v = (pxl.red + pxl.green) / 2;
    pxl = BmpImage::BmpPixel(v, v, v, 255);
  });

  int max_val = 0;
  boxed_area_only.image.data.foreach_sync(
      [&](BmpImage::BmpPixel &pxl, size_t idx) {
        max_val = std::max(max_val, static_cast<int>(pxl.gray()));
      });

  std::ofstream boxed_area_only_file("output/boxed_area_only.bmp",
                                     std::ios::binary);
  BmpImage::write_bmp(boxed_area_only_file, boxed_area_only);

  auto th_by_otsu_boxed =
      Segmentation::SegmentationByThreshold::auto_find_threshold_by_otsu(
          boxed_area_only);

  auto segmented_by_otsu_boxed =
      Segmentation::SegmentationByThreshold::segment_by_threshold(
          boxed_area_only, (max_val + 2 * th_by_otsu_boxed) / 3);

  std::ofstream segmented_by_otsu_boxed_file(
      "output/segmented_by_otsu_boxed.bmp", std::ios::binary);
  BmpImage::write_bmp(segmented_by_otsu_boxed_file, segmented_by_otsu_boxed);

  int tolerance = segmented_by_otsu_boxed.header.infoHeader.height / 32;
  int tolerance_keep = 2;

  std::vector<int> split_at;
  bool is_peak = false;
  int tolerance_counter = 0;

  for (int i = 0; i < segmented_by_otsu_boxed.header.infoHeader.width; i++) {
    int count = 0;
    for (int j = 0; j < segmented_by_otsu_boxed.header.infoHeader.height; j++) {
      if (segmented_by_otsu_boxed.image.data
              .data[j * segmented_by_otsu_boxed.header.infoHeader.width + i]
              .gray() > 1) {
        count++;
      }
    }

    if (count > tolerance) {
      if (!is_peak) {
        split_at.push_back(i);
      }
      is_peak = true;
      tolerance_counter = 0;
    } else {
      if (is_peak) {
        tolerance_counter++;
        if (tolerance_counter > tolerance_keep) {
          split_at.push_back(i);
          is_peak = false;
        }
      }
    }
  }

  int j = 0;
  bool drawing_flag = true;
  for (int i = 0; i < segmented_by_otsu_boxed.header.infoHeader.width; i++) {
    if (i == split_at[j]) {
      drawing_flag = !drawing_flag;
      ++j;
    }
    if (drawing_flag) {
      Plot::draw_line(segmented_by_otsu_boxed, i, 0, i,
                      segmented_by_otsu_boxed.header.infoHeader.height - 1);
    }
  }

  std::ofstream split_at_file("output/split_at.bmp", std::ios::binary);
  BmpImage::write_bmp(split_at_file, segmented_by_otsu_boxed);
}

void task13(std::string path) {
  std::ifstream in_file(path, std::ios::binary);
  auto raw_img = BmpImage::read_bmp(in_file);

  auto fft_img = raw_img;
  auto gray = fft_img
                  .get_channel([&](BmpImage::BmpPixel pixel) {
                    return (pixel.red + pixel.green) / 2;
                  })
                  .interpret(fft_img.header.infoHeader.height,
                             fft_img.header.infoHeader.width);
  Frequency::pad(gray);
  std::ofstream fft_img_gray("output/fft_img_gray.bmp", std::ios::binary);
  auto gray_img = Frequency::plot(gray);
  gray_img.regenerate_header();
  BmpImage::write_bmp(fft_img_gray, gray_img);

  auto fft_transformed = Frequency::fft(gray);

  Frequency::cutoff_freq(fft_transformed, 100);
  auto [mag, phase] = Frequency::polar_transform(fft_transformed);
  std::ofstream fft_mag("output/fft_mag.bmp", std::ios::binary);
  auto fft_mag_img = Frequency::plot(mag);
  fft_mag_img.regenerate_header();
  BmpImage::write_bmp(fft_mag, fft_mag_img);

  std::ofstream fft_phase("output/fft_phase.bmp", std::ios::binary);
  auto fft_phase_img = Frequency::plot(phase);
  fft_phase_img.regenerate_header();
  BmpImage::write_bmp(fft_phase, fft_phase_img);

  auto ifft_transformed = Frequency::ifft(fft_transformed);
  auto ifft_img = Frequency::plot(ifft_transformed);
  std::ofstream ifft_img_file("output/ifft_img.bmp", std::ios::binary);
  BmpImage::write_bmp(ifft_img_file, ifft_img);
}

void printDivider() {
  std::cout << CYAN << "┌────────────────────────────────────────────┐" << RESET
            << std::endl;
}

void printMenu() {
  printDivider();
  std::cout << BOLD << CYAN << "│                📊 图像处理菜单             │"
            << RESET << std::endl;
  std::cout << CYAN << "├────────────────────────────────────────────┤" << RESET
            << std::endl;
  std::cout << GREEN << "│ 1. ➤ 通道分离                              │"
            << RESET << std::endl;
  std::cout << GREEN << "│ 2. ➤ 直方图处理                            │"
            << RESET << std::endl;
  std::cout << GREEN << "│ 3. ➤ 空间域滤波                            │"
            << RESET << std::endl;
  std::cout << GREEN << "│ 4. ➤ 线性变换                              │"
            << RESET << std::endl;
  std::cout << GREEN << "│ 5. ➤ 阈值分割                              │"
            << RESET << std::endl;
  std::cout << GREEN << "│ 6. ➤ 基于区域的分割                        │"
            << RESET << std::endl;
  std::cout << GREEN << "│ 7. ➤ 边缘检测                              │"
            << RESET << std::endl;
  std::cout << GREEN << "│ 8. ➤ Hough 变换                            │"
            << RESET << std::endl;
  std::cout << GREEN << "│ 9. ➤ 区域标记                              │"
            << RESET << std::endl;
  std::cout << GREEN << "│10. ➤ 轮廓提取                              │"
            << RESET << std::endl;
  std::cout << GREEN << "│12. ➤ 车牌提取                              │"
            << RESET << std::endl;
  std::cout << RED << "│13. ➤ 清空输出文件夹                        │" << RESET
            << std::endl;
  std::cout << RED << "│ 0. ➤ 退出程序                              │" << RESET
            << std::endl;
  std::cout << CYAN << "└────────────────────────────────────────────┘" << RESET
            << std::endl;
}

int getUserChoice() {
  int choice;
  while (true) {
    std::cout << YELLOW << "➤ 请输入任务编号 (0-13): " << RESET;
    std::cin >> choice;

    if (std::cin.fail() || choice < 0 ||
        (choice > 10 && choice != 12 && choice != 13)) {
      std::cin.clear();
      std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
      std::cout << RED << "✘ 无效输入，请输入有效的任务编号！" << RESET
                << std::endl;
    } else {
      std::cout << GREEN << "✔ 输入有效！您选择了任务编号 " << choice << RESET
                << std::endl;
      return choice;
    }
  }
}

bool getBatchMode() {
  std::cout << YELLOW << "是否批量处理? (y/n): " << RESET;
  std::string answer;
  while (true) {
    std::cin >> answer;
    if (answer == "y" || answer == "Y") {
      return true;
    } else if (answer == "n" || answer == "N") {
      return false;
    } else {
      std::cout << RED << "✘ 无效输入，请输入 'y' 或 'n'!" << RESET
                << std::endl;
    }
  }
}

std::string getPath(const std::string &prompt) {
  std::string path;
  std::cout << YELLOW << "📂 " << prompt << RESET;
  std::cin >> path;
  return path;
}

void showProgressBar(int current, int total) {
  const int barWidth = 50;
  float progress = (float)current / total;
  int pos = barWidth * progress;

  std::cout << "\r\033[K";
  std::cout << CYAN << "[";
  for (int i = 0; i < barWidth; ++i) {
    if (i < pos)
      std::cout << "█";
    else
      std::cout << "░";
  }
  std::cout << "] " << int(progress * 100.0) << " % (" << current << "/"
            << total << ")" << RESET;
  std::cout.flush();
}
void process_task(const std::string &path, int choice) {
  std::ifstream in_file(path, std::ios::binary);
  auto raw_img = BmpImage::read_bmp(in_file);
  in_file.close();
  std::cout << "原始图像预览" << std::endl;
  print_image(raw_img);
  std::cout << "原始图像信息" << std::endl;
  raw_img.pretty_print_info();

  switch (choice) {
  case 1:
    task1(path);
    break;
  case 2:
    task2(path);
    break;
  case 3:
    std::cout << "输入 kernel 大小（必须为奇数）" << std::endl;
    int kernel_size;
    std::cin >> kernel_size;
    task3_with_parameters(path, kernel_size);
    break;
  case 4:
    std::cout << "输入缩放因子" << std::endl;
    double scale_factor;
    std::cin >> scale_factor;
    std::cout << "输入旋转角度(弧度)" << std::endl;
    double rotation_angle;
    std::cin >> rotation_angle;
    std::cout << "输入 x 方向平移距离" << std::endl;
    double translation_x;
    std::cin >> translation_x;
    std::cout << "输入 y 方向平移距离" << std::endl;
    double translation_y;
    std::cin >> translation_y;
    task4_with_parameters(path, scale_factor, translation_x, translation_y,
                          rotation_angle);
    break;
  case 5:
    std::cout << "输入人工阈值" << std::endl;
    int threshold;
    std::cin >> threshold;
    task5_with_parameters(path, threshold);
    break;
  case 6:
    task6(path);
    break;
  case 7:
    task7(path);
    break;
  case 8:
    task8(path);
    break;
  case 9:
    task9(path);
    break;
  case 10:
    task10(path);
    break;
  case 12:
    task12(path);
    break;
  default:
    break;
  }
}

void moveProcessedFiles(const std::string &path,
                        const std::string &output_dir) {
  std::string file_name_without_ext =
      std::filesystem::path(path).filename().string();
  file_name_without_ext.erase(file_name_without_ext.find(".bmp"));

  std::string new_dir = output_dir + "/" + file_name_without_ext;
  if (!std::filesystem::exists(new_dir)) {
    std::filesystem::create_directory(new_dir);
  }

  for (const auto &entry : std::filesystem::directory_iterator(output_dir)) {
    if (entry.path().filename().string().find(".bmp") != std::string::npos) {
      std::string new_path = new_dir + "/" + entry.path().filename().string();
      std::filesystem::copy(entry.path(), new_path);
      std::filesystem::remove(entry.path());
    }
  }
}

void processBatchTask(const std::vector<std::string> &files,
                      std::function<void(std::string)> task) {
  int total = files.size();
  for (int i = 0; i < total; ++i) {
    std::cout << std::endl;
    std::cout << BLUE << "正在处理文件 " << (i + 1) << "/" << total << ": "
              << files[i] << RESET << std::endl;
    task(files[i]);
    showProgressBar(i + 1, total);
    moveProcessedFiles(files[i], "output");
  }
  std::cout << std::endl;
  std::cout << GREEN << "✔ 批量处理完成！" << RESET << std::endl;
}

void task() {
  while (true) {
    printMenu();
    int choice = getUserChoice();

    if (choice == 0) {
      std::cout << GREEN << "程序已退出！" << RESET << std::endl;
      break;
    }
    if (choice == 13) {
      std::cout << RED << "确认要清空输出文件夹 (y/n) ?" << RESET << std::endl;
      std::string answer;
      while (true) {
        std::cin >> answer;
        if (answer == "y" || answer == "Y") {
          std::filesystem::remove_all("output");
          std::filesystem::create_directory("output");
          std::cout << GREEN << "✔ 输出文件夹已清空！" << RESET << std::endl;
          break;
        } else if (answer == "n" || answer == "N") {
          break;
        } else {
          std::cout << RED << "✘ 无效输入，请输入 'y' 或 'n'!" << RESET
                    << std::endl;
        }
      }
      continue;
    }

    bool is_batch = getBatchMode();
    if (is_batch) {
      std::string folder_path = getPath("请输入文件夹路径: ");
      std::vector<std::string> files;
      for (const auto &entry :
           std::filesystem::directory_iterator(folder_path)) {
        if (entry.path().filename().string().find(".bmp") !=
            std::string::npos) {
          files.push_back(entry.path().string());
        }
      }

      if (files.empty()) {
        std::cout << RED << "✘ 文件夹中没有找到 .bmp 文件，请重试！" << RESET
                  << std::endl;
        continue;
      }

      std::function<void(std::string)> task;
      switch (choice) {
      case 1:
        task = task1;
        break;
      case 2:
        task = task2;
        break;
      case 3:
        task = task3;
        break;
      case 4:
        task = task4;
        break;
      case 5:
        task = task5;
        break;
      case 6:
        task = task6;
        break;
      case 7:
        task = task7;
        break;
      case 8:
        task = task8;
        break;
      case 9:
        task = task9;
        break;
      case 10:
        task = task10;
        break;
      case 12:
        task = task12;
        break;
      default:
        break;
      }

      processBatchTask(files, task);
    } else {
      std::string file_path = getPath("请输入文件路径: ");
      process_task(file_path, choice);
    }
  }
}

int main() {
  task();
  return 0;
}