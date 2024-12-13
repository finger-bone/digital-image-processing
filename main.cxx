#include "lib/bmp_image.hxx"
#include "lib/convolution.hxx"
#include "lib/frequency.hxx"
#include "lib/linalg.hxx"
#include "lib/linear_transform.hxx"
#include "lib/numeric_array.hxx"
#include "lib/plot.hxx"
#include "lib/segmentation.hxx"
#include "lib/terminal_print.hxx"
#include "lib/hough.hxx"

#include <cmath>
#include <fstream>

void task1() {
  std::cout << "Input the path of the image: " << std::endl;
  std::string path;
  std::cin >> path;
  std::ifstream in_file(path, std::ios::binary);
  auto raw_img = BmpImage::read_bmp(in_file);
  std::cout << "Input Image:" << std::endl;
  print_image(raw_img);
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
  std::cout << "Gray Image:" << std::endl;
  print_image(gray_img);
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
  std::cout << "Inverted Image:" << std::endl;
  print_image(inverted_grey_img);
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
  std::cout << "R Channel Image:" << std::endl;
  print_image(r_img);
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
  std::cout << "G Channel Image:" << std::endl;
  print_image(g_img);
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
  std::cout << "B Channel Image:" << std::endl;
  print_image(b_img);
  std::ofstream b_img_file("output/b_img.bmp", std::ios::binary);
  BmpImage::write_bmp(b_img_file, b_img);
}

void task2() {
  std::cout << "Input the path of the image: " << std::endl;
  std::string path;
  std::cin >> path;
  std::ifstream in_file(path, std::ios::binary);
  auto raw_img = BmpImage::read_bmp(in_file);
  std::cout << "Input Image:" << std::endl;
  print_image(raw_img);
  auto hist = Plot::generate_gray_scale_histogram(raw_img);
  print_image(hist);
  std::ofstream hist_file("output/hist_before.bmp", std::ios::binary);
  BmpImage::write_bmp(hist_file, hist);
  auto balanced_img = BmpImage::gray_balanced_image(raw_img);
  std::cout << "Balanced Image:" << std::endl;
  print_image(balanced_img);
  std::ofstream balanced_img_file("output/balanced_img.bmp", std::ios::binary);
  BmpImage::write_bmp(balanced_img_file, balanced_img);
  auto balanced_hist = Plot::generate_gray_scale_histogram(balanced_img);
  print_image(balanced_hist);
  std::ofstream balanced_hist_file("output/hist_after.bmp", std::ios::binary);
  BmpImage::write_bmp(balanced_hist_file, balanced_hist);
}

void task3() {
  std::cout << "Input the path of the image: " << std::endl;
  std::string path;
  std::cin >> path;
  std::ifstream in_file(path, std::ios::binary);
  auto raw_img = BmpImage::read_bmp(in_file);
  std::cout << "Input Image:" << std::endl;
  print_image(raw_img);
  auto value = 1.0 / 25;
  auto avg_filtered_image = Convolution::apply_kernel(
      raw_img, {
                   {value, value, value, value, value},
                   {value, value, value, value, value},
                   {value, value, value, value, value},
                   {value, value, value, value, value},
                   {value, value, value, value, value},
               });
  std::cout << "average filtered" << std::endl;
  print_image(avg_filtered_image);
  std::ofstream avg_filtered_file("output/avg_filtered.bmp", std::ios::binary);
  BmpImage::write_bmp(avg_filtered_file, avg_filtered_image);
  auto mid_filtered_image =
      Convolution::apply_mid_value_kernel(raw_img, 5, (5 * 5) / 2);
  print_image(mid_filtered_image);
  std::ofstream mid_filtered_file("output/mid_filtered.bmp", std::ios::binary);
  BmpImage::write_bmp(mid_filtered_file, mid_filtered_image);
}

void task4() {
  std::cout << "Input the path of the image: " << std::endl;
  std::string path;
  std::cin >> path;
  std::ifstream in_file(path, std::ios::binary);
  auto raw_img = BmpImage::read_bmp(in_file);
  std::cout << "Input Image:" << std::endl;
  print_image(raw_img);
  raw_img.change_to_twenty_four_bit();
  auto scaled_img = LinearTransform::linear_transform(
      raw_img, Linalg::LinearTransformMatrix().scale(0.5, 0.5).take());
  std::cout << "Scaled Image:" << std::endl;
  print_image(scaled_img);
  std::ofstream scaled_img_file("output/scaled_img.bmp", std::ios::binary);
  BmpImage::write_bmp(scaled_img_file, scaled_img);

  auto rotated_img = LinearTransform::linear_transform(
      raw_img, Linalg::LinearTransformMatrix().rotate(3.14 / 4).take());
  std::ofstream rotated_img_file("output/rotated_img.bmp", std::ios::binary);
  BmpImage::write_bmp(rotated_img_file, rotated_img);
  std::cout << "Rotated Image:" << std::endl;
  print_image(rotated_img);

  auto translated_img = LinearTransform::linear_transform(
      raw_img, Linalg::LinearTransformMatrix().translate(100, 100).take());
  std::ofstream translated_img_file("output/translated_img.bmp",
                                    std::ios::binary);
  BmpImage::write_bmp(translated_img_file, translated_img);
  std::cout << "Translated Image:" << std::endl;
  print_image(translated_img);

  auto flipped_image = LinearTransform::linear_transform(
      raw_img, Linalg::LinearTransformMatrix()
                   .translate(-raw_img.header.infoHeader.width, 0)
                   .scale(-1, 1)
                   .take());
  std::ofstream flipped_img_file("output/flipped_img.bmp", std::ios::binary);
  BmpImage::write_bmp(flipped_img_file, flipped_image);
  std::cout << "Flipped Image:" << std::endl;
  print_image(flipped_image);

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
  std::cout << "Perspective Image:" << std::endl;
  print_image(perspective_img);

  auto combined =
      LinearTransform::linear_transform(raw_img, Linalg::LinearTransformMatrix()
                                                     .scale(0.5, 0.5)
                                                     .translate(10, -10)
                                                     .rotate(3.14 / 4)
                                                     .take());
  std::ofstream combined_img_file("output/combined_img.bmp", std::ios::binary);
  BmpImage::write_bmp(combined_img_file, combined);
  std::cout << "Combined Image:" << std::endl;
  print_image(combined);
}

void task5() {
  std::cout << "Input the path of the image: " << std::endl;
  std::string path;
  std::cin >> path;
  std::ifstream in_file(path, std::ios::binary);
  auto raw_img = BmpImage::read_bmp(in_file);
  std::cout << "Input Image:" << std::endl;
  print_image(raw_img);
  auto segmented_img =
      Segmentation::SegmentationByThreshold::segment_by_threshold(raw_img, 128);
  std::cout << "Segmented Image:" << std::endl;
  print_image(segmented_img);
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
  std::cout << "Threshold by iteration: " << th_by_iteration << std::endl;
  auto segmented_img_by_iteration =
      Segmentation::SegmentationByThreshold::segment_by_threshold(
          raw_img, th_by_iteration);
  std::cout << "Segmented Image by iteration:" << std::endl;
  print_image(segmented_img_by_iteration);
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
  std::cout << "Threshold by otsu: " << th_by_otsu << std::endl;
  auto segmented_img_by_otsu =
      Segmentation::SegmentationByThreshold::segment_by_threshold(raw_img,
                                                                  th_by_otsu);
  std::cout << "Segmented Image by otsu:" << std::endl;
  print_image(segmented_img_by_otsu);
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

void task6() {
  std::cout << "Input the path of the image: " << std::endl;
  std::string path;
  std::cin >> path;
  std::ifstream in_file(path, std::ios::binary);
  auto raw_img = BmpImage::read_bmp(in_file);
  raw_img.change_to_twenty_four_bit();

  auto seed_segmented_img = raw_img;
  std::cout << "Input Image:" << std::endl;
  print_image(seed_segmented_img);
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
  print_image(seed_segmented_img);
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
  print_image(quad_tree_segmented_img);
  std::ofstream quad_tree_segmented_img_before_merge_file(
      "output/quad_tree_segmented_img.bmp", std::ios::binary);
  BmpImage::write_bmp(quad_tree_segmented_img_before_merge_file,
                      quad_tree_segmented_img);
}

void task7() {
  std::cout << "Input the path of the image: " << std::endl;
  std::string path;
  std::cin >> path;
  std::ifstream in_file(path, std::ios::binary);
  auto raw_img = BmpImage::read_bmp(in_file);
  std::cout << "Input Image:" << std::endl;
  print_image(raw_img);
  auto sobel_filtered_image =
      Convolution::apply_kernel(raw_img, {{1, 2, 1}, {0, 0, 0}, {-1, -2, -1}});

  std::cout << "Sobel Filtered Image:" << std::endl;
  print_image(sobel_filtered_image);

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

  std::cout << "Segmented by Otsu Sobel Filtered Image:" << std::endl;
  print_image(segmented_by_otsu_sobel_filtered_image);

  std::ofstream segmented_by_otsu_sobel_filtered_file(
      "output/segmented_by_otsu_sobel_filtered.bmp", std::ios::binary);
  BmpImage::write_bmp(segmented_by_otsu_sobel_filtered_file,
                      segmented_by_otsu_sobel_filtered_image);

  // Prewitt

  auto prewitt_filtered_image =
      Convolution::apply_kernel(raw_img, {{1, 1, 1}, {0, 0, 0}, {-1, -1, -1}});

  std::cout << "Prewitt Filtered Image:" << std::endl;
  print_image(prewitt_filtered_image);

  std::ofstream prewitt_filtered_file("output/prewitt_filtered.bmp",
                                      std::ios::binary);
  BmpImage::write_bmp(prewitt_filtered_file, prewitt_filtered_image);

  auto th_by_otsu_prewitt_filtered_image =
      Segmentation::SegmentationByThreshold::auto_find_threshold_by_otsu(
          prewitt_filtered_image);

  auto segmented_by_otsu_prewitt_filtered_image =
      Segmentation::SegmentationByThreshold::segment_by_threshold(
          prewitt_filtered_image, th_by_otsu_prewitt_filtered_image);

  std::cout << "Segmented by Otsu Prewitt Filtered Image:" << std::endl;
  print_image(segmented_by_otsu_prewitt_filtered_image);

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

  std::cout << "LOG Filtered Image:" << std::endl;
  print_image(log_filtered_image);

  std::ofstream log_filtered_file("output/log_filtered.bmp", std::ios::binary);
  BmpImage::write_bmp(log_filtered_file, log_filtered_image);

  auto th_by_otsu_log_filtered_image =
      Segmentation::SegmentationByThreshold::auto_find_threshold_by_otsu(
          log_filtered_image);

  auto segmented_by_otsu_log_filtered_image =
      Segmentation::SegmentationByThreshold::segment_by_threshold(
          log_filtered_image, th_by_otsu_log_filtered_image);

  std::cout << "Segmented by Otsu LOG Filtered Image:" << std::endl;
  print_image(segmented_by_otsu_log_filtered_image);

  std::ofstream segmented_by_otsu_log_filtered_file(
      "output/segmented_by_otsu_log_filtered.bmp", std::ios::binary);
  BmpImage::write_bmp(segmented_by_otsu_log_filtered_file,
                      segmented_by_otsu_log_filtered_image);
}

void task8() {
  std::cout << "Input the path of the image: " << std::endl;
  std::string path;
  std::cin >> path;
  std::ifstream in_file(path, std::ios::binary);
  auto raw_img = BmpImage::read_bmp(in_file);
  std::cout << "Input Image:" << std::endl;
  print_image(raw_img);
  auto hough_data = raw_img.get_channel(
      [&](BmpImage::BmpPixel pixel) { 
        return static_cast<double>(pixel.gray()) / 256;
      }
    );

  auto hough_param = Hough::HoughLineParam{};
  auto hough_transformed = Hough::hough_linear_transform(hough_data.interpret(
      raw_img.header.infoHeader.height,
      raw_img.header.infoHeader.width
  ), hough_param);
  auto img = Hough::plot(hough_transformed);
  img.regenerate_header();

  std::ofstream hough_file("output/hough.bmp", std::ios::binary);
  BmpImage::write_bmp(hough_file, img);

  auto lines = Hough::get_lines(hough_transformed, hough_param);

  Hough::draw_lines(lines, raw_img);

  for(auto line : lines) {
    std::cout << std::get<0>(line) << " " << std::get<1>(line) << std::endl;
  }
  std::cout << lines.size() << std::endl;

  std::ofstream lines_file("output/lines.bmp", std::ios::binary);
  BmpImage::write_bmp(lines_file, raw_img);
}

void task12() {
  std::cout << "Input the path of the image: " << std::endl;
  std::string path;
  std::cin >> path;
  std::ifstream in_file(path, std::ios::binary);
  auto raw_img = BmpImage::read_bmp(in_file);
  std::cout << "Input Image:" << std::endl;
  print_image(raw_img);
  auto scale_channel = raw_img.get_channel(
      [&](BmpImage::BmpPixel pixel) {
        return std::clamp<double>(static_cast<double>(
          pixel.blue - (pixel.red + pixel.green) / 2
        ), 0, 256);
      }
  );
  auto scaled_img = raw_img;
  scaled_img.image.data.foreach([&](BmpImage::BmpPixel& pixel, size_t idx) {
    pixel = BmpImage::BmpPixel(
        pixel.red * scale_channel.data[idx] / 256,
        pixel.green * scale_channel.data[idx] / 256,
        pixel.blue * scale_channel.data[idx] / 256,
        pixel.alpha
    );
    pixel = BmpImage::BmpPixel(
      pixel.gray(),
      pixel.gray(),
      pixel.gray(),
      pixel.alpha
    );
  });

  std::ofstream scale_channel_file("output/scaled_img.bmp", std::ios::binary);
  BmpImage::write_bmp(scale_channel_file, scaled_img);

  // LOG 
  auto mid_filtered_image = Convolution::apply_mid_value_kernel(
      scaled_img, 5, 1
  );
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

  std::ofstream segmented_by_otsu_log_filtered_image_file("output/segmented_by_otsu_log_filtered_image.bmp", std::ios::binary);
  BmpImage::write_bmp(segmented_by_otsu_log_filtered_image_file, segmented_by_otsu_log_filtered_image);

  // Hough

  auto hough_data = segmented_by_otsu_log_filtered_image.get_channel(
      [](BmpImage::BmpPixel pixel) { 
        return pixel.gray();
      }
    );

  auto hough_param = Hough::HoughLineParam{
    .theta_steps = 360,
  };
  auto hough_transformed = Hough::hough_linear_transform(hough_data.interpret(
      scaled_img.header.infoHeader.height,
      scaled_img.header.infoHeader.width
  ), hough_param, true);
  auto img = Hough::plot(hough_transformed);
  img.regenerate_header();

  std::ofstream hough_file("output/hough.bmp", std::ios::binary);
  BmpImage::write_bmp(hough_file, img);

  auto raw_lines = Hough::get_lines_bfs(hough_transformed, hough_param, 1, -1, 0.2);
  std::cout << raw_lines.size() << std::endl;
  Hough::draw_lines(raw_lines, segmented_by_otsu_log_filtered_image);

  std::ofstream hough_lines_file("output/hough_lines.bmp", std::ios::binary);
  BmpImage::write_bmp(hough_lines_file, segmented_by_otsu_log_filtered_image);

  auto intersects = Hough::all_intersects(raw_lines);
  auto closing_hull = Hough::hull(intersects);

  std::cout << closing_hull.size() << std::endl;
  auto hull_image = raw_img;
  for(int i = 0; i < closing_hull.size(); i++) {
    auto [x, y] = closing_hull[i];
    auto [x2, y2] = closing_hull[(i + 1) % closing_hull.size()];
    std::cout << x << " " << y << " " << x2 << " " << y2 << std::endl;
    Plot::draw_line(hull_image, y, x, y2, x2);
  }

  std::ofstream closing_hull_file("output/closing_hull.bmp", std::ios::binary);
  BmpImage::write_bmp(closing_hull_file, hull_image);


  auto boxed_area_r = -hull_image.header.infoHeader.width;
  auto boxed_area_l = hull_image.header.infoHeader.width;
  auto boxed_area_t = -hull_image.header.infoHeader.height;
  auto boxed_area_b = hull_image.header.infoHeader.height;
  for(auto [y, x] : closing_hull) {
    boxed_area_r = std::max(boxed_area_r, x);
    boxed_area_l = std::min(boxed_area_l, x);
    boxed_area_t = std::max(boxed_area_t, y);
    boxed_area_b = std::min(boxed_area_b, y);
  }

  auto boxed_area_only = Plot::generate_blank_canvas(boxed_area_r - boxed_area_l, boxed_area_t - boxed_area_b);
  for(int i = boxed_area_l; i < boxed_area_r; i++) {
    for(int j = boxed_area_b; j < boxed_area_t; j++) {
      boxed_area_only.image.data.data[
          (j - boxed_area_b) * (boxed_area_r - boxed_area_l) + (i - boxed_area_l)
      ] = raw_img.image.data.data[
          j * raw_img.header.infoHeader.width + i
      ];
    }
  }

  boxed_area_only.image.data.foreach([&](BmpImage::BmpPixel &pxl, size_t idx) {
    auto v = (pxl.red + pxl.green) / 2;
    pxl = BmpImage::BmpPixel(v, v, v, 255);
  });

  std::ofstream boxed_area_only_file("output/boxed_area_only.bmp", std::ios::binary);
  BmpImage::write_bmp(boxed_area_only_file, boxed_area_only);

  auto th_by_otsu_boxed = Segmentation::SegmentationByThreshold::auto_find_threshold_by_otsu(
    boxed_area_only
  );

  auto segmented_by_otsu_boxed = Segmentation::SegmentationByThreshold::segment_by_threshold(
    boxed_area_only,
    th_by_otsu_boxed
  );

  std::ofstream segmented_by_otsu_boxed_file("output/segmented_by_otsu_boxed.bmp", std::ios::binary);
  BmpImage::write_bmp(segmented_by_otsu_boxed_file, segmented_by_otsu_boxed);
}

void task13() {
  std::cout << "Input the path of the image: " << std::endl;
  std::string path;
  std::cin >> path;
  std::ifstream in_file(path, std::ios::binary);
  auto raw_img = BmpImage::read_bmp(in_file);
  std::cout << "Input Image:" << std::endl;
  print_image(raw_img);
  // fast fourier transform

  auto fft_img = raw_img;
  auto gray =
      fft_img
          .get_channel([&](BmpImage::BmpPixel pixel) { return (pixel.red + pixel.green) / 2; })
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

int main() {
  // task1();
  // task3();
  // task3();
  // task4();
  // task5();
  // task6();
  // task7();
  // task8();
  task12();
  // task13();
  return 0;
}
