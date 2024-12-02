#include "lib/bmp_image.hxx"
#include "lib/convolution.hxx"
#include "lib/linalg.hxx"
#include "lib/linear_transform.hxx"
#include "lib/numeric_array.hxx"
#include "lib/plot.hxx"
#include "lib/terminal_print.hxx"
#include "lib/segmentation.hxx"

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
  auto avg_filtered_image = Convolution::apply_kernel(raw_img,
                            {
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
  auto mid_filtered_image = Convolution::apply_mid_value_kernel(raw_img, 5, (5 * 5) / 2);
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
  auto perspective_img =
      LinearTransform::linear_transform(raw_img, Linalg::LinearTransformMatrix()
                                                     .perspective_by_points(
                                                        {
                                                          std::make_tuple(0., 0.),
                                                          std::make_tuple(0., raw_img.header.infoHeader.height),
                                                          std::make_tuple(582., 582.),
                                                          std::make_tuple(582., raw_img.header.infoHeader.height - 582.)
                                                        },
                                                        {
                                                          std::make_tuple(0., 0.),
                                                          std::make_tuple(0., raw_img.header.infoHeader.height),
                                                          std::make_tuple(raw_img.header.infoHeader.width, raw_img.header.infoHeader.height),
                                                          std::make_tuple(raw_img.header.infoHeader.width, 0.)
                                                        }
                                                     )
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
  auto segmented_img = Segmentation::segment_by_threshold(raw_img, 128);
  std::cout << "Segmented Image:" << std::endl;
  print_image(segmented_img);
  std::ofstream segmented_img_file("output/segmented.bmp", std::ios::binary);
  BmpImage::write_bmp(segmented_img_file, segmented_img);
  BmpImage::BmpImage segmented_img_histogram = Plot::generate_gray_scale_histogram(raw_img);
  Plot::draw_line(segmented_img_histogram, 128, 256, 128, 0);
  std::ofstream segmented_img_histogram_file("output/segmented_histogram.bmp", std::ios::binary);
  BmpImage::write_bmp(segmented_img_histogram_file, segmented_img_histogram);
  auto th_by_iteration = Segmentation::auto_find_threshold_by_iteration(raw_img);
  std::cout << "Threshold by iteration: " << th_by_iteration << std::endl;
  auto segmented_img_by_iteration = Segmentation::segment_by_threshold(raw_img, th_by_iteration);
  std::cout << "Segmented Image by iteration:" << std::endl;
  print_image(segmented_img_by_iteration);
  std::ofstream segmented_img_by_iteration_file("output/segmented_img_by_iteration.bmp", std::ios::binary);
  BmpImage::write_bmp(segmented_img_by_iteration_file, segmented_img_by_iteration);
  auto segmented_by_iteration_histogram = Plot::generate_gray_scale_histogram(raw_img);
  Plot::draw_line(segmented_by_iteration_histogram, th_by_iteration, 256, th_by_iteration, 0);
  std::ofstream segmented_by_iteration_histogram_file("output/segmented_by_iteration_histogram.bmp", std::ios::binary);
  BmpImage::write_bmp(segmented_by_iteration_histogram_file, segmented_by_iteration_histogram);
  auto th_by_otsu = Segmentation::auto_find_threshold_by_otsu(raw_img);
  std::cout << "Threshold by otsu: " << th_by_otsu << std::endl;
  auto segmented_img_by_otsu = Segmentation::segment_by_threshold(raw_img, th_by_otsu);
  std::cout << "Segmented Image by otsu:" << std::endl;
  print_image(segmented_img_by_otsu);
  std::ofstream segmented_img_by_otsu_file("output/segmented_img_by_otsu.bmp", std::ios::binary);
  BmpImage::write_bmp(segmented_img_by_otsu_file, segmented_img_by_otsu);
  auto segmented_by_otsu_histogram = Plot::generate_gray_scale_histogram(raw_img);
  Plot::draw_line(segmented_by_otsu_histogram, th_by_otsu, 256, th_by_otsu, 0);
  std::ofstream segmented_by_otsu_histogram_file("output/segmented_by_otsu_histogram.bmp", std::ios::binary);
  BmpImage::write_bmp(segmented_by_otsu_histogram_file, segmented_by_otsu_histogram);
}

int main() {
  // task1();
  // task3();
  // task3();
  task4();
  // task5();
  return 0;
}
