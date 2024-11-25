#include "lib/bmp_image.hxx"
#include "lib/convolution.hxx"
#include "lib/numeric_array.hxx"
#include "lib/plot.hxx"
#include "lib/terminal_print.hxx"
#include "lib/linear_transform.hxx"
#include "lib/linalg.hxx"

#include <fstream>
#include <cmath>

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
  auto avg_filtered_image = raw_img;
  auto value = 1.0 / 25;
  Convolution::apply_kernel(avg_filtered_image,
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
  auto mid_filtered_image = raw_img;
  Convolution::apply_mid_value_kernel(mid_filtered_image, 5, (5 * 5) / 2);
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
      raw_img,
      Linalg::LinearTransformMatrix().scale(0.5, 0.5).take()
  );
  std::cout << "Scaled Image:" << std::endl;
  print_image(scaled_img);
  std::ofstream scaled_img_file("output/scaled_img.bmp", std::ios::binary);
  BmpImage::write_bmp(scaled_img_file, scaled_img);

  auto rotated_img = LinearTransform::linear_transform(
      raw_img,
      Linalg::LinearTransformMatrix().rotate(3.14 / 4).take()
  );
  std::ofstream rotated_img_file("output/rotated_img.bmp", std::ios::binary);
  BmpImage::write_bmp(rotated_img_file, rotated_img);
  std::cout << "Rotated Image:" << std::endl;
  print_image(rotated_img);

  auto translated_img = LinearTransform::linear_transform(
      raw_img,
      Linalg::LinearTransformMatrix().translate(100, 100).take()
  );
  std::ofstream translated_img_file("output/translated_img.bmp", std::ios::binary);
  BmpImage::write_bmp(translated_img_file, translated_img);
  std::cout << "Translated Image:" << std::endl;
  print_image(translated_img);

  auto flipped_image = LinearTransform::linear_transform(
      raw_img,
      Linalg::LinearTransformMatrix()
      .translate(-raw_img.header.infoHeader.width, 0)
      .scale(-1, 1)
      .take()
  );
  std::ofstream flipped_img_file("output/flipped_img.bmp", std::ios::binary);
  BmpImage::write_bmp(flipped_img_file, flipped_image);
  std::cout << "Flipped Image:" << std::endl;
  print_image(flipped_image);

  auto half_height = static_cast<double>(raw_img.header.infoHeader.height) / 2;
  auto perspective_img = LinearTransform::linear_transform(
      raw_img,
      Linalg::LinearTransformMatrix()
      .translate(0, -half_height)
      .perspective(0.001, 0)
      .translate(0, half_height)
      .take()
  );
  std::ofstream perspective_img_file("output/perspective_img.bmp", std::ios::binary);
  BmpImage::write_bmp(perspective_img_file, perspective_img);
  std::cout << "Perspective Image:" << std::endl;
  print_image(perspective_img);

  auto combined = LinearTransform::linear_transform(
      raw_img,
      Linalg::LinearTransformMatrix().scale(0.5, 0.5).translate(10, -10).rotate(3.14 / 4).take()
  );
  std::ofstream combined_img_file("output/combined_img.bmp", std::ios::binary);
  BmpImage::write_bmp(combined_img_file, combined);
  std::cout << "Combined Image:" << std::endl;
  print_image(combined);
}

int main() {
  // task1();
  // task3();
  task4();
  // std::ifstream in_file("input/lena.bmp", std::ios::binary);
  // auto raw_img = BmpImage::read_bmp(in_file);
  // auto p = Plot::generate_gray_scale_histogram(raw_img, 256, 1024);
  // print_image(p);
  // std::ofstream f("output/bar_plot.bmp");
  // BmpImage::write_bmp(f, p);
  // auto img = Plot::generate_blank_canvas(10, 10);
  // std::vector<int> values = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  // Plot::bar_plot(img, values, 10);
  // print_image(img);
  // img.regenerate_header();
  // std::ofstream file("bar_plot.bmp", std::ios::binary);
  // BmpImage::write_bmp(file, img);
  // auto img = Image::load_bmp("input/rgb3.bmp");

  // img.data.foreach([](Image::BmpRgbPixel pixel) {
  //     return Image::BmpRgbPixel{
  //         .red = pixel.red,
  //         .green = pixel.green,
  //         .blue = 0,
  //     };
  // });

  // Image::save_bmp("rgb3_rg.bmp", img);
  return 0;
}
