#include "lib/bar_plot.hxx"
#include "lib/bmp_image.hxx"
#include "lib/numeric_array.hxx"
#include "lib/terminal_print.hxx"

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
  gray_img.image.data.map_inplace([](BmpImage::BmpPixel pixel) {
    auto gray = static_cast<uint8_t>(0.299 * pixel.red + 0.587 * pixel.green +
                                     0.114 * pixel.blue);
    return BmpImage::BmpPixel{
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
  inverted_grey_img.image.data.map_inplace([](BmpImage::BmpPixel pixel) {
    return BmpImage::BmpPixel{
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
  r_img.image.data.map_inplace([](BmpImage::BmpPixel pixel) {
    return BmpImage::BmpPixel{
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
  g_img.image.data.map_inplace([](BmpImage::BmpPixel pixel) {
    return BmpImage::BmpPixel{
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
  b_img.image.data.map_inplace([](BmpImage::BmpPixel pixel) {
    return BmpImage::BmpPixel{
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

int main() {
  // task1();
  std::ifstream in_file("input/lena.bmp", std::ios::binary);
  auto raw_img = BmpImage::read_bmp(in_file);
  auto p = BarPlot::generate_gray_scale_histogram(raw_img, 256, 1024);
  print_image(p);
  std::ofstream f("output/bar_plot.bmp");
  BmpImage::write_bmp(f, p);
  // auto img = BarPlot::generate_blank_canvas(10, 10);
  // std::vector<int> values = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  // BarPlot::bar_plot(img, values, 10);
  // print_image(img);
  // img.regenerate_header();
  // std::ofstream file("bar_plot.bmp", std::ios::binary);
  // BmpImage::write_bmp(file, img);
  // auto img = Image::load_bmp("input/rgb3.bmp");

  // img.data.map_inplace([](Image::BmpRgbPixel pixel) {
  //     return Image::BmpRgbPixel{
  //         .red = pixel.red,
  //         .green = pixel.green,
  //         .blue = 0,
  //     };
  // });

  // Image::save_bmp("rgb3_rg.bmp", img);
  return 0;
}
