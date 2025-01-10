// #ifndef IMAGE_PROCESSING_TERMINAL_PRINT_HXX
// #define IMAGE_PROCESSING_TERMINAL_PRINT_HXX

// #include "bmp_image.hxx"
// #include <algorithm>
// #include <cstdio>
// #include <sys/ioctl.h>
// #include <unistd.h>

// void print_image(const BmpImage::BmpImage &image) {
//   struct winsize w;
//   ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);

//   int term_height = w.ws_row;
//   // int term_width = w.ws_col;
//   int term_width = term_height * 3;

//   double chunk_width = static_cast<double>(image.image.size.width) /
//   term_width; double chunk_height =
//       static_cast<double>(image.image.size.height) / term_height;

//   for (int ty = 0; ty < term_height; ty++) {
//     for (int tx = 0; tx < term_width; tx++) {
//       // 计算块中心像素的坐标
//       int center_x = static_cast<int>((tx + 0.5) * chunk_width);
//       int center_y =
//           static_cast<int>((term_height - 1 - ty + 0.5) * chunk_height);

//       // 检查中心点是否在图像范围内
//       if (center_x >= 0 && center_x < image.image.size.width && center_y >= 0
//       &&
//           center_y < image.image.size.height) {

//         // 获取中心点像素颜色
//         auto &pixel =
//             image.image.data.data[center_y * image.image.size.width +
//             center_x];
//         int avg_red = pixel.red;
//         int avg_green = pixel.green;
//         int avg_blue = pixel.blue;

//         // 输出对应颜色的ANSI代码
//         printf("\033[48;2;%d;%d;%dm \033[0m", avg_red, avg_green, avg_blue);
//       } else {
//         // 出界时显示空格
//         printf(" ");
//       }
//     }
//     printf("\n");
//   }
// }

// #endif