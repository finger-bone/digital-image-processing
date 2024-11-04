#include "lib/numeric_array.h"
#include "lib/bmp_image.h"
#include "lib/terminal_print.h"

#include <fstream>

int main() {
    auto f = std::ifstream("input/rgb.bmp", std::ios::binary);
    auto p = Image::read_bmp(
        f
    );
    p.image.data.map_inplace([](Image::BmpPixel p) {
        return Image::BmpPixel{
            .red = p.red,
            .green = p.red,
            .blue = p.red,
            .alpha = 0,
        };
    });
    p.set_bbp(8);
    p.regenerate_palette();
    p.regenerate_header();
    auto fo = std::ofstream("output/rgb.bmp", std::ios::binary);
    print_image(p);
    Image::write_bmp(fo, p);
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
