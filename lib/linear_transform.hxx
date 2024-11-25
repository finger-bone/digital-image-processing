#ifndef IMAGE_PROCESSING_LINEAR_TRANSFORM_HXX
#define IMAGE_PROCESSING_LINEAR_TRANSFORM_HXX

#include "bmp_image.hxx"
#include "linalg.hxx"

#include <functional>

namespace LinearTransform {

    using interplate_function = std::function<
        BmpImage::BmpPixel(
            BmpImage::BmpPixel, BmpImage::BmpPixel, BmpImage::BmpPixel, BmpImage::BmpPixel,
            double, double
        )
    >;

    double f_part(double x) {
        return x - floor(x);
    }

    BmpImage::BmpPixel bilinear_interpolate(
        BmpImage::BmpPixel top_left, BmpImage::BmpPixel top_right,
        BmpImage::BmpPixel bottom_left, BmpImage::BmpPixel bottom_right,
        double x_from_top_left, double y_from_top_left
    ) {
        auto map_one_value = [&](
            uint8_t top_left, uint8_t top_right, uint8_t bottom_left, uint8_t bottom_right,
            double x_from_top_left, double y_from_top_left
        ) {
            return static_cast<uint8_t>(
                top_left * (1 - x_from_top_left) * (1 - y_from_top_left) +
                top_right * x_from_top_left * (1 - y_from_top_left) +
                bottom_left * (1 - x_from_top_left) * y_from_top_left +
                bottom_right * x_from_top_left * y_from_top_left
            );
        };
        return BmpImage::BmpPixel {
            .red = map_one_value(top_left.red, top_right.red, bottom_left.red, bottom_right.red, x_from_top_left, y_from_top_left),
            .green = map_one_value(top_left.green, top_right.green, bottom_left.green, bottom_right.green, x_from_top_left, y_from_top_left),
            .blue = map_one_value(top_left.blue, top_right.blue, bottom_left.blue, bottom_right.blue, x_from_top_left, y_from_top_left),
            .alpha = map_one_value(top_left.alpha, top_right.alpha, bottom_left.alpha, bottom_right.alpha, x_from_top_left, y_from_top_left),
        };
    }

    BmpImage::BmpImage linear_transform(
        BmpImage::BmpImage image,
        Linalg::Matrix<double> matrix,
        BmpImage::BmpPixel canvas_color = BmpImage::BmpPixel{
            .red = 0,
            .green = 0,
            .blue = 0,
            .alpha = 1,
        }
    ){
        auto result_image = image;
        auto inverse_matrix = matrix.pinv();
        auto pixels = image.image.data.interpret(image.image.size.height, image.image.size.width);
        auto get_pixel = [&](int x, int y) {
            if(x < 0 || x >= image.image.size.width || y < 0 || y >= image.image.size.height) {
                return canvas_color;
            }
            return pixels[y][x];
        };

        result_image.image.data.foreach([&](BmpImage::BmpPixel &pixel, size_t idx) {
            auto x = idx % image.image.size.width;
            auto y = idx / image.image.size.width;
            auto point = Linalg::Matrix(
                {
                    {static_cast<double>(x)},
                    {static_cast<double>(y)},
                    {1.}
                }
            );
            auto pre_transformed_point = inverse_matrix * point;
            auto pre_transformed_point_divided = std::make_tuple(
                pre_transformed_point[0][0] / pre_transformed_point[2][0],
                pre_transformed_point[1][0] / pre_transformed_point[2][0]
            );
            auto x_from_top_left = f_part(std::get<0>(pre_transformed_point_divided));
            auto y_from_top_left = f_part(std::get<1>(pre_transformed_point_divided));
            pixel = bilinear_interpolate(
                get_pixel(floor(std::get<0>(pre_transformed_point_divided)), floor(std::get<1>(pre_transformed_point_divided))),
                get_pixel(ceil(std::get<0>(pre_transformed_point_divided)), floor(std::get<1>(pre_transformed_point_divided))),
                get_pixel(floor(std::get<0>(pre_transformed_point_divided)), ceil(std::get<1>(pre_transformed_point_divided))),
                get_pixel(ceil(std::get<0>(pre_transformed_point_divided)), ceil(std::get<1>(pre_transformed_point_divided))),
                x_from_top_left, y_from_top_left
            );
        });
        return result_image;
    }
}
#endif //IMAGE_PROCESSING_LINEAR_TRANSFORM_HXX