#ifndef CONVOLUTION_HXX
#define CONVOLUTION_HXX

namespace Convolution {

#include "bmp_image.hxx"
#include "numeric_array.hxx"
#include <vector>
#include <algorithm>

    BmpImage::BmpPixel& get_pixel_with_padding(BmpImage::BmpImage& img, int x, int y) {
        auto width = img.image.size.width;
        auto height = img.image.size.height;
        x = std::clamp(x, 0, width - 1);
        y = std::clamp(y, 0, height - 1);
        return img.image.data.data[y * width + x];
    }

    void apply_kernel(BmpImage::BmpImage& img, const std::vector<std::vector<double>>& kernel) {
        int kernel_size = kernel.size();
        int kernel_half_size = kernel_size / 2;

        // Validate kernel
        if (kernel_size % 2 == 0) {
            throw std::invalid_argument("Kernel size must be odd.");
        }

        // Create a new array to store the convolved image
        NumericArray::NumericArray<BmpImage::BmpPixel> newData(
            img.image.data.data.size(),
            BmpImage::BmpPixel {
                .red = 0,
                .green = 0,
                .blue = 0,
                .alpha = 255 // Assuming alpha remains 255 for all pixels
            }
        );

        int width = img.image.size.width;
        int height = img.image.size.height;

        // Perform convolution
        img.image.data.foreach([&](BmpImage::BmpPixel& pxl, size_t idx) {
            int x = idx % width;
            int y = idx / width;

            double red = 0, green = 0, blue = 0;

            // Apply the kernel
            for (int ky = -kernel_half_size; ky <= kernel_half_size; ++ky) {
                for (int kx = -kernel_half_size; kx <= kernel_half_size; ++kx) {
                    BmpImage::BmpPixel neighbor = get_pixel_with_padding(img, x + kx, y + ky);
                    double weight = kernel[ky + kernel_half_size][kx + kernel_half_size];

                    red += neighbor.red * weight;
                    green += neighbor.green * weight;
                    blue += neighbor.blue * weight;
                }
            }

            // Assign the convolved values to the new pixel array
            BmpImage::BmpPixel& new_pixel = newData.data[idx];
            new_pixel.red = std::clamp(static_cast<int>(red), 0, 255);
            new_pixel.green = std::clamp(static_cast<int>(green), 0, 255);
            new_pixel.blue = std::clamp(static_cast<int>(blue), 0, 255);
        });

        // Replace the original image data with the convolved data
        img.image.data = newData;
    }

    void apply_mid_value_kernel(BmpImage::BmpImage& img, size_t kernel_size, int k) {
        if (kernel_size % 2 == 0) {
            throw std::invalid_argument("Kernel size must be odd.");
        }
        int kernel_half_size = kernel_size / 2;

        NumericArray::NumericArray<BmpImage::BmpPixel> newData(
            img.image.data.data.size(),
            BmpImage::BmpPixel {
                .red = 0,
                .green = 0,
                .blue = 0,
                .alpha = 255 // Assuming alpha remains 255 for all pixels
            }
        );

        int width = img.image.size.width;
        int height = img.image.size.height;

        // Perform convolution
        img.image.data.foreach([&](BmpImage::BmpPixel& pxl, size_t idx) {
            int x = idx % width;
            int y = idx / width;

            double red = 0, green = 0, blue = 0;

            std::vector<BmpImage::BmpPixel> window;
            for (int ky = -kernel_half_size; ky <= kernel_half_size; ++ky) {
                for (int kx = -kernel_half_size; kx <= kernel_half_size; ++kx) {
                    window.push_back(get_pixel_with_padding(img, x + kx, y + ky));
                }
            }

            std::sort(
                window.begin(),
                window.end(),
                [](const BmpImage::BmpPixel& lhs, const BmpImage::BmpPixel& rhs) {
                    return lhs.gray() > rhs.gray();
                }
            );
            auto target_pixel = window[k];
            newData.data[idx] = target_pixel;
        });

        // Replace the original image data with the convolved data
        img.image.data = newData;
    }

} // namespace Convolution

#endif