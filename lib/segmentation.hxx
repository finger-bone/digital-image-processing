#ifndef IMAGE_PROCESSING_SEGMENTATION_HXX
#define IMAGE_PROCESSING_SEGMENTATION_HXX

#include "bmp_image.hxx"
#include <vector>
#include <algorithm>

namespace Segmentation {

BmpImage::BmpImage segment_by_threshold(
    BmpImage::BmpImage& img_src, int threshold,
    BmpImage::BmpPixel left_color = {0, 0, 0, 255},
    BmpImage::BmpPixel right_color = {255, 255, 255, 255}
) {
    auto img = img_src;
    img.image.data.foreach ([&](BmpImage::BmpPixel& pxl, size_t idx) {
        if (pxl.gray() < threshold) {
            pxl = left_color;
        } else {
            pxl = right_color;
        }
    });
    return img;
}

int auto_find_threshold_by_iteration(BmpImage::BmpImage& img_src, int max_iterations = 1000, double eps = 2) {
    auto img = img_src;
    int threshold = 128; // Initial threshold
    double left_mean = 0;
    double right_mean = 0;
    int left_count = 0;
    int right_count = 0;
    int iterations = 0;
    while (iterations < max_iterations) {
        left_mean = 0;
        right_mean = 0;
        left_count = 0;
        right_count = 0;
        img.image.data.foreach_sync ([&](BmpImage::BmpPixel& pxl, size_t idx) {
            if (pxl.gray() < threshold) {
                left_mean += pxl.gray();
                left_count++;
            } else {
                right_mean += pxl.gray();
                right_count++;
            }
        });
        if (left_count == 0 || right_count == 0) {
            break;
        }
        left_mean /= left_count;
        right_mean /= right_count;
        if (left_mean == threshold || right_mean == threshold) {
            break;
        }
        if(threshold - left_mean < eps && right_mean - threshold < eps) {
            break;
        }
        threshold = (left_mean + right_mean) / 2;
        iterations++;
    }
    return threshold;
}

int auto_find_threshold_by_otsu(BmpImage::BmpImage& img_src) {
    auto img = img_src;
    int threshold = 0;
    double max_variance = 0;
    for(int i = 0; i <= 256; i++) {
        int left_count = 0;
        int right_count = 0;
        int left_sum = 0;
        int right_sum = 0;
        img.image.data.foreach_sync ([&](BmpImage::BmpPixel& pxl, size_t idx) {
            if (pxl.gray() < i) {
                left_sum += pxl.gray();
                left_count++;
            } else {
                right_sum += pxl.gray();
                right_count++;
            }
        });
        if (left_count == 0 || right_count == 0) {
            continue;
        }
        double left_mean = static_cast<double>(left_sum) / left_count;
        double right_mean = static_cast<double>(right_sum) / right_count;
        double variance = left_count * right_count * (left_mean - right_mean) * (left_mean - right_mean);
        if (variance > max_variance) {
            max_variance = variance;
            threshold = i;
        }
    }
    return threshold;
}

}

#endif