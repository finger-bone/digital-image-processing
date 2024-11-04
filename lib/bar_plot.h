#ifndef IMAGE_PROCESSING_BAR_PLOT_H
#define IMAGE_PROCESSING_BAR_PLOT_H

#include "bmp_image.h"
#include "numeric_array.h"

#include <vector>

namespace BarPlot { 
    BmpImage::BmpImage generate_blank_canvas(int width, int height, BmpImage::BmpPixel color = { 255, 255, 255, 255 }) {
        BmpImage::BmpHeader header {
            .fileHeader = {
                .fileType = 0x4D42,
            },
            .infoHeader = {
                .headerSize = 40,
                .width = width,
                .height = height,
                .planes = 1,
                .bitsPerPixel = 24,
            }
        };
        return BmpImage::BmpImage {
            .header = header,
            .image = {
                .size = {
                    .width = width,
                    .height = height,
                },
                .data = NumericArray::NumericArray<BmpImage::BmpPixel> {
                    std::vector<BmpImage::BmpPixel>(width * height, color),
                }
            }
        };
    }

    void bar_plot(BmpImage::BmpImage &image, std::vector<int> values, int chunks, BmpImage::BmpPixel color = { 0, 0, 0, 255}) {
        int width = image.image.size.width;
        int height = image.image.size.height;
        int bar_width = width / chunks;
        int max_value = *std::max_element(values.begin(), values.end());
        int chunk_size = values.size() / chunks;
        std::vector<int> bar_heights(chunks);
        for (int i = 0; i < chunks; i++) {
            int sum = 0;
            for(int j = 0; j < chunk_size; j++) {
                sum += values[i * chunk_size + j];
            }
            bar_heights[i] = height * sum / chunk_size / max_value;
        }

        image.image.data.map_inplace([&](BmpImage::BmpPixel p, size_t idx) {
            int x =  idx % width;
            int y = idx / width;
            // find the correct bar of x
            int bar_idx = x / bar_width;
            int bar_height = bar_heights[bar_idx];
            if (bar_idx < 0 || bar_idx >= chunks) {
                return p;
            }
            if (y <= bar_height) {
                return color;
            } else {
                return p;
            }
        });
    }
}

#endif