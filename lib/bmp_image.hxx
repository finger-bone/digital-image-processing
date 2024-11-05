//
// Created by Zend on 2024/11/3.
//

#ifndef IMAGE_PROCESSING_BMP_IMAGE_HXX
#define IMAGE_PROCESSING_BMP_IMAGE_HXX

#include "numeric_array.hxx"
#include <fstream>
#include <functional>
#include <future>
#include <iostream>
#include <map>
#include <set>
#include <thread>
#include <vector>

namespace BmpImage {
struct BmpFileHeader {
  uint16_t fileType;  // 2 bytes: BMP file type, should be 0x4D42 ("BM")
  uint32_t fileSize;  // 4 bytes: Size of the BMP file in bytes
  uint16_t reserved1; // 2 bytes: Reserved, must be 0
  uint16_t reserved2; // 2 bytes: Reserved, must be 0
  uint32_t
      pixelDataOffset; // 4 bytes: Offset from start of file to the pixel data
};

struct BmpInfoHeader {
  uint32_t headerSize;   // 4 bytes: Size of this header, typically 40 bytes
  int32_t width;         // 4 bytes: Width of the bitmap in pixels
  int32_t height;        // 4 bytes: Height of the bitmap in pixels
  uint16_t planes;       // 2 bytes: Number of color planes, must be 1
  uint16_t bitsPerPixel; // 2 bytes: Bits per pixel (color depth), e.g., 24 for
                         // true color
  uint32_t
      compression;    // 4 bytes: Compression type, 0 = BI_RGB (no compression)
  uint32_t imageSize; // 4 bytes: Size of the image data in bytes, can be 0 if
                      // no compression
  int32_t xPixelsPerMeter; // 4 bytes: Horizontal resolution (pixels per meter)
  int32_t yPixelsPerMeter; // 4 bytes: Vertical resolution (pixels per meter)
  uint32_t totalColors; // 4 bytes: Number of colors in the color palette, 0 if
                        // all colors are important
  uint32_t importantColors; // 4 bytes: Number of important colors, 0 if all are
                            // important
};

struct BmpHeader {
  BmpFileHeader fileHeader;
  BmpInfoHeader infoHeader;
};

struct ImageSize {
  int width;
  int height;
};

struct BmpPixel {
  uint8_t red;
  uint8_t green;
  uint8_t blue;
  uint8_t alpha;
};

struct ColorPalette {
  std::vector<BmpPixel> data;
  std::map<std::tuple<uint8_t, uint8_t, uint8_t, uint8_t>, int> generate_map() {
    std::map<std::tuple<uint8_t, uint8_t, uint8_t, uint8_t>, int> map;
    for (int i = 0; i < data.size(); i++) {
      auto pixel = data[i];
      map[std::make_tuple(pixel.red, pixel.green, pixel.blue, pixel.alpha)] = i;
    }
    return map;
  }
};

template <typename T> struct Image {
  ImageSize size;
  NumericArray::NumericArray<T> data;
};

struct BmpImage {
  BmpHeader header;
  Image<BmpPixel> image;
  ColorPalette palette;

  void set_bbp(int bbp) { header.infoHeader.bitsPerPixel = bbp; }

  void regenerate_palette() {
    palette.data.clear();
    std::set<std::tuple<uint8_t, uint8_t, uint8_t, uint8_t>> unique_colors;
    for (int i = 0; i < image.data.data.size(); i++) {
      auto pixel = image.data.data[i];
      unique_colors.insert(
          std::make_tuple(pixel.red, pixel.green, pixel.blue, pixel.alpha));
    }
    for (auto color : unique_colors) {
      palette.data.push_back({std::get<0>(color), std::get<1>(color),
                              std::get<2>(color), std::get<3>(color)});
    }
  }

  void regenerate_header() {
    int header_size = 14 + this->header.infoHeader.headerSize;
    int palette_size = this->palette.data.size() * 4;
    int image_size = this->image.data.data.size();
    int bbp = this->header.infoHeader.bitsPerPixel;
    this->header.fileHeader.fileSize =
        header_size + palette_size + image_size * (bbp / 8);
    this->header.fileHeader.pixelDataOffset = header_size + palette_size;
  }

  void change_to_eight_bit() {
    this->set_bbp(8);
    this->regenerate_palette();
    this->regenerate_header();
  }
};

bool has_palette(int bbp) { return bbp <= 8; }

void read_header(std::ifstream &file, BmpHeader &header) {
  size_t file_header_size = sizeof(BmpFileHeader);
  size_t info_header_size = sizeof(BmpInfoHeader);
  file.read(reinterpret_cast<char *>(&header.fileHeader.fileType),
            sizeof(header.fileHeader.fileType));
  file.read(reinterpret_cast<char *>(&header.fileHeader.fileSize),
            sizeof(header.fileHeader.fileSize));
  file.read(reinterpret_cast<char *>(&header.fileHeader.reserved1),
            sizeof(header.fileHeader.reserved1));
  file.read(reinterpret_cast<char *>(&header.fileHeader.reserved2),
            sizeof(header.fileHeader.reserved2));
  file.read(reinterpret_cast<char *>(&header.fileHeader.pixelDataOffset),
            sizeof(header.fileHeader.pixelDataOffset));
  file.read(reinterpret_cast<char *>(&header.infoHeader.headerSize),
            sizeof(header.infoHeader.headerSize));
  file.read(reinterpret_cast<char *>(&header.infoHeader.width),
            sizeof(header.infoHeader.width));
  file.read(reinterpret_cast<char *>(&header.infoHeader.height),
            sizeof(header.infoHeader.height));
  file.read(reinterpret_cast<char *>(&header.infoHeader.planes),
            sizeof(header.infoHeader.planes));
  file.read(reinterpret_cast<char *>(&header.infoHeader.bitsPerPixel),
            sizeof(header.infoHeader.bitsPerPixel));
  file.read(reinterpret_cast<char *>(&header.infoHeader.compression),
            sizeof(header.infoHeader.compression));
  file.read(reinterpret_cast<char *>(&header.infoHeader.imageSize),
            sizeof(header.infoHeader.imageSize));
  file.read(reinterpret_cast<char *>(&header.infoHeader.xPixelsPerMeter),
            sizeof(header.infoHeader.xPixelsPerMeter));
  file.read(reinterpret_cast<char *>(&header.infoHeader.yPixelsPerMeter),
            sizeof(header.infoHeader.yPixelsPerMeter));
  file.read(reinterpret_cast<char *>(&header.infoHeader.totalColors),
            sizeof(header.infoHeader.totalColors));
  file.read(reinterpret_cast<char *>(&header.infoHeader.importantColors),
            sizeof(header.infoHeader.importantColors));
}

int get_palette_bytes(BmpHeader &header) {
  return header.fileHeader.pixelDataOffset - header.infoHeader.headerSize - 14;
}

ColorPalette read_palette(std::ifstream &file, BmpHeader &header) {
  int palette_size = get_palette_bytes(header) / 4;
  ColorPalette palette;
  for (int i = 0; i < palette_size; i++) {
    auto pixel = BmpPixel();
    file.read(reinterpret_cast<char *>(&pixel.blue), sizeof(pixel.blue));
    file.read(reinterpret_cast<char *>(&pixel.green), sizeof(pixel.green));
    file.read(reinterpret_cast<char *>(&pixel.red), sizeof(pixel.red));
    file.read(reinterpret_cast<char *>(&pixel.alpha), sizeof(pixel.alpha));
    palette.data.push_back(pixel);
  }
  return palette;
}

std::vector<BmpPixel> read_8_bit_image(std::ifstream &file, BmpHeader &header,
                                       ColorPalette &palette) {
  auto width = header.infoHeader.width;
  auto height = header.infoHeader.height;
  auto image_size = width * height;
  auto padding =
      (4 - (width % 4)) % 4; // Padding per row to align to a 4-byte boundary
  std::vector<BmpPixel> image(image_size);

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      uint8_t colorIndex;
      file.read(reinterpret_cast<char *>(&colorIndex), sizeof(colorIndex));

      // Map the color index to the actual color in the palette
      image[y * width + x] = palette.data[colorIndex];
    }

    // Skip the padding bytes at the end of each row
    file.ignore(padding);
  }

  return image;
}

std::vector<BmpPixel> read_24_bit_image(std::ifstream &file,
                                        BmpHeader &header) {
  auto width = header.infoHeader.width;
  auto height = header.infoHeader.height;
  auto padding = (4 - (width * 3) % 4) %
                 4; // Each row must be padded to a multiple of 4 bytes
  std::vector<BmpPixel> image(width * height);

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      BmpPixel pixel;
      file.read(reinterpret_cast<char *>(&pixel.blue), sizeof(pixel.blue));
      file.read(reinterpret_cast<char *>(&pixel.green), sizeof(pixel.green));
      file.read(reinterpret_cast<char *>(&pixel.red), sizeof(pixel.red));
      pixel.alpha = 255; // BMP 24-bit images don’t have an alpha channel, so
                         // set it to opaque
      image[(height - 1 - y) * width + x] =
          pixel; // BMP stores pixels from bottom to top
    }
    file.ignore(padding); // Skip padding bytes at the end of each row
  }

  return image;
}

BmpImage read_bmp(std::ifstream &file) {
  BmpHeader header;
  read_header(file, header);
  if (header.fileHeader.fileType != 0x4D42) {
    throw std::runtime_error("Not a BMP file");
  }
  auto palette = read_palette(file, header);
  auto bbp = header.infoHeader.bitsPerPixel;
  auto image_size = header.infoHeader.width * header.infoHeader.height;
  auto padding = (4 - (header.infoHeader.width * bbp / 8) % 4) % 4;
  auto image = std::vector<BmpPixel>(image_size);
  if (has_palette(bbp)) {
    if (bbp == 8) {
      image = read_8_bit_image(file, header, palette);
    } else {
      throw std::runtime_error("Unsupported bit depth");
    }
  } else {
    if (bbp == 24) {
      image = read_24_bit_image(file, header);
    } else {
      throw std::runtime_error("Unsupported bit depth");
    }
  }
  BmpImage bmpImage;
  bmpImage.header = header;
  bmpImage.image.size = {header.infoHeader.width, header.infoHeader.height};
  bmpImage.image.data.data = image;
  bmpImage.palette = palette;
  return bmpImage;
}

void write_header(std::ofstream &file, BmpHeader &header) {
  file.write(reinterpret_cast<char *>(&header.fileHeader.fileType),
             sizeof(header.fileHeader.fileType));
  file.write(reinterpret_cast<char *>(&header.fileHeader.fileSize),
             sizeof(header.fileHeader.fileSize));
  file.write(reinterpret_cast<char *>(&header.fileHeader.reserved1),
             sizeof(header.fileHeader.reserved1));
  file.write(reinterpret_cast<char *>(&header.fileHeader.reserved2),
             sizeof(header.fileHeader.reserved2));
  file.write(reinterpret_cast<char *>(&header.fileHeader.pixelDataOffset),
             sizeof(header.fileHeader.pixelDataOffset));
  file.write(reinterpret_cast<char *>(&header.infoHeader.headerSize),
             sizeof(header.infoHeader.headerSize));
  file.write(reinterpret_cast<char *>(&header.infoHeader.width),
             sizeof(header.infoHeader.width));
  file.write(reinterpret_cast<char *>(&header.infoHeader.height),
             sizeof(header.infoHeader.height));
  file.write(reinterpret_cast<char *>(&header.infoHeader.planes),
             sizeof(header.infoHeader.planes));
  file.write(reinterpret_cast<char *>(&header.infoHeader.bitsPerPixel),
             sizeof(header.infoHeader.bitsPerPixel));
  file.write(reinterpret_cast<char *>(&header.infoHeader.compression),
             sizeof(header.infoHeader.compression));
  file.write(reinterpret_cast<char *>(&header.infoHeader.imageSize),
             sizeof(header.infoHeader.imageSize));
  file.write(reinterpret_cast<char *>(&header.infoHeader.xPixelsPerMeter),
             sizeof(header.infoHeader.xPixelsPerMeter));
  file.write(reinterpret_cast<char *>(&header.infoHeader.yPixelsPerMeter),
             sizeof(header.infoHeader.yPixelsPerMeter));
  file.write(reinterpret_cast<char *>(&header.infoHeader.totalColors),
             sizeof(header.infoHeader.totalColors));
  file.write(reinterpret_cast<char *>(&header.infoHeader.importantColors),
             sizeof(header.infoHeader.importantColors));
}

void write_palette(std::ofstream &file, std::vector<BmpPixel> &palette) {
  for (auto &pixel : palette) {
    file.write(reinterpret_cast<char *>(&pixel.blue), sizeof(pixel.blue));
    file.write(reinterpret_cast<char *>(&pixel.green), sizeof(pixel.green));
    file.write(reinterpret_cast<char *>(&pixel.red), sizeof(pixel.red));
    file.write(reinterpret_cast<char *>(&pixel.alpha), sizeof(pixel.alpha));
  }
}

void write_8_bit_image(std::ofstream &file, BmpImage &bmpImage) {
  auto palette = bmpImage.palette.generate_map();
  auto image = bmpImage.image.data.data;
  auto bbp = bmpImage.header.infoHeader.bitsPerPixel;
  auto width = bmpImage.image.size.width;
  auto height = bmpImage.image.size.height;
  auto padding = (4 - (width * bbp / 8) % 4) %
                 4; // Padding per row to align to 4-byte boundary

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      auto pixel = image[y * width + x];
      auto index =
          std::make_tuple(pixel.red, pixel.green, pixel.blue, pixel.alpha);

      // Retrieve the corresponding palette index for the pixel color
      uint8_t palette_pixel = static_cast<uint8_t>(palette.at(index));
      file.write(reinterpret_cast<char *>(&palette_pixel),
                 sizeof(palette_pixel));
    }

    // Write padding bytes at the end of each row
    for (int p = 0; p < padding; p++) {
      uint8_t padding_byte = 0;
      file.write(reinterpret_cast<char *>(&padding_byte), sizeof(padding_byte));
    }
  }
}

void write_24_bit_image(std::ofstream &file, BmpImage &bmpImage) {
  auto image = bmpImage.image.data.data;
  auto bbp = bmpImage.header.infoHeader.bitsPerPixel;
  auto width = bmpImage.image.size.width;
  auto height = bmpImage.image.size.height;
  auto padding = (4 - (width * bbp / 8) % 4) %
                 4; // Padding per row to align to 4-byte boundary
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      auto pixel = image[y * width + x];
      file.write(reinterpret_cast<char *>(&pixel.blue), sizeof(pixel.blue));
      file.write(reinterpret_cast<char *>(&pixel.green), sizeof(pixel.green));
      file.write(reinterpret_cast<char *>(&pixel.red), sizeof(pixel.red));
    }
    // Write padding bytes at the end of each row
    for (int p = 0; p < padding; p++) {
      uint8_t padding_byte = 0;
      file.write(reinterpret_cast<char *>(&padding_byte), sizeof(padding_byte));
    }
  }
}

void write_bmp(std::ofstream &file, BmpImage &bmpImage) {
  write_header(file, bmpImage.header);
  if (bmpImage.header.infoHeader.bitsPerPixel == 8) {
    write_palette(file, bmpImage.palette.data);
    write_8_bit_image(file, bmpImage);
  } else {
    write_24_bit_image(file, bmpImage);
  }
}
} // namespace BmpImage

#endif // IMAGE_PROCESSING_BMP_IMAGE_H
