#include "bmp_image.h"
#include <sys/ioctl.h>
#include <unistd.h>

void print_image(const Image::BmpImage &image) {
    // Get terminal size
    struct winsize w;
    ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);
    
    // Terminal dimensions (columns = width, rows = height)
    int term_width = w.ws_col;
    int term_height = w.ws_row;
    
    // Calculate scaling factors
    double scale_x = static_cast<double>(term_width) / image.image.size.width;
    double scale_y = static_cast<double>(term_height) / image.image.size.height;
    double scale = std::min(scale_x, scale_y * 4);
    
    // New dimensions
    int new_width = static_cast<int>(image.image.size.width * scale);
    int new_height = static_cast<int>(image.image.size.height * scale);
    
    // Resize and print using ASCII characters or ANSI colors
    for (int y = 0; y < new_height; y++) {
        for (int x = 0; x < new_width; x++) {
            int orig_x = static_cast<int>(x / scale);
            int orig_y = static_cast<int>(y / scale);
            
            // Get the pixel color
            auto pixel = image.image.data.data[orig_y * image.image.size.width + orig_x];
            // Print using ANSI escape codes for colored output
            printf("\033[48;2;%d;%d;%dm \033[0m", 
                   pixel.red, pixel.green, pixel.blue);
        }
        printf("\n");
    }
}
