#include <stdio.h>
#define DEBUG
#define NUM_CHANNELS 3

__global__ void color_to_grayscale_conversion(unsigned char* in, char* out, int width, int height){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row<0 || row>=height || col < 0 || col >= width) return;

    int grey_offset = row * width + col;

    int rgb_offset = row * width + col;

    int rgb_offset = grey_offset * NUM_CHANNELS;

    unsigned char r = in[rgb_offset + 0];
    unsigned char g = in[rgb_offset + 1];
    unsigned char b = in[rgb_offset + 2];

    out[grey_offset] = (unsigned char)(0.21f * r + 0.71f * g + 0.07f * b);
}

int main(int argc, char* argv[]){
    int block_dim;
    int image_width;
    int image_height;
    int size;
    unsigned char* h_input_image, *h_output_image;
    unsigned char* d_input_image, *d_output_image;

    block_dim = 32;
    image_width = 10;
    image_height = 10;
    size = image_width * image_height;
}