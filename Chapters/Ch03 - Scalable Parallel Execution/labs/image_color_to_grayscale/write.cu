#include <stdio.h>

#define DEBUG

#define NUM_CHANNELS 3

__global__
void color_to_grayscale_conversion(unsigned char* in, unsigned char* out, int width, int height){

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < 0 || row >= height || col < 0 || col >= width) return;

    int grey_offset = row * width + col;

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

    // Allocate memory for the input and output images on host
    h_input_image = (unsigned char*) malloc(NUM_CHANNELS * size * sizeof(unsigned char));
    h_output_image = (unsigned char*) malloc(size * sizeof(unsigned char));
    
    // Allocate memory for the input and output images on device
    cudaMalloc((void**) &d_input_image, NUM_CHANNELS * size * sizeof(unsigned char));
    cudaMalloc((void**) &d_output_image, size * sizeof(unsigned char));

    // Initialize the input image with random values between 0 and 255
    srand(time(NULL));
    for(int i = 0; i < NUM_CHANNELS * size; ++i)
        h_input_image[i] = rand() % 256;

#ifdef DEBUG
    // Show the input image
    for(int c = 0; c < NUM_CHANNELS; ++c){
        for(int i = 0; i < image_height; ++i){
            for(int j = 0; j < image_width; ++j){
                printf("%3d ", h_input_image[NUM_CHANNELS * (i * image_width + j) + c]);
            }
            printf("\n");
        }
        printf("\n");
    }
    printf("\n");
#endif

    // Copy the input image to the device
    cudaMemcpy(d_input_image, h_input_image, NUM_CHANNELS * size * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // Blur the image
    dim3 dimBlock(block_dim, block_dim, 1);
    dim3 dimGrid(ceil((float)image_width/block_dim), ceil((float)image_height/block_dim), 1);
    color_to_grayscale_conversion<<<dimGrid, dimBlock>>>(d_input_image, d_output_image, image_width, image_height);

    cudaMemcpy(h_output_image, d_output_image, size * sizeof(unsigned char *), cudaMemcpyDeviceToDevice);

    cudaFree(d_input_image);
    cudaFree(d_output_image);

    free(h_input_image);
    free(h_output_image);
}
