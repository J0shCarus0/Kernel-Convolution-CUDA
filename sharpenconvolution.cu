#include <stdio.h>
#include <math.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

__global__ void evaluate(int x, int* coeffArr, int* outputTerms)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int coefficient = coeffArr[index];
	int result = pow(x, index);
	outputTerms[index] = coefficient*result;
}

#define THREADS_PER_BLOCK 512

void convolve(unsigned char* image, unsigned char* convolvedImage, int height, int width, int channels, float* kernel, int kernelSize) 
{
	int outIndex = 0;
	// Image looping by height x width x color channels
    for(int y = 0; y < height; y++) 
	{
        for(int x = 0; x < width; x++)
		{
			for(int channel = 0; channel < channels; channel++)
			{	
				int radiusx = kernelSize / 2;
				int radiusy = kernelSize / 2;

				float retf = 0.0f;
				float totalWeight = 0.0f;

				for(int iy = -radiusy; iy <= radiusy; iy++)
				{
					int ready = (y + iy + height) % height;
					for(int ix = -radiusx; ix <= radiusx; ix++)
					{
						int readx = (x + ix + width) % width;
						
						float pixelValue = float(image[(ready * width + readx) * channels + channel]) / 255.0f;

						float kernelValue = kernel[(iy + radiusy) * kernelSize + ix + radiusx];

						retf += pixelValue * kernelValue;
						totalWeight += kernelValue;
					}
				}

				retf /= totalWeight;
				convolvedImage[outIndex] = (unsigned char)(fmax(fmin(retf * 256.0f, 255.0f),0.0f));
				outIndex++;
			}
        }
    }
}

int main()
{
	int width, height, channels;
    unsigned char* image;
    if(!(image = stbi_load("./chicago.jpg", &width, &height, &channels, 0))) 
	{
        fprintf(stderr, "Error loading image\n");
        exit(1);
    }

	int N = width*height;
	int k_size = 3;
	float* kernel = new float[k_size*k_size];
	kernel[0] = kernel[2] = kernel[6] = kernel[8] = 0;//0.0023f; // Corners
	kernel[1] = kernel[3] = kernel[5] = kernel[7] = 0;//0.0432f; // Middles
	kernel[4] = 1;//0.8180f; // Center
	
	unsigned char* convolvedImage = stbi_load("./chicago.jpg", &width, &height, &channels, 0);//= (unsigned char*)malloc(N * channels * sizeof(unsigned char*));
    convolve(image, convolvedImage, height, width, channels, kernel, k_size);

	int result; 
    if(result = stbi_write_png("./output.jpg", width, height, channels, convolvedImage, width * channels)) 
	{
		printf("Image saved successfully\n");
		for(int i = 0; i < 10; i++)
		{
			printf("Value check: %d | %d\n", image[i], convolvedImage[i]);
		}
    }
	else 
	{
        printf("Error saving image\n");
    }

    stbi_image_free(image);
	delete kernel;

	// int size = N * sizeof(int);
    // int *coeffArr, *outputTerms;
	// int *d_coeffArr, *d_outputTerms;
	
	// // Allocate space for CPU & GPU arrays
	// cudaMalloc((void**) &d_coeffArr, size);
	// cudaMalloc((void**) &d_outputTerms, size);
	// coeffArr = (int*)malloc(size);
	// outputTerms = (int*)malloc(size);

	// // Fill coefficient and output arrays
	// for(int i = 0; i < N; i++)
	// {
	// 	coeffArr[i] = i;
	// 	outputTerms[i] = 0;
	// }

	// // Copy coefficient array to GPU
	// cudaMemcpy(d_coeffArr, coeffArr, size, cudaMemcpyHostToDevice);

	// // Evaluate polynomial using CUDA and copy results back to main memory
	// int x = 1;
	// evaluate<<<(N + (THREADS_PER_BLOCK-1)) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(x, d_coeffArr, d_outputTerms);
	// cudaMemcpy(outputTerms, d_outputTerms, size, cudaMemcpyDeviceToHost);

	// // Sum output
	// int sum = 0;
	// for(int i = 0; i < N; i++)
	// 	sum += outputTerms[i];
	// printf("N = %d\nx = %d\noutputTerms sum = %d\n", N, x, sum);

	// // Clean up
	// free(coeffArr);
	// free(outputTerms);
	// cudaFree(d_coeffArr);
	// cudaFree(d_outputTerms);
	
	return 0;
}
