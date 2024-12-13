#include <stdio.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <cuda_runtime.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

__global__ void cuda_convolve(unsigned char* image, unsigned char* convolvedImage, int height, int width, int channels, float* kernel, int kernelSize)
{
	// Thread identifiers
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int c = threadIdx.z;

	// Thread boundary check
	if(x < width && y < height && c < channels)
	{
		float pixelTotal = 0.0f;
		float totalWeight = 0.0f;
		int halfKernel = kernelSize / 2;
		for(int rowOffset = -halfKernel; rowOffset <= halfKernel; rowOffset++)
		{
			int xCoordImg = x + rowOffset;
			for(int colOffset = -halfKernel; colOffset <= halfKernel; colOffset++)
			{
				int yCoordImg = y + colOffset;
				int imageIdx = (yCoordImg * width + xCoordImg) * channels + c;
				int kernelIdx = (rowOffset + halfKernel) * kernelSize + (colOffset + halfKernel);

				// Boundary check
				if((xCoordImg < 0 || xCoordImg >= width || yCoordImg < 0 || yCoordImg >= height))
				{
					continue;
				}
				totalWeight += kernel[kernelIdx];
				pixelTotal += kernel[kernelIdx] * float(image[imageIdx]);
			}
		}

		pixelTotal = pixelTotal / abs(totalWeight);
		int outIdx = (y * width + x) * channels + c;
		convolvedImage[outIdx] = (unsigned char)fmax(0.0f, fmin(255.0f, pixelTotal));
	}
}

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
	// Image looping by height x width x color channels
    for(int y = 0; y < height; y++) 
	{
        for(int x = 0; x < width; x++)
		{
			for(int channel = 0; channel < channels; channel++)
			{	
				int radiusx = kernelSize / 2;
				int radiusy = kernelSize / 2;

				float pixelTotal = 0.0f;
				float totalWeight = 0.0f;

				for(int iy = -radiusy; iy <= radiusy; iy++)
				{
					int ready = y + iy;
					for(int ix = -radiusx; ix <= radiusx; ix++)
					{
						int readx = x + ix;
						// Boundary check
						if((readx < 0 || readx >= width || ready < 0 || ready >= height))
						{
							continue;
						}

						float pixelValue = float(image[(ready * width + readx) * channels + channel]);
						float kernelValue = kernel[(iy + radiusy) * kernelSize + ix + radiusx];

						totalWeight += kernelValue;
						pixelTotal += pixelValue * kernelValue;
					}
				}

				pixelTotal = pixelTotal / totalWeight;
				int outIndex = (y * width + x) * channels + channel;
				convolvedImage[outIndex] = (unsigned char)fmax(0.0f, fmin(255.0f, pixelTotal));
			}
        }
    }
}

int main()
{
	// Image vars
	int width, height, channels;
    unsigned char* image;
	char* filenames[] = {
		"./Cat.jpg",
		"./chicago.jpg", 
		"./cliff.jpg", 
		"./coffee.jpg", 
		"./dog.jpg", 
		"./flowers.jpg", 
		"./palace.jpg", 
		"./panda.jpg", 
		"./sunflower.jpg", 
		"./forest(vertical).jpg", 
		"./girl-with-camera(vertical).jpg"
	};

	int numFiles = 11;

	// Timers
	clock_t startTime;
	clock_t endTime;
    double timeElapsed;

	for(int fileNum = 0; fileNum < numFiles; fileNum++)
	{
		char* file = filenames[fileNum];
		
		if(!(image = stbi_load(file, &width, &height, &channels, 0))) 
		{
			fprintf(stderr, "Error loading image:\t%s\n", file);
			exit(1);
		}

		int N = width*height;
		printf("Filename:\t%s\n", file);
		printf("Dimensions:\t%dx%d\n", width, height);

		for(int kernelSize = 3; kernelSize <= 15; kernelSize += 2)
		{
			float* kernel = new float[kernelSize*kernelSize];
			printf("\tKernel:\t%d\n", kernelSize);
			// Create kernel
			for(int row = 0; row < kernelSize; row++)
			{
				for(int col = 0; col < kernelSize; col++)
				{
					kernel[row * kernelSize + col] = 1.0f; // Adjust values for different kernels
				}
			}
			kernel[(kernelSize * kernelSize) / 2 ] = 1.0f;
			
			unsigned char* convolvedImage = (unsigned char*)malloc(N * channels * sizeof(unsigned char));

			// Allocate GPU memory
			unsigned char* d_image;
			unsigned char* d_convolvedImage;
			float* d_kernel;
			cudaMalloc(&d_image, N * channels * sizeof(unsigned char));
			cudaMalloc(&d_convolvedImage, N * channels * sizeof(unsigned char));
			cudaMalloc(&d_kernel, kernelSize * kernelSize * sizeof(float));

			// Copy data into GPU
			cudaMemcpy(d_image, image, N * channels * sizeof(unsigned char), cudaMemcpyHostToDevice);
			cudaMemcpy(d_kernel, kernel, kernelSize * kernelSize * sizeof(float), cudaMemcpyHostToDevice);

			// Define threads
			dim3 blockSize(16, 16, 3);
			dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

			// Start timer
			startTime = clock();

			// Perform operation
			cuda_convolve<<<gridSize, blockSize>>>(d_image, d_convolvedImage, height, width, channels, d_kernel, kernelSize);

			// Wait for operation to complete
			cudaDeviceSynchronize();

			// End timer
			endTime = clock();
			timeElapsed = ((double) (endTime - startTime)) / CLOCKS_PER_SEC;

			// Copy data out of GPU
			cudaMemcpy(convolvedImage, d_convolvedImage, N * channels * sizeof(unsigned char), cudaMemcpyDeviceToHost);

			// Free GPU memory
			cudaFree(d_image);
			cudaFree(d_convolvedImage);
			cudaFree(d_kernel);

			int result; 	
			char outFile[256];
    		snprintf(outFile, sizeof(outFile), "%s%s", file, "_cuda.jpg");
			if(result = stbi_write_png(outFile, width, height, channels, convolvedImage, width * channels)) 
			{
				printf("\t\tCuda convolved Image saved successfully\n");
				for(int i = width; i < width + 10; i++)
				{
					printf("\t\tValue check: %d | %d\n", image[i], convolvedImage[i]);
				}
				printf("\t\tCuda completion time:\t%f\n", timeElapsed);
			}
			else 
			{
				printf("\t\tError saving image\n");
			}

			startTime = clock();
			convolve(image, convolvedImage, height, width, channels, kernel, kernelSize);
			endTime = clock();
			timeElapsed = ((double) (endTime - startTime)) / CLOCKS_PER_SEC;

    		snprintf(outFile, sizeof(outFile), "%s%s", file, "_linear.jpg");
			if(result = stbi_write_png("./output_linear.jpg", width, height, channels, convolvedImage, width * channels)) 
			{
				printf("\t\tLinear convolved Image saved successfully\n");
				for(int i = width; i < width + 10; i++)
				{
					printf("\t\tValue check: %d | %d\n", image[i], convolvedImage[i]);
				}
				printf("\t\tLinear completion time:\t%f\n", timeElapsed);
			}
			else 
			{
				printf("\t\tError saving image\n");
			}
			free(convolvedImage);
			delete kernel;
		}
		stbi_image_free(image);
	}
	
	return 0;
}
