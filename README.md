# Exp3-Sobel-edge-detection-filter-using-CUDA-to-enhance-the-performance-of-image-processing-tasks.

<h3>NAME : THILLAI AJAY.L</h3>
<h3>REGISTER NO. : 212224040354</h3>
<h3>EX. NO. : 3</h3>
<h3>DATE : 26 - 03 - 2026</h3>
<h1> <align=center> Sobel edge detection filter using CUDA </h3>
  Implement Sobel edge detection filtern using GPU.</h3>
Experiment Details:
  
## AIM:
  The Sobel operator is a popular edge detection method that computes the gradient of the image intensity at each pixel. It uses convolution with two kernels to determine the gradient in both the x and y directions. This lab focuses on utilizing CUDA to parallelize the Sobel filter implementation for efficient processing of images.

Code Overview: You will work with the provided CUDA implementation of the Sobel edge detection filter. The code reads an input image, applies the Sobel filter in parallel on the GPU, and writes the result to an output image.
## EQUIPMENTS REQUIRED:
Hardware – PCs with NVIDIA GPU & CUDA NVCC
Google Colab with NVCC Compiler
CUDA Toolkit and OpenCV installed.
A sample image for testing.

## PROCEDURE:
Tasks: 
a. Modify the Kernel:

Update the kernel to handle color images by converting them to grayscale before applying the Sobel filter.
Implement boundary checks to avoid reading out of bounds for pixels on the image edges.

b. Performance Analysis:

Measure the performance (execution time) of the Sobel filter with different image sizes (e.g., 256x256, 512x512, 1024x1024).
Analyze how the block size (e.g., 8x8, 16x16, 32x32) affects the execution time and output quality.

c. Comparison:

Compare the output of your CUDA Sobel filter with a CPU-based Sobel filter implemented using OpenCV.
Discuss the differences in execution time and output quality.

## PROGRAM:
```cpp
%%writefile sobelEdgeDetectionFilter.cu

#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

using namespace cv;

__global__ void sobelFilter(unsigned char *srcImage, unsigned char *dstImage,  
                            unsigned int width, unsigned int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= 1 && x < width - 1 && y >= 1 && y < height - 1) {
        int Gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
        int Gy[3][3] = {{1, 2, 1}, {0, 0, 0}, {-1, -2, -1}};

        int sumX = 0;
        int sumY = 0;

        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                unsigned char pixel = srcImage[(y + i) * width + (x + j)];
                sumX += pixel * Gx[i + 1][j + 1];
                sumY += pixel * Gy[i + 1][j + 1];
            }
        }

        int magnitude = sqrtf(sumX * sumX + sumY * sumY);
        magnitude = min(max(magnitude, 0), 255);
        dstImage[y * width + x] = static_cast<unsigned char>(magnitude);
    }
}



void checkCudaErrors(cudaError_t r)
{
    if (r != cudaSuccess)
    {
        fprintf(stderr,"CUDA Error: %s\n", cudaGetErrorString(r));
        exit(EXIT_FAILURE);
    }
}

int main()
{
    // Read input image
    Mat image = imread("/content/image.jpg", IMREAD_GRAYSCALE);

    if(image.empty())
    {
        printf("Error: Image not found\n");
        return -1;
    }

    int width = image.cols;
    int height = image.rows;

    size_t imageSize = width * height * sizeof(unsigned char);

    // Host memory
    unsigned char *h_outputImage = (unsigned char*)malloc(imageSize);

    // Device memory
    unsigned char *d_inputImage;
    unsigned char *d_outputImage;

    checkCudaErrors(cudaMalloc(&d_inputImage, imageSize));
    checkCudaErrors(cudaMalloc(&d_outputImage, imageSize));

    checkCudaErrors(cudaMemcpy(d_inputImage, image.data, imageSize, cudaMemcpyHostToDevice));

    // CUDA events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // CUDA configuration
    dim3 blockSize(16,16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,(height + blockSize.y - 1) / blockSize.y);

    cudaEventRecord(start);

    sobelFilter<<<gridSize, blockSize>>>(d_inputImage, d_outputImage, width, height);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Copy result
    checkCudaErrors(cudaMemcpy(h_outputImage, d_outputImage, imageSize, cudaMemcpyDeviceToHost));

    // Save output
    Mat outputImage(height, width, CV_8UC1, h_outputImage);
    imwrite("output_sobel.jpeg", outputImage);

    // Free memory
    free(h_outputImage);
    cudaFree(d_inputImage);
    cudaFree(d_outputImage);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("Total time taken: %f milliseconds\n", milliseconds);

    return 0;
}
```

## OUTPUT:
## Sobel Edge Detection Result

| Input Image | CUDA Output Image | CPU Output Image |
|-------------|--------------|--------------|
| ![image (1)](https://github.com/user-attachments/assets/088d4318-0146-4e25-87c2-a6e0223ff36a) | <img width="389" height="411" alt="image" src="https://github.com/user-attachments/assets/89c1e8a4-d9d1-4bfc-aaaa-19482923717b" /> | <img width="389" height="411" alt="image" src="https://github.com/user-attachments/assets/3f7cfa2d-9eef-49d8-a3d9-0f26b12efe55" /> |



## Answers to Questions:

1.What challenges did you face while implementing the Sobel filter for color images?

While implementing the Sobel filter for color images, the main challenge was that the Sobel operator works best on grayscale images. Color images contain three channels (Red, Green, and Blue), which increases computational complexity. Therefore, the color image must first be converted into grayscale before applying the Sobel filter. Another challenge was handling boundary pixels, since accessing neighboring pixels outside the image boundary can cause memory access errors.

2.How did changing the block size influence the performance of your CUDA implementation?

Changing the block size affects how threads are organized and executed on the GPU. Smaller block sizes such as 8×8 may lead to underutilization of GPU resources, while larger block sizes like 32×32 can improve parallelism but may increase resource usage per block. In this experiment, 16×16 blocks provided a balanced performance, ensuring efficient thread execution and better GPU utilization.

3.What were the differences in output between the CUDA and CPU implementations? Discuss any discrepancies.

The CUDA and CPU implementations produced similar edge detection results since both apply the same Sobel convolution kernels. However, minor differences may occur due to floating-point precision and parallel computation order. The main difference observed was in execution time, where the CUDA implementation performed faster than the CPU version due to parallel processing on the GPU.

4.Suggest potential optimizations for improving the performance of the Sobel filter.

   - Several optimizations can improve the performance of the Sobel filter:
    
   - Using shared memory to reduce global memory access.
    
   - Storing Sobel kernels in constant memory for faster access.
    
   - Optimizing thread block size based on GPU architecture.
    
   - Reducing redundant memory accesses through memory coalescing.
    
   - Processing multiple pixels per thread for better GPU utilization.

## Deliverables


- **Report of Findings:**  
  The Sobel edge detection filter was implemented using CUDA to improve the performance of image processing tasks. The program reads an input grayscale image and applies the Sobel operator to detect edges by computing the gradient in both horizontal and vertical directions. CUDA was used to parallelize the computation so that multiple pixels could be processed simultaneously using GPU threads organized in 2D grids and blocks.

  The execution time of the CUDA implementation was measured using CUDA events. The results show that GPU-based processing significantly reduces the computation time compared to the CPU-based implementation. As the image size increases, the execution time also increases due to the larger number of pixels being processed. However, the CUDA implementation still performs faster because the GPU executes many threads in parallel.

  A comparison between CPU and CUDA implementations shows that both produce similar edge detection results. The difference lies mainly in performance, where CUDA achieves faster execution due to parallel processing capabilities of the GPU.

- **Performance Table:**
  | Image Size  | Block Size | CUDA Execution Time |
  | ----------- | ---------- | ------------------- |
  | 256 × 256   | 8 × 8      | 0.163648 ms         |
  | 512 × 512   | 16 × 16    | 0.161856 ms         |
  | 1024 × 1024 | 32 × 32    | 0.196960 ms         |

  | Implementation | Execution Time |
  | -------------- | -------------- |
  | CPU (OpenCV)   | 30.23672103881836 ms |
  | CUDA (GPU)     | ~0.16 ms       |


- **Performance Analysis:**
  From the results, it can be observed that the CUDA implementation provides significant performance improvement compared to the CPU implementation. As the image resolution increases, the processing time also increases. However, the GPU handles larger workloads efficiently due to parallel execution of threads. Increasing the block size improves GPU utilization up to a certain limit, after which the performance gain becomes minimal.

## RESULT:

Thus the program has been executed by using CUDA to perform Sobel edge detection on a grayscale image using parallel GPU threads, thereby enhancing the performance of image processing tasks.



