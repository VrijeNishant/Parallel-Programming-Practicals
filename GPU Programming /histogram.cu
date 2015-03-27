
#include <Timer.hpp>
#include <iostream>
#include <iomanip>

using LOFAR::NSTimer;
using std::cout;
using std::cerr;
using std::endl;
using std::fixed;
using std::setprecision;

#define        OFFLOAD_KERNEL_HISTOGRAM                1

#define        THREADS_PER_BLOCK_X        16
#define        THREADS_PER_BLOCK_Y        16

 __global__ void 
kernel_optimized5( unsigned char *grayImage,long size, unsigned int *histogram, unsigned int height, unsigned int width )
{   
    __shared__ unsigned int temp[256];
    int index = (threadIdx.x * blockDim.y) + threadIdx.y;
    if (index < 256)
	temp[index] = 0;
     int x = (threadIdx.x + blockIdx.x * blockDim.x);
     int y = (threadIdx.y + blockIdx.y * blockDim.y);
    
     if (x >= width || y >= height) return; 		
     int i = y * width + x;	

  float grayPix = 0.0f;

  float r = static_cast< float >(grayImage[i]);
  float g = static_cast< float >(grayImage[(width * height) + i]);
  float b = static_cast< float >(grayImage[(2 * width * height) + i]);

  grayPix = __fadd_rn( __fadd_rn(__fadd_rn(__fmul_rn(0.3f, r),__fmul_rn(0.59f, g)), __fmul_rn(0.11f, b)), 0.5f);

  atomicAdd( &temp[static_cast< unsigned char >(grayPix)], 1);
  __syncthreads();

  if (index < 256 && temp[index] > 0){
	atomicAdd(&histogram[index],temp[index]);
  } 
}

void histogram1D(const int width, const int height, const unsigned char * inputImage, unsigned char * grayImage, unsigned int * histogram, unsigned char * histogramImage) {
	cudaError_t devRetVal = cudaSuccess; 
	NSTimer kernelTime = NSTimer("histogram", false, false);
	NSTimer memoryTime = NSTimer("memoryTime", false, false);
	NSTimer globalTime = NSTimer("globalTime", false, false);


	// start of computation
	globalTime.start();

	unsigned char *devInputImage;
	unsigned int *devHistogram;
	//unsigned int *devHistogram;

        int iImageSize = height * width * sizeof(unsigned char);

	         devRetVal = cudaMalloc((void**)&devHistogram, sizeof(unsigned int) * 256);
        if (cudaSuccess != devRetVal)
        {
                cout << "Cannot allocate memory" << endl;
                return;
        }

	 devRetVal = cudaMalloc((void**)&devInputImage, 3*iImageSize);
        if (cudaSuccess != devRetVal)
        {
                cout << "Cannot allocate memory" << endl;
                return;
        }


	memoryTime.start();
        devRetVal = cudaMemcpy(devInputImage, inputImage, 3*iImageSize, cudaMemcpyHostToDevice);

        if (cudaSuccess != devRetVal)
        {

                cout << "Cannot copy memory";
                cudaFree(devInputImage);
                return;
        }


	 if ( ( cudaMemset(devHistogram, 0, 256 * sizeof(unsigned int))) != cudaSuccess ) {
               cout << "Error in function memset." << endl;
               return ;
       }


       memoryTime.stop();

       dim3 dimBlock(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y);

       int blockWidth = static_cast<unsigned int>(ceil(width / static_cast<float>(THREADS_PER_BLOCK_X)));

       int blockHeight = static_cast<unsigned int>(ceil(height / static_cast<float>(THREADS_PER_BLOCK_Y)));


       dim3 dimGrid(blockWidth, blockHeight);

	kernelTime.start();
//	kernel_darkGray<<<dimGrid, dimBlock>>>(width, height, devInputImage, devGrayImage);
//	kernel_histogram1D<<<dimGrid, dimBlock>>>(devInputImage, devGrayImage, devHistogram, width, height);
	kernel_optimized5<<<dimGrid, dimBlock>>>( devInputImage, iImageSize, devHistogram, height, width );  

    
	
      cudaDeviceSynchronize();
        kernelTime.stop();

	 if ((devRetVal = cudaGetLastError()) != cudaSuccess)
        {
                cerr << "Uh, the kernel had some kind of issue: " << devRetVal << endl;
                cudaFree(devInputImage);
                return;
        }

	memoryTime.start();


	devRetVal = cudaMemcpy(histogram, devHistogram, 256 * sizeof(unsigned int),  cudaMemcpyDeviceToHost);
        if (cudaSuccess != devRetVal)
        {
                cout << "Cannot copy memory";
                cudaFree(devHistogram);
           

                return;
        }
//	int i;
//	for (i = 0; i < 256; i ++)
//		cout << histogram[i] << endl;
        memoryTime.stop();

	cudaFree(devInputImage);
	cudaFree(devHistogram);

	globalTime.stop();
	//end of computation	


	 // Time GFLOP/s GB/s
        cout << fixed << setprecision(6) << kernelTime.getElapsed() << endl;
        cout << fixed << setprecision(6) << memoryTime.getElapsed() << endl;
        cout << fixed << setprecision(6) << globalTime.getElapsed() << endl;




}
