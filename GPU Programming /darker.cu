
#include <Timer.hpp>
#include <iostream>
#include <iomanip>

using LOFAR::NSTimer;
using std::cout;
using std::cerr;
using std::endl;
using std::fixed;
using std::setprecision;

#define        OFFLOAD_KERNEL_DARK       1

#define        THREADS_PER_BLOCK        512
#define        THREADS_PER_BLOCK_X        32
#define        THREADS_PER_BLOCK_Y        16







__global__ void
kernel_darkGray(
	const int width,
	const int height,
	unsigned char *inputImage,
	unsigned char *darkGrayImage
	)
{
  int row;
  int col;

  row =(blockIdx.y * blockDim.y + threadIdx.y);
  col = (blockIdx.x * blockDim.x + threadIdx.x);

  __syncthreads(); 

  int index = (row * width) + col;


  if (index >= (width * height))
  {
	 
    return;
  }

 float grayPix = 0.0f;
 float r = static_cast< float >(inputImage[index]);
 float g = static_cast< float >(inputImage[(width * height) + index]);
 float b = static_cast< float >(inputImage[(2 * width * height) + index]);

    grayPix = __fadd_rn(__fadd_rn(__fmul_rn(0.3f, r),__fmul_rn(0.59f, g)), __fmul_rn(0.11f, b));
    grayPix = __fadd_rn(__fmul_rn(grayPix, 0.6f), 0.5f);   
  //grayPix = ((0.3f * r) + (0.59f * g) + (0.11f * b));
  //grayPix = (grayPix * 0.6f) + 0.5f;

  darkGrayImage[index] = static_cast< unsigned char >(grayPix);
}
  

void darkGray(const int width, const int height, const unsigned char * inputImage, unsigned char * darkGrayImage) {

	cudaError_t devRetVal = cudaSuccess;
	NSTimer kernelTime = NSTimer("darker", false, false);
	NSTimer memoryTime = NSTimer("memoryTime", false, false);
	NSTimer globalTime = NSTimer("globalTime", false, false);


        #if OFFLOAD_KERNEL_DARK 

	// start of computation
	globalTime.start();	

	unsigned char *devInputImage;
	unsigned char *devDarkGrayImage;

	int iImageSize = height * width * sizeof(unsigned char);


	devRetVal = cudaMalloc((void**)&devInputImage, iImageSize * 3);
	if (cudaSuccess != devRetVal)
	{
		cout << "Cannot allocate memory" << endl;
		return;
	}
	
	devRetVal = cudaMalloc((void**)&devDarkGrayImage, iImageSize);
	if (cudaSuccess != devRetVal)
	{
		cout << "Cannot allocate memory" << endl;
		cudaFree(devInputImage);
		return;
	}

	memoryTime.start();
	devRetVal = cudaMemcpy(devInputImage, inputImage, iImageSize * 3, cudaMemcpyHostToDevice);

	if (cudaSuccess != devRetVal)
	{

		cout << "Cannot copy memory";
		cudaFree(devInputImage);
		cudaFree(devDarkGrayImage);
		return;
	}
	
	memoryTime.stop();

	dim3 dimBlock(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y);

	int blockWidth = static_cast<unsigned int>(ceil(width / static_cast<float>(THREADS_PER_BLOCK_X)));

	int blockHeight = static_cast<unsigned int>(ceil(height / static_cast<float>(THREADS_PER_BLOCK_Y)));


	dim3 dimGrid(blockWidth, blockHeight);

	kernelTime.start();
	kernel_darkGray<<<dimGrid, dimBlock>>>(width, height, devInputImage, devDarkGrayImage);
	cudaDeviceSynchronize();
	kernelTime.stop();

	if ((devRetVal = cudaGetLastError()) != cudaSuccess)
	{
		cerr << "Uh, the kernel had some kind of issue: " << devRetVal << endl;
		cudaFree(devInputImage);
		cudaFree(devDarkGrayImage);
		return;
	}


	memoryTime.start();
	devRetVal = cudaMemcpy(darkGrayImage, devDarkGrayImage, iImageSize, cudaMemcpyDeviceToHost);
	if (cudaSuccess != devRetVal)
	{
		cout << "Cannot copy memory";
		cudaFree(devInputImage);
		cudaFree(devDarkGrayImage);
		
		return;
	}
	
	memoryTime.stop();

	cudaFree(devInputImage);
	cudaFree(devDarkGrayImage);


	globalTime.stop();
	// end of computation


	#else
	kernelTime.start();
	for (int y =0; y< height; ++y)
	{
		for(int x = 0; x < width; ++x)
			{
			float grayPix = 0.0f;	
			float r = static_cast< float >(inputImage[(y * width) + x]);
			float g = static_cast< float >(inputImage[(width * height) + (y * width) + x]);
			float b = static_cast< float >(inputImage[(2 * width * height) + (y * width) + x]);

			grayPix = ((0.3f * r) + (0.59f * g) + (0.11f * b));
			grayPix = (grayPix * 0.6f) + 0.5f;

			darkGrayImage[(y * width) + x] = static_cast< unsigned char >(grayPix);
		}
	}
	
	kernelTime.stop();
	#endif
	
	// Time GFLOP/s GB/s
	 cout << fixed << setprecision(6) << kernelTime.getElapsed() << endl;
        cout << fixed << setprecision(6) << memoryTime.getElapsed() << endl;
        cout << fixed << setprecision(6) << globalTime.getElapsed() << endl;	




}
