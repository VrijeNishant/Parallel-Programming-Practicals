#include <Timer.hpp>
#include <iostream>
#include <iomanip>
#define SPECTRUM 3
#define BLOCK_THREAD_X 16//16
#define BLOCK_THREAD_Y 16//16



using LOFAR::NSTimer;
using std::cout;
using std::cerr;
using std::endl;
using std::fixed;
using std::setprecision;

__constant__ float filter[] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f, 1.0f, 1.0f, 2.0f, 3.0f, 2.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};


__global__ void triangularSmooth_parallel(unsigned char *input,unsigned char *smooth,const int width, const int height) {

	int z;
        int x = blockIdx.y * blockDim.y + threadIdx.y;
        int y = blockIdx.x * blockDim.x + threadIdx.x;
        int current_position = x*width + y;
        if (x < height && y < width)
        {
                //for (int z = 0; z < SPECTRUM; z++)  
		
	// Loop unrolling for z = 0
		{
                        z = 0;
		        unsigned int filterItem = 0;
                        float filterSum = 0.0f;
                        float smoothPix = 0.0f;

                        for ( int fy = x - 2; fy < x + 3; fy++ ) {
                                if ( fy < 0 ) {
                                        filterItem += 5;
                                        continue;
                                }
                                else if ( fy == height ) {
                                        break;
                                }

       //fx loop unrolling                    //    for ( int fx = y - 2; fx < y + 3; fx++ ) 
				{int fx = y-2;
                                        if ( (fx < 0) || (fx >= width) ) {
                                                filterItem++;
                                                
                                        }
					
					else{

                                        smoothPix += static_cast< float >(input[(z * width * height) + (fy * width) + fx]) * filter[filterItem];
                                        filterSum += filter[filterItem];
                                        filterItem++; }
                                } 
                        


			{int fx = y-1;
                                        if ( (fx < 0) || (fx >= width) ) {
                                                filterItem++;
                                                
                                        }

                                        else{

                                        smoothPix += static_cast< float >(input[(z * width * height) + (fy * width) + fx]) * filter[filterItem];
                                        filterSum += filter[filterItem];
                                        filterItem++; }
                                } 

			{int fx = y;
                                        if ( (fx < 0) || (fx >= width) ) {
                                                filterItem++;
                                                
                                        }

                                        else{

                                        smoothPix += static_cast< float >(input[(z * width * height) + (fy * width) + fx]) * filter[filterItem];
                                        filterSum += filter[filterItem];
                                        filterItem++; }
                                } 


			{int fx = y + 1;
                                        if ( (fx < 0) || (fx >= width) ) {
                                                filterItem++;
                                                
                                        }

                                        else{

                                        smoothPix += static_cast< float >(input[(z * width * height) + (fy * width) + fx]) * filter[filterItem];
                                        filterSum += filter[filterItem];
                                        filterItem++; }
                                } 
	

			{int fx = y+2;
                                        if ( (fx < 0) || (fx >= width) ) {
                                                filterItem++;
                                                
                                        }

                                        else{

                                        smoothPix += static_cast< float >(input[(z * width * height) + (fy * width) + fx]) * filter[filterItem];
                                        filterSum += filter[filterItem];
                                        filterItem++; }
                                } 

			}

                        smoothPix /= filterSum;
                        smooth[(z * width * height) + current_position] = static_cast< unsigned char >(smoothPix + 0.5f);
                }

		 // Loop unrolling for z = 1
                {
                        z = 1;
                        unsigned int filterItem = 0;
                        float filterSum = 0.0f;
                        float smoothPix = 0.0f;

                        for ( int fy = x - 2; fy < x + 3; fy++ ) {
                                if ( fy < 0 ) {
                                        filterItem += 5;
                                        continue;
                                }
                                else if ( fy == height ) {
                                        break;
                                }

       //fx loop unrolling                    //    for ( int fx = y - 2; fx < y + 3; fx++ ) 
                                {int fx = y-2;
                                        if ( (fx < 0) || (fx >= width) ) {
                                                filterItem++;

                                        }

                                        else{

                                        smoothPix += static_cast< float >(input[(z * width * height) + (fy * width) + fx]) * filter[filterItem];
                                        filterSum += filter[filterItem];
                                        filterItem++; }
                                }



                        {int fx = y-1;
                                        if ( (fx < 0) || (fx >= width) ) {
                                                filterItem++;

                                        }
					else{

                                        smoothPix += static_cast< float >(input[(z * width * height) + (fy * width) + fx]) * filter[filterItem];
                                        filterSum += filter[filterItem];
                                        filterItem++; }
                                }

                        {int fx = y;
                                        if ( (fx < 0) || (fx >= width) ) {
                                                filterItem++;

                                        }

                                        else{

                                        smoothPix += static_cast< float >(input[(z * width * height) + (fy * width) + fx]) * filter[filterItem];
                                        filterSum += filter[filterItem];
                                        filterItem++; }
                                }


                        {int fx = y + 1;
                                        if ( (fx < 0) || (fx >= width) ) {
                                                filterItem++;

                                        }

                                        else{

                                        smoothPix += static_cast< float >(input[(z * width * height) + (fy * width) + fx]) * filter[filterItem];
                                        filterSum += filter[filterItem];
                                        filterItem++; }
                                }

			 {int fx = y+2;
                                        if ( (fx < 0) || (fx >= width) ) {
                                                filterItem++;

                                        }

                                        else{

                                        smoothPix += static_cast< float >(input[(z * width * height) + (fy * width) + fx]) * filter[filterItem];
                                        filterSum += filter[filterItem];
                                        filterItem++; }
                                }

                        }

                        smoothPix /= filterSum;
                        smooth[(z * width * height) + current_position] = static_cast< unsigned char >(smoothPix + 0.5f);
                }

		
                 // Loop unrolling for z = 2
                {
                        z = 2;
                        unsigned int filterItem = 0;
                        float filterSum = 0.0f;
                        float smoothPix = 0.0f;

                        for ( int fy = x - 2; fy < x + 3; fy++ ) {
                                if ( fy < 0 ) {
                                        filterItem += 5;
                                        continue;
                                }
                                else if ( fy == height ) {
                                        break;
                                }

       //fx loop unrolling                    //    for ( int fx = y - 2; fx < y + 3; fx++ ) 
                                {int fx = y-2;
                                        if ( (fx < 0) || (fx >= width) ) {
                                                filterItem++;

                                        }

                                        else{

                                        smoothPix += static_cast< float >(input[(z * width * height) + (fy * width) + fx]) * filter[filterItem];
                                        filterSum += filter[filterItem];
                                        filterItem++; }
                                }

				 {int fx = y-1;
                                        if ( (fx < 0) || (fx >= width) ) {
                                                filterItem++;

                                        }
                                        else{

                                        smoothPix += static_cast< float >(input[(z * width * height) + (fy * width) + fx]) * filter[filterItem];
                                        filterSum += filter[filterItem];
                                        filterItem++; }
                                }

                        {int fx = y;
                                        if ( (fx < 0) || (fx >= width) ) {
                                                filterItem++;

                                        }

                                        else{

                                        smoothPix += static_cast< float >(input[(z * width * height) + (fy * width) + fx]) * filter[filterItem];
                                        filterSum += filter[filterItem];
                                        filterItem++; }
                                }


                        {int fx = y + 1;
                                        if ( (fx < 0) || (fx >= width) ) {
                                                filterItem++;

                                        }
			  else{

                                        smoothPix += static_cast< float >(input[(z * width * height) + (fy * width) + fx]) * filter[filterItem];
                                        filterSum += filter[filterItem];
                                        filterItem++; }
                                }

                         {int fx = y+2;
                                        if ( (fx < 0) || (fx >= width) ) {
                                                filterItem++;

                                        }

                                        else{

                                        smoothPix += static_cast< float >(input[(z * width * height) + (fy * width) + fx]) * filter[filterItem];
                                        filterSum += filter[filterItem];
                                        filterItem++; }
                                }

                        }

                        smoothPix /= filterSum;
                        smooth[(z * width * height) + current_position] = static_cast< unsigned char >(smoothPix + 0.5f);
                }


        }

}

int triangularSmooth(const int width, const int height, const int spectrum, unsigned char * inputImage, unsigned char * smoothImage) {
        
	NSTimer kernelTime = NSTimer("smooth", false, false);
	NSTimer memoryTime = NSTimer("memoryTime", false, false);
	NSTimer globalTime = NSTimer("globalTime", false, false);


	// start of computation
	globalTime.start();

        cudaError_t devRetVal = cudaSuccess;
        const unsigned int DIM = width*height;
        unsigned char *dev_smoothImage=0;
        unsigned char *dev_inputImage =0;

        if ( (devRetVal = cudaMalloc(reinterpret_cast< void ** >(&dev_inputImage), 3*DIM * sizeof(unsigned char))) != cudaSuccess ) {
                cerr << "Impossible to allocate device memory for input image" << endl;
                return 1;
        }

	memoryTime.start();

        if ( (devRetVal = cudaMemcpy(dev_inputImage, reinterpret_cast< void * >(inputImage),3*DIM * sizeof(unsigned char), cudaMemcpyHostToDevice)) != cudaSuccess ) {
                cerr << "Impossible to copy dev for input image" << endl;
                return 1;
        }

	memoryTime.stop();
        if ( (devRetVal = cudaMalloc(reinterpret_cast< void ** >(&dev_smoothImage), 3*DIM * sizeof(unsigned char))) != cudaSuccess ) {
                cerr << "Impossible to allocate device memory for dev_smoothImage." << endl;
                return 1;
        }

        dim3  dimBlock = dim3(BLOCK_THREAD_X,BLOCK_THREAD_Y);
        unsigned int blockWidth = static_cast<unsigned int>(ceil(width / static_cast<float>(BLOCK_THREAD_X)));
        unsigned int  blockHeight = static_cast<unsigned int>(ceil(height / static_cast<float>(BLOCK_THREAD_Y)));
        dim3 dimGrid = dim3(blockWidth,blockHeight);


        kernelTime.start();
        triangularSmooth_parallel<<<dimGrid, dimBlock>>>(dev_inputImage, dev_smoothImage, width, height);
        cudaDeviceSynchronize();
        kernelTime.stop();

	memoryTime.start();

        if ( (devRetVal = cudaMemcpy(reinterpret_cast< void * >(smoothImage),dev_smoothImage, 3*DIM * sizeof(unsigned char), cudaMemcpyDeviceToHost)) != cudaSuccess ) {
                cerr << "Impossible to copy smooth image to host." << endl;
                return 1;
        }

	memoryTime.stop();
        cudaFree(dev_inputImage);
        cudaFree(dev_smoothImage);

	globalTime.stop();
	// end of computation

        // Time GFLOP/s GB/s
        cout << fixed << setprecision(6) << kernelTime.getElapsed() << endl;
	cout << fixed << setprecision(6) << memoryTime.getElapsed() << endl;
	cout << fixed << setprecision(6) << globalTime.getElapsed() << endl;


}
                                                                                                                                                     




                                                                                                                                         
