#include "Kernel.h"

__device__
int getFlatIndex(int col, int row, int max_width)
{
	return row*max_width + col;
}

__global__
void resetTemp(float * temp_in, int max_width, int max_height, PlateInfo plate)
{
	const int col = blockIdx.x*blockDim.x + threadIdx.x;
	const int row = blockIdx.y*blockDim.y + threadIdx.y;

	/*Initial check to ensure that the index is inbounds*/
	if ((col >= max_width) || (row >= max_height)){ return; }
	
	//Top
	if (row == plate.plate_top_left_y && (col >= plate.plate_top_left_x && col <= plate.plate_top_right_x))
	{
		temp_in[getFlatIndex(col,row,max_width)] = plate.top_temp;
		return;
	}
	//Bot
	else if (row == plate.plate_bot_left_y && (col >= plate.plate_bot_left_x && col <= plate.plate_bot_right_x))
	{
		temp_in[getFlatIndex(col, row, max_width)] = plate.bot_temp;
		return;
	}
	//Left
	else if (col == plate.plate_top_left_x && (row > plate.plate_top_left_y && row < plate.plate_bot_left_y))
	{
		temp_in[getFlatIndex(col, row, max_width)] = plate.left_temp;
		return;
	}
	//Right
	else if (col == plate.plate_top_right_x && (row > plate.plate_top_right_y && row < plate.plate_bot_right_y))
	{
		temp_in[getFlatIndex(col, row, max_width)] = plate.right_temp;
		return;
	}
	else if ((col > plate.plate_top_left_x && col < plate.plate_top_right_x) && (row > plate.plate_top_right_y && row < plate.plate_bot_right_y))
	{
		temp_in[getFlatIndex(col, row, max_width)] = plate.average_temp;
		return;
	}

	temp_in[getFlatIndex(col, row, max_width)] = 0.0;
}

__global__
void calcTemp(uchar4 * out, float * temp_data ,int max_width, int max_height, PlateInfo plate)
{
	extern __shared__ float shared_temp[];
	const int col = blockIdx.x*blockDim.x + threadIdx.x;
	const int row = blockIdx.y*blockDim.y + threadIdx.y;

	/*Initial check to ensure that the index is inbounds*/
	if ((col >= max_width) || (row >= max_height)){ return; }
	const int idx = getFlatIndex(col, row, max_width);
	
	//R
	out[idx].x = 0;
	//G
	out[idx].y = 0;
	//B
	out[idx].z = 0;
	
	//Alpha
	out[idx].w = 0;

}

KernelInterface::KernelInterface(PlateInfo & p_i, float * temperature_array) : plate(p_i)
{
	temp_in = temperature_array;
}

KernelInterface::~KernelInterface()
{

}

/*
Debug Method, Debugging this stuff is a nightmare.
*/
void KernelInterface::debugSetUpTemperature(float * temp_out, int max_width, int max_height)
{
	//32x32
	const dim3 blockSize(THREADS_X, THREADS_Y);
	const dim3 gridSize(divBlocksInGrid(max_width, THREADS_X), divBlocksInGrid(max_height, THREADS_Y));

	resetTemp <<<gridSize, blockSize>>>(temp_in, max_width, max_height, plate);
	cudaDeviceSynchronize();
	cudaMemcpy(temp_out, temp_in,  max_height*max_width*sizeof(float), cudaMemcpyDeviceToHost);
}

/*
Calculates how many blocks should exist in a grid based upon then number
of threads that are in our program.
*/
int KernelInterface::divBlocksInGrid(int screenSize, int numThreads)
{
	return (screenSize + numThreads - 1) / numThreads;
}

/*
Release Method
*/
void KernelInterface::setUpTemperature(int max_width, int max_height)
{
	const dim3 blockSize(THREADS_X, THREADS_Y);
	const dim3 gridSize(divBlocksInGrid(max_width, THREADS_X), divBlocksInGrid(max_height, THREADS_Y));
	resetTemp <<<gridSize, blockSize>>>(temp_in, max_width, max_height, plate);
	//cudaDeviceSynchronize();
}

void KernelInterface::releaseMem()
{
	cudaFree(temp_in);

}

void KernelInterface::launchCalculations(uchar4 * out, int max_width, int max_height)
{
	const dim3 blockSize(THREADS_X, THREADS_Y);
	const dim3 gridSize(divBlocksInGrid(max_width, THREADS_X), divBlocksInGrid(max_height, THREADS_Y));
	const size_t  shared_array_size = (plate.plate_width*plate.plate_height) - 4;
	calcTemp <<<gridSize, blockSize, shared_array_size>>>(out, temp_in, max_width, max_height, plate);
}
