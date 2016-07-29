#include "Kernel.h"

__device__ 
float calcIntensity(float cur_temp, float min_temp, float max_temp)
{
	if (min_temp == max_temp){ return 2.0f; }
	return 2.0f*(cur_temp - min_temp) / (max_temp - min_temp);
}

__device__
int getFlatIndex(int col, int row, int max_width)
{
	return row*max_width + col;
}

__device__
int clipTemperature(int temp)
{
	if (temp < 0){ return 0; }
	if (temp > 255){ return 255; }
	return temp;
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
	//Center Value
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
	
	/*
	Set the default value to be black.
	*/

	//R
	out[idx].x = 0;
	//G
	out[idx].y = 0;
	//B
	out[idx].z = 0;
	
	//Alpha
	out[idx].w = 255;


	const int shared_col_size = blockDim.x + 2; //Width


	const int shared_array_col = threadIdx.x + 1;
	const int shared_array_row = threadIdx.y + 1;

	const int array_index = getFlatIndex(shared_array_col, shared_array_row, shared_col_size);

	shared_temp[array_index] = temp_data[idx];
	/*
	If first row of the thread, it is responsible for assigning the
	first and the last row
	*/
	if (threadIdx.x == 0)
	{
		//Square to the left of the first entry.
		shared_temp[getFlatIndex(shared_array_col - 1, shared_array_row, shared_col_size)] = temp_data[getFlatIndex(col - 1, row, max_width)];
		shared_temp[getFlatIndex(shared_array_col + blockDim.x, shared_array_row, shared_col_size)] = temp_data[getFlatIndex(col + blockDim.x, row, max_width)];
	}
	if (threadIdx.y == 0)
	{
		shared_temp[getFlatIndex(shared_array_col, shared_array_row - 1, shared_col_size)] = temp_data[getFlatIndex(col, row - 1, max_width)];
		shared_temp[getFlatIndex(shared_array_col, shared_array_row + blockDim.y, shared_col_size)] = temp_data[getFlatIndex(col, row + +blockDim.y, max_width)];
	}

	//Top
	if (row == plate.plate_top_left_y && (col >= plate.plate_top_left_x && col <= plate.plate_top_right_x))
	{
		float intensity = calcIntensity(plate.top_temp,plate.min_temp, plate.max_temp);

		int red = clipTemperature((int)255.0f * (intensity - 1.0f));
		int blue = clipTemperature((int)255.0f * (1.0f - intensity));
		int green = clipTemperature((int) 255 - blue - red);

		out[idx].x = red;
		out[idx].y = green;
		out[idx].z = blue;

		return;
	}
	//Bot
	else if (row == plate.plate_bot_left_y && (col >= plate.plate_bot_left_x && col <= plate.plate_bot_right_x))
	{
		float intensity = calcIntensity(plate.bot_temp, plate.min_temp, plate.max_temp);

		int red = clipTemperature((int)255.0f * (intensity - 1.0f));
		int blue = clipTemperature((int)255.0f * (1.0f - intensity));
		int green = clipTemperature(255 - blue - red);

		out[idx].x = red;
		out[idx].y = green;
		out[idx].z = blue;
		return;
	}
	//Left
	else if (col == plate.plate_top_left_x && (row > plate.plate_top_left_y && row < plate.plate_bot_left_y))
	{

		float intensity = calcIntensity(plate.left_temp, plate.min_temp, plate.max_temp);

		int red = clipTemperature((int)255.0f * (intensity - 1.0f));
		int blue = clipTemperature((int)255.0f * (1.0f - intensity));
		int green = clipTemperature(255 - blue - red);

		out[idx].x = red;
		out[idx].y = green;
		out[idx].z = blue;

		return;
	}
	//Right
	else if (col == plate.plate_top_right_x && (row > plate.plate_top_right_y && row < plate.plate_bot_right_y))
	{
		
		float intensity = calcIntensity(plate.right_temp, plate.min_temp, plate.max_temp);

		int red = clipTemperature((int)255.0f * (intensity - 1.0f));
		int blue = clipTemperature((int)255.0f * (1.0f - intensity));
		int green = clipTemperature(255 - blue - red);

		out[idx].x = red;
		out[idx].y = green;
		out[idx].z = blue;
		
		return;
	}
	//Center Value
	else if ((col > plate.plate_top_left_x && col < plate.plate_top_right_x) && (row > plate.plate_top_right_y && row < plate.plate_bot_right_y))
	{




		__syncthreads();
	
		
		float top_piece = shared_temp[getFlatIndex(shared_array_col, shared_array_row + 1, shared_col_size)];
		float right_piece = shared_temp[getFlatIndex(shared_array_col + 1, shared_array_row, shared_col_size)];
		float bot_piece = shared_temp[getFlatIndex(shared_array_col, shared_array_row - 1, shared_col_size)];
		float left_piece = shared_temp[getFlatIndex(shared_array_col - 1, shared_array_row, shared_col_size)];
	

		float new_temp = .25f*(top_piece+right_piece+bot_piece+left_piece);
		temp_data[idx] = new_temp;
		float intensity = calcIntensity(temp_data[idx], plate.min_temp, plate.max_temp);
		int red = clipTemperature((int)255.0f * (intensity - 1.0f));
		int blue = clipTemperature((int)255.0f * (1.0f - intensity));
		int green = clipTemperature(255 - blue - red);

		out[idx].x = red;
		out[idx].y = green;
		out[idx].z = blue;
		


		return;
	}
	else
	{
		return;
	}
}

KernelInterface::KernelInterface(GLuint & pbo, GLuint & tex, PlateInfo & p_i, 
																 float * temperature_array, struct cudaGraphicsResource *cpr) : plate(p_i),
																																																pixel_buffer_object(pbo), 
																																																texture_object(tex)
{
	cuda_pbo_resource = cpr;
	temp_in = temperature_array;
}

KernelInterface::~KernelInterface()
{

}

/*
Debug Method, Debugging this stuff is a nightmare.
*/
void KernelInterface::debugSetUpTemperature(float  temp_out[], int max_width, int max_height)
{
	//32x32
	const dim3 blockSize(THREADS_X, THREADS_Y);
	const dim3 gridSize(divBlocksInGrid(max_width, THREADS_X), divBlocksInGrid(max_height, THREADS_Y));

	resetTemp <<<gridSize, blockSize>>>(temp_in, max_width, max_height, plate);
	cudaDeviceSynchronize();
	cudaError_t cudaStatus;
	cudaStatus = cudaMemcpy(temp_out, temp_in,  max_height*max_width*sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Call to Memcpy failed.\n");
	}
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
}

/*
Release all the memory.
*/
void KernelInterface::releaseMem()
{
	if (pixel_buffer_object)
	{
		cudaGraphicsUnregisterResource(cuda_pbo_resource);
		glDeleteBuffers(1, &pixel_buffer_object);
		glDeleteBuffers(1, &texture_object);
	}
	cudaFree(temp_in);

}

/*
Launches the calculations for the Heat Plane
*/
void KernelInterface::launchCalculations(uchar4 * out, int max_width, int max_height)
{
	const dim3 blockSize(THREADS_X, THREADS_Y);
	const dim3 gridSize(divBlocksInGrid(max_width, THREADS_X), divBlocksInGrid(max_height, THREADS_Y));
	const size_t  shared_array_size = (THREADS_X + 2)*(THREADS_Y + 2)*sizeof(float);

	calcTemp <<<gridSize, blockSize, shared_array_size>>>(out, temp_in, max_width, max_height, plate);
}
