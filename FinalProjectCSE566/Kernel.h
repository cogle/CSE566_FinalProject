#pragma once

#include "Common.h"
#include <device_functions.h>

#define THREADS_X 32
#define THREADS_Y 32

class KernelInterface
{
public:
	KernelInterface(GLuint & pbo, GLuint & tex, PlateInfo & p_i, float * temperature_array, struct cudaGraphicsResource *cpr);
	~KernelInterface();

	/*
	Debugging method in order to ensure that the plate is being correctly set up.
	*/
	void debugSetUpTemperature(float * temp_out, int max_width, int max_height);

	/*
	Runtime method of the plate.
	*/
	void setUpTemperature(int max_width, int max_height);
	void releaseMem();
	void launchCalculations(uchar4 * out, int width, int height);

private: 
	/*Private Variables*/
	float * temp_in;
	PlateInfo & plate;
	GLuint & pixel_buffer_object;
	GLuint & texture_object;
	struct cudaGraphicsResource *cuda_pbo_resource;

	/*Private Functions*/
	int  divBlocksInGrid(int screenSize, int numThreads);
};









