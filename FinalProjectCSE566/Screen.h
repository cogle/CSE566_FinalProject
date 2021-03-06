#pragma once

#include <iostream>
#include "Common.h"

class KernelInterface;

class Screen
{
public:
	Screen(GLuint & pbo, GLuint & tex, PlateInfo & plate, struct cudaGraphicsResource * cuda_pbo_ptr, float * temp_arr, std::shared_ptr<KernelInterface> kernel_ptr);
	~Screen();

	/*GLUT setup method*/
	void setupGLUT(int * argc, char ** argv);
	void initPixelBuffer();

	/*GLUT callbacks*/
	void display();
	void keyboard(unsigned char key, int x, int y);


private:
	/*Shared Variables*/
	GLuint & pixel_buffer_object;
	GLuint & texture_object;
	struct cudaGraphicsResource *cuda_pbo_resource;
	PlateInfo & plate;
	std::shared_ptr<KernelInterface> kernel_ptr;
	float * temp_in;

	/*Private Functions*/
	void render();
	void drawTexture();

	unsigned int iteration_count;
};