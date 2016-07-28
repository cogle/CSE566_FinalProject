#include <stdio.h>

#include "Common.h"
#include "Kernel.h"
#include "Screen.h"

#include <iostream>

int plate_width  = DEFAULT_PLATE_SIZE;
int plate_height = DEFAULT_PLATE_SIZE;

char title[256] = "Christopher Ogle CSE566 Project";

std::shared_ptr<Screen> screen{ nullptr };
std::shared_ptr<KernelInterface> kernel{ nullptr };

/*GLUT callbacks*/
void display();
void idle();
void keyboard(unsigned char key, int x, int y);

/*Kernel callbacks*/
void releaseMem();

int main(int argc, char** argv)
{

	if (argc == Args::VISUAL)
	{
		float * temp_array = nullptr;
		struct cudaGraphicsResource *cpr = nullptr;
		cudaError_t rc = cudaMalloc((void **)&temp_array, MAX_WIN_HEIGHT*MAX_WIN_WIDTH*sizeof(float));
		if (rc != cudaSuccess)
		{ 
			printf("Could not allocate memory: %d", rc); 
			return Returns::CUDA_MEMORY_ERROR;
		}
		/*
		Set up variables
		*/
		GLuint pbo = 0;
		GLuint to = 0;
		PlateInfo plate{ plate_width, plate_height, MAX_WIN_WIDTH, MAX_WIN_HEIGHT };
		kernel = std::make_shared<KernelInterface>(pbo, to, plate, temp_array, cpr);
		screen = std::make_shared<Screen>(pbo, to, plate, cpr, temp_array, kernel);


		/*
		Allocates our Temperature array and stores it on the Graphics Card's DRAM slot.
		Exists globally but must use cudaMemcpy to view it on the machine.
		Call is async, since the average, is stored in the Plate upon init.
		*/
		float * p1 = new float[MAX_WIN_WIDTH*MAX_WIN_HEIGHT];
		kernel->debugSetUpTemperature(p1, MAX_WIN_WIDTH, MAX_WIN_HEIGHT);
		/*
		Set up the GLUT interface and callbacks.
		*/
		
		screen->setupGLUT(&argc, argv);
		glutKeyboardFunc(keyboard);
		glutDisplayFunc(display);
		glutIdleFunc(idle);
		screen->initPixelBuffer();
		glutMainLoop();
		atexit(releaseMem);
		


		return Returns::SUCCESS;
	}
	else if (argc == Args::NO_VISUAL)
	{
		
		return Returns::SUCCESS;
	}
	usageHelp();
	return Returns::INCORRECT_ARGS;
}


void display()
{
	screen->display();
}

void idle()
{
	glutPostRedisplay();
}

void keyboard(unsigned char key, int x, int y)
{
	screen->keyboard(key,x,y);
}

void releaseMem()
{
	kernel->releaseMem();
}

