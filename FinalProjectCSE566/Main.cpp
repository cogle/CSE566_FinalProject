#include <stdio.h>

#include "Common.h"
#include "Kernel.h"
#include "Screen.h"

int plate_width  = DEFAULT_PLATE_SIZE;
int plate_height = DEFAULT_PLATE_SIZE;

char title[256] = "Christopher Ogle CSE566 Project";

std::shared_ptr<Screen> screen{ nullptr };
std::shared_ptr<KernelInterface> kernel{ nullptr };

/*GLUT callbacks*/
void display();
void keyboard(unsigned char key, int x, int y);

/*Kernel callbacks*/
void releaseMem();

int main(int argc, char** argv)
{
	float * temp_array = nullptr;
	struct cudaGraphicsResource *cpr = nullptr;
	if (argc == Args::VISUAL)
	{
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
		kernel = std::make_unique<KernelInterface>(plate, temp_array);
		screen = std::make_unique<Screen>(pbo, to, plate, cpr, temp_array, kernel);


		/*
		Allocates our Temperature array and stores it on the Graphics Card's DRAM slot.
		Exists globally but must use cudaMemcpy to view it on the machine.
		Call is async, since the average, is stored in the Plate upon init.
		*/
		
		kernel->setUpTemperature(MAX_WIN_WIDTH, MAX_WIN_HEIGHT);

		/*
		Set up the GLUT interface and callbacks.
		*/
	
		screen->setupGLUT(&argc, argv);
		glutKeyboardFunc(keyboard);
		glutDisplayFunc(display);
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

void keyboard(unsigned char key, int x, int y)
{
	screen->keyboard(key,x,y);
}

void releaseMem()
{
	kernel->releaseMem();
}

