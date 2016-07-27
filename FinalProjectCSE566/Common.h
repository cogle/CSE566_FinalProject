#pragma once

#include <stdio.h>
#include <memory>

/*
The code was developed using the windows enviroment in order to use
OpenGL I needed to include the following files.
*/
#include "OpenGLDeps\glew\glew.h"
#include "OpenGLDeps\freeglut\freeglut.h"

/*
Include the CUDA libraries
*/
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_gl_interop.h"

/*
Define the maximum width of the GUI window
*/

#define MAX_WIN_HEIGHT 840
#define MAX_WIN_WIDTH  840
#define DEFAULT_PLATE_SIZE 503
#define ITERATIONS_PER_RENDER 18
/*
Debugging variables

#define MAX_WIN_HEIGHT 10
#define MAX_WIN_WIDTH  10
#define DEFAULT_PLATE_SIZE 5
*/


enum Args    {VISUAL = 1, NO_VISUAL = 6};
enum Returns {SUCCESS, INCORRECT_ARGS, CUDA_MEMORY_ERROR};

extern char title[256];
extern int plate_width, plate_height;

struct PlateInfo
{
	/*
	This constructor is called when the GUI version of the code is ran.
	*/
	PlateInfo(int plate_width, int plate_height, int max_width, int max_height) : top_temp(0.0), left_temp(100.0),
		                                                                            bot_temp(100.0), right_temp(100.0),
																																								plate_width(plate_width), plate_height(plate_height)
	{
		initPlateBoundariesGUI(plate_width, plate_height, max_width, max_height);
		average_temp = (top_temp*plate_width + bot_temp*plate_width + left_temp*(plate_height - 2.0f) + right_temp*(plate_height - 2)) / (2.0f*plate_width + 2.0f*(plate_height - 2));
	}
	/*
	PlateInfo(float t_t, float l_t, float b_t, float r_t) : top_temp(t_t), left_temp(l_t), bot_temp(b_t), right_temp(r_t)
	{

	}
	*/
	void initPlateBoundariesGUI(int plate_width, int plate_height, int max_width, int max_height)
	{
		/*
		Find the center of our GUI, sub by one for array index
		*/
		int gui_center_x = (max_width / 2);
		int gui_center_y = (max_height / 2);

		/*
		Find the offset for each corner. Sub one for the center.
		*/
		int diff_x = gui_center_x - (plate_width / 2) - 1;
		int diff_y = gui_center_y - (plate_height / 2) - 1;

		int gui_array_center_x = gui_center_x - 1;
		int gui_array_center_y = gui_center_y - 1;

		/*
		Assign the plate bounderies
		*/

		/*
		Top Left                               Top Right
		*---------------------------------------*
		|																				|
		|																				|
		|																				|
		|																				|
		|																				|
		|																				|
		|																				|
		|																				|
		|																				|
		|																				|
		|																				|
		|																				|
		|																				|
		*---------------------------------------*
		Bot Left																Bot Right
		*/
		plate_top_left_x = gui_array_center_x - diff_x;
		plate_top_left_y = gui_array_center_y - diff_y;

		plate_top_right_x = gui_array_center_x + diff_x;
		plate_top_right_y = gui_array_center_y - diff_y;

		plate_bot_left_x = gui_array_center_x - diff_x;
		plate_bot_left_y = gui_array_center_y + diff_y;

		plate_bot_right_x = gui_array_center_x + diff_x;
		plate_bot_right_y = gui_array_center_y + diff_y;
	}


	int plate_top_left_x, plate_top_left_y;
	int plate_top_right_x, plate_top_right_y;
	int plate_bot_left_x, plate_bot_left_y;
	int plate_bot_right_x, plate_bot_right_y;
	int plate_width, plate_height;
	float top_temp, left_temp, bot_temp, right_temp, average_temp;

};



inline void usageHelp()
{
	printf("Program usage\n");
	printf("In order to run the program without the visual interface(large plates) enter the following\n");
	printf("ExecutableName <Plate Size> <Top Temp> <Right Temp> <Bot Temp> <Left Temp>\n");
}

