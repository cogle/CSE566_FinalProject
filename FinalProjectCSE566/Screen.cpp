#include "Screen.h"
#include "Kernel.h"

Screen::Screen(GLuint & pbo, GLuint & tex, PlateInfo & p_i, 
							 struct cudaGraphicsResource * cuda_pbo_ptr, 
							 float * temp_arr, std::shared_ptr<KernelInterface> kernel) : pixel_buffer_object(pbo),
																																			      texture_object(tex),
										  																								      plate(p_i)
{
	cuda_pbo_resource = cuda_pbo_ptr;
	temp_in = temp_arr;
	kernel_ptr = kernel;
	iteration_count = 0;
}

Screen::~Screen()
{

}

void Screen::setupGLUT(int * argc, char ** argv)
{
	glutInit(argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	//720 x 720
	glutInitWindowSize(MAX_WIN_WIDTH, MAX_WIN_WIDTH);
	glutCreateWindow(title);
	glewInit();

	/*
	Below function says that the left will start at 0 and the right most will be 720
	Then our increasing Y will be in the direction down the screen and at the very top right
	is zero.

	(0,0)                                  (720,0)
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
	(0,720)																	(720,720)

	*/

	gluOrtho2D(0, MAX_WIN_WIDTH, MAX_WIN_WIDTH, 0);

}

void Screen::initPixelBuffer()
{
	glGenBuffers(1, &pixel_buffer_object);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pixel_buffer_object);
	glBufferData(GL_PIXEL_UNPACK_BUFFER, MAX_WIN_HEIGHT*MAX_WIN_WIDTH*sizeof(GLubyte) * 4, 0, GL_STREAM_DRAW);
	glGenTextures(1, &texture_object);
	glBindTexture(GL_TEXTURE_2D, texture_object);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

	cudaError_t cudaStatus = cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pixel_buffer_object, cudaGraphicsMapFlagsWriteDiscard);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Call to cudaGraphicsGLRegisterBuffer failed.\n");
	}
}

void Screen::display()
{
	render();
	drawTexture();
	glutSwapBuffers();
}


void Screen::keyboard(unsigned char key, int x, int y)
{
	if (key == 27){ exit(0); }
	glutPostRedisplay();
}


/*Private Methods*/
void Screen::render()
{
	iteration_count++;
	uchar4 * out_data = 0;
	cudaError_t cudaStatus;
	
	cudaStatus = cudaGraphicsMapResources(1, &cuda_pbo_resource, 0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Call to cudaGraphicsMapResources failed.\n");
	}

	cudaStatus = cudaGraphicsResourceGetMappedPointer((void **)&out_data, NULL, cuda_pbo_resource);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Call to cudaGraphicsResourceGetMappedPointer failed.\n");
	}

	for (int i = 0; i < ITERATIONS_PER_RENDER; i++)
	{
		kernel_ptr->launchCalculations(out_data, MAX_WIN_WIDTH, MAX_WIN_HEIGHT);
	}
	cudaStatus = cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Call to cudaGraphicsUnmapResources failed.\n");
	}
	
}

void Screen::drawTexture()
{
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, MAX_WIN_WIDTH, MAX_WIN_HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	glEnable(GL_TEXTURE_2D);
	glBegin(GL_QUADS);
	glTexCoord2f(0.0f, 0.0f); glVertex2f(0, 0);
	glTexCoord2f(0.0f, 1.0f); glVertex2f(0, MAX_WIN_HEIGHT);
	glTexCoord2f(1.0f, 1.0f); glVertex2f(MAX_WIN_WIDTH, MAX_WIN_HEIGHT);
	glTexCoord2f(1.0f, 0.0f); glVertex2f(MAX_WIN_WIDTH, 0);
	glEnd();
	glDisable(GL_TEXTURE_2D);
}

