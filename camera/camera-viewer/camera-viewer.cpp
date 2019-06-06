/*
 * Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include "gstCamera.h"

#include "glDisplay.h"
#include "glTexture.h"

#include <stdio.h>
#include <signal.h>
#include <unistd.h>

#include "cudaNormalize.h"


#define DEFAULT_CAMERA -1		// -1 for onboard CSI camera, or change to index of /dev/video V4L2 camera (>=0)	
#define DEFAULT_CAMERA_WIDTH 1280	// default camera width is 1280 pixels, change this if you want a different size
#define DEFAULT_CAMERA_HEIGHT 720	// default camera height is 720 pixels, change this is you want a different size


bool signal_recieved = false;

void sig_handler(int signo)
{
	if( signo == SIGINT )
	{
		printf("received SIGINT\n");
		signal_recieved = true;
	}
}


int main( int argc, char** argv )
{
	printf("camera-viewer\n  args (%i):  ", argc);

	for( int i=0; i < argc; i++ )
		printf("%i [%s]  ", i, argv[i]);
		
	printf("\n");
	
		
	if( signal(SIGINT, sig_handler) == SIG_ERR )
		printf("\ncan't catch SIGINT\n");

	/*
	 * create the camera device
	 */
	gstCamera* camera = gstCamera::Create(DEFAULT_CAMERA_WIDTH, DEFAULT_CAMERA_HEIGHT, DEFAULT_CAMERA);
	
	if( !camera )
	{
		printf("\ncamera-viewer:  failed to initialize camera device\n");
		return 0;
	}
	
	printf("\ncamera-viewer:  successfully initialized camera device\n");
	printf("    width:  %u\n", camera->GetWidth());
	printf("   height:  %u\n", camera->GetHeight());
	printf("    depth:  %u (bpp)\n", camera->GetPixelDepth());
	

	/*
	 * create openGL window
	 */
	glDisplay* display = glDisplay::Create();
	
	if( !display )
		printf("\ncamera-viewer:  failed to create openGL display\n");
	

	/*
	 * start streaming
	 */
	if( !camera->Open() )
	{
		printf("\ncamera-viewer:  failed to open camera for streaming\n");
		return 0;
	}
	
	printf("\ncamera-viewer:  camera open for streaming\n");
	
	
	/*
	 * processing loop
	 */
	while( !signal_recieved )
	{
		// capture latest image
		float* imgRGBA = NULL;
		
		if( !camera->CaptureRGBA(&imgRGBA, 1000) )
			printf("camera-viewer:  failed to capture RGBA image\n");

		// update display
		if( display != NULL )
		{
			display->RenderOnce((float*)imgRGBA, camera->GetWidth(), camera->GetHeight());

			// update status bar
			char str[256];
			sprintf(str, "Camera Viewer (%ux%u) | %.0f FPS", camera->GetWidth(), camera->GetHeight(), display->GetFPS());
			display->SetTitle(str);	

			// check if the user quit
			if( display->IsClosed() )
				signal_recieved = true;
		}
	}
	
	printf("\ncamera-viewer:  un-initializing camera device\n");
	
	
	/*
	 * shutdown the camera device
	 */
	if( camera != NULL )
	{
		delete camera;
		camera = NULL;
	}

	if( display != NULL )
	{
		delete display;
		display = NULL;
	}
	
	printf("camera-viewer:  camera device has been un-initialized.\n");
	printf("camera-viewer:  this concludes the test of the camera device.\n");
	return 0;
}
