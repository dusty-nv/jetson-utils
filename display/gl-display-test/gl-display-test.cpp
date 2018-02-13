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

#include "glDisplay.h"
#include "glTexture.h"

#include <stdio.h>
#include <signal.h>
#include <unistd.h>

//#include "cudaNormalize.h"

#define TEXTURE_WIDTH  512
#define TEXTURE_HEIGHT 512


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
	printf("gl-display-test\n  args (%i):  ", argc);

	for( int i=0; i < argc; i++ )
		printf("%i [%s]  ", i, argv[i]);
		
	printf("\n");
	
		
	if( signal(SIGINT, sig_handler) == SIG_ERR )
		printf("\ncan't catch SIGINT\n");


	/*
	 * create openGL window
	 */
	glDisplay* display = glDisplay::Create();
	
	if( !display )
	{
		printf("\ngl-display:  failed to create openGL display\n");
		return 0;
	}

	
	/*
	 * initialize default test texture pattern
	 */
	const size_t texSize = TEXTURE_WIDTH * TEXTURE_HEIGHT * sizeof(float4);
	
	float4* texIn = (float4*)malloc(texSize);

	if( !texIn )
	{
		printf("failed to allocate texture initialization input\n");
		return 0;
	}
	
	for( uint32_t y=0; y < TEXTURE_HEIGHT; y++ )
		for( uint32_t x=0; x < TEXTURE_WIDTH; x++ )
			texIn[y*TEXTURE_WIDTH+x] = make_float4(0.0f, 1.0f, 1.0f, 1.0f);

		
	/*
	 * allocate openGL texture
	 */
	glTexture* texture = glTexture::Create(TEXTURE_WIDTH, TEXTURE_HEIGHT, GL_RGBA32F_ARB/*GL_RGBA8*/, texIn);

	if( !texture )
	{
		printf("gl-display:  failed to create openGL texture\n");
		return 0;
	}
	

	/*
	 * rendering loop
	 */
	while( !signal_recieved )
	{
		void* imgCPU  = NULL;
		void* imgCUDA = NULL;
		
		// rescale image pixel intensities
		/*CUDA(cudaNormalizeRGBA((float4*)imgRGBA, make_float2(0.0f, 255.0f), 
						   (float4*)imgRGBA, make_float2(0.0f, 1.0f), 
 						   camera->GetWidth(), camera->GetHeight()));*/

		// update display
		if( display != NULL )
		{
			display->UserEvents();
			display->BeginRender();

			if( texture != NULL )
			{
				/*void* tex_map = texture->MapCUDA();

				if( tex_map != NULL )
				{
					cudaMemcpy(tex_map, imgRGBA, texture->GetSize(), cudaMemcpyDeviceToDevice);
					CUDA(cudaDeviceSynchronize());
					texture->Unmap();
				}*/

				texture->Render(50,50);		
			}

			display->EndRender();
		}
	}
	
	printf("\ngl-display:  un-initializing video device\n");
	
	
	/*
	 * close the window
	 */
	if( display != NULL )
	{
		delete display;
		display = NULL;
	}
	
	printf("gl-display:  video device has been un-initialized.\n");
	printf("gl-display:  this concludes the test of the video device.\n");
	return 0;
}
