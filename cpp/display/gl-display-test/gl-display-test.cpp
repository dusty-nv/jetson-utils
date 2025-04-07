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
#include "glBuffer.h"
#include "glCamera.h"

#include "cudaFont.h"
#include "cudaNormalize.h"
#include "cudaInteropKernels.h"

#include "timespec.h"

#include <stdio.h>
#include <signal.h>
#include <unistd.h>


#define TEXTURE_WIDTH 768
#define TEXTURE_HEIGHT 64
#define TEXTURE_OFFSET 30

#define GRID_N          128
#define GRID_POINTS     (GRID_N * GRID_N)
#define GRID_WORLD_SIZE 10.0f


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
	/*
	 * register signal handler (Ctrl+C)
	 */
	if( signal(SIGINT, sig_handler) == SIG_ERR )
		printf("can't catch SIGINT\n");


	/*
	 * create openGL window
	 */
	glDisplay* display = glDisplay::Create("NVIDIA OpenGL/CUDA Interoperability Test");
	
	if( !display )
	{
		printf("gl-display-test:  failed to create openGL display\n");
		return 0;
	}


	/*
	 * allocate openGL texture
	 */
	glTexture* texture = glTexture::Create(TEXTURE_WIDTH, TEXTURE_HEIGHT, GL_RGBA32F, NULL);

	if( !texture )
	{
		printf("gl-display-test:  failed to create openGL texture\n");
		return 0;
	}
	

	/*
	 * create font
	 */
	cudaFont* font = cudaFont::Create();
	
	if( !font )
	{
		printf("gl-display-test:  failed to create cudaFont object\n");
		return 0;
	}


	/*
	 * create 3D camera
	 */
	glCamera* camera = glCamera::Create(glCamera::LookAt);

	if( !camera )
	{
		printf("gl-display-test:  failed to create glCamera object\n");
		return 0;
	}

	camera->SetEye(0.0f, GRID_WORLD_SIZE, GRID_WORLD_SIZE);
	camera->StoreDefaults();


	/*
	 * create vertex buffer
	 */
	glBuffer* buffer = glBuffer::Create(GL_VERTEX_BUFFER, GRID_POINTS * sizeof(PointVertex), NULL, GL_DYNAMIC_DRAW);

	if( !buffer )
	{
		printf("gl-display-test:  failed to create glBuffer object\n");
		return 0;
	}


	/*
	 * rendering loop
	 */
	while( !signal_recieved && display->IsOpen() )
	{
		display->BeginRender();

		display->RenderRect( 10, 100, 200, 100, 0.9f, 0.0f, 0.2f);
		display->RenderRect(210, 100, 200, 100, 0.0f, 0.9f, 0.4f);
		display->RenderRect(410, 100, 200, 100, 0.0f, 0.4f, 0.9f);
		
		// draw point buffer
		PointVertex* points = (PointVertex*)buffer->Map(GL_MAP_CUDA, GL_WRITE_DISCARD);

		if( points != NULL )
		{
			// animate the points in CUDA
			CUDA(cudaGeneratePointGrid(points, GRID_N, GRID_WORLD_SIZE, apptime()));
			CUDA(cudaDeviceSynchronize());

			buffer->Unmap();

			// change the viewport
			display->SetViewport(display->GetWidth() / 2, display->GetHeight() / 2, 
							 display->GetWidth(), display->GetHeight());

			display->RenderRect(0.15f, 0.15f, 0.15f);

			// enable the camera and buffer
			camera->Activate();
			buffer->Bind();

			GL(glEnableClientState(GL_VERTEX_ARRAY));
			GL(glVertexPointer(3, GL_FLOAT, sizeof(PointVertex), 0));

			GL(glEnableClientState(GL_COLOR_ARRAY));
			GL(glColorPointer(4, GL_UNSIGNED_BYTE, sizeof(PointVertex), (void*)offsetof(PointVertex, color)));

			// draw the points
			GL(glDrawArrays(GL_POINTS, 0, GRID_POINTS));
	
			// disable the buffer and camera
			GL(glDisableClientState(GL_COLOR_ARRAY));
			GL(glDisableClientState(GL_VERTEX_ARRAY));

			buffer->Unbind();
			camera->Deactivate();
			display->ResetViewport();
		}

		// draw test texture
		if( texture != NULL && font != NULL )
		{
			void* textureCUDA = texture->Map(GL_MAP_CUDA, GL_WRITE_DISCARD);

			if( textureCUDA != NULL )
			{
				// clear the texture (from last frame)
				//CUDA(cudaMemset(textureCUDA, 0, texture->GetSize()));

				// test text
				char str[256];
				sprintf(str, "AaBbCcDdEeFfGgHhIiJjKkLlMmNn123456890");

				font->OverlayText((float4*)textureCUDA, texture->GetWidth(), texture->GetHeight(),
							   str, 0, 0, make_float4(0.0f, 190.0f, 255.0f, 255.0f));

				// FPS counter
				sprintf(str, "%.0f FPS", display->GetFPS());

				font->OverlayText((float4*)textureCUDA, texture->GetWidth(), texture->GetHeight(),
							   str, 0, 36, make_float4(255.0f, 190.0f, 0.0f, 255.0f));

				// rescale image pixel intensities for display
				CUDA(cudaNormalize((float4*)textureCUDA, make_float2(0.0f, 255.0f), 
							    (float4*)textureCUDA, make_float2(0.0f, 1.0f), 
		 					    texture->GetWidth(), texture->GetHeight()));

				texture->Unmap();
			}

			texture->Render(TEXTURE_OFFSET, TEXTURE_OFFSET);		
		}

		display->EndRender();
	}
	

	/*
	 * close the window
	 */
	if( display != NULL )
	{
		delete display;
		display = NULL;
	}
	
	printf("gl-display-test:  OpenGL display has been un-initialized.\n");
	printf("gl-display-test:  this concludes the test of the device.\n");

	return 0;
}
