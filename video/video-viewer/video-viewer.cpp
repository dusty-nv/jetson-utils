/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#include "imageFormat.h"
#include "commandLine.h"
#include "logging.h"

#include "videoSource.h"
#include "glDisplay.h"

#include <signal.h>


bool signal_recieved = false;

void sig_handler(int signo)
{
	if( signo == SIGINT )
	{
		LogInfo("received SIGINT\n");
		signal_recieved = true;
	}
}


int main( int argc, char** argv )
{
	commandLine cmdLine(argc, argv);
	
	/*
	 * attach signal handler
	 */	
	if( signal(SIGINT, sig_handler) == SIG_ERR )
		LogError("can't catch SIGINT\n");


	/*
	 * open video stream
	 */
	videoSource* inputStream = videoSource::Create(cmdLine);

	if( !inputStream )
	{
		LogError("video-viewer:  failed to create input stream\n");
		return 0;
	}

	inputStream->GetOptions().Print();


	/*
	 * create openGL window
	 */
	glDisplay* display = glDisplay::Create();
	
	if( !display )
		printf("video-viewer:  failed to create openGL display\n");
	

	/*
	 * capture/display loop
	 */
	uint32_t numFrames = 0;

	while( !signal_recieved )
	{
		float4* nextFrame = NULL;

		if( !inputStream->Capture(&nextFrame, 1000) )
		{
			LogError("video-viewer:  failed to capture video frame\n");
			continue;
		}

		LogInfo("video-viewer:  captured %u frames (%u x %u)\n", ++numFrames, inputStream->GetWidth(), inputStream->GetHeight());

		if( display != NULL )
		{
			display->RenderOnce((float*)nextFrame, inputStream->GetWidth(), inputStream->GetHeight());

			// update status bar
			char str[256];
			sprintf(str, "Video Viewer (%ux%u) | %.0f FPS", inputStream->GetWidth(), inputStream->GetHeight(), display->GetFPS());
			display->SetTitle(str);	

			// check if the user quit
			if( display->IsClosed() )
				signal_recieved = true;
		}

		if( !inputStream->IsStreaming() )
			signal_recieved = true;
	}


	/*
	 * destroy resources
	 */
	printf("video-viewer:  shutting down...\n");
	
	SAFE_DELETE(inputStream);
	SAFE_DELETE(display);

	printf("video-viewer:  shutdown complete\n");
}

