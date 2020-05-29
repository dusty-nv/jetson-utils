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

#include "gstDecoder.h"
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


	LogError(LOG_CUDA "test error\n");
	LogWarning(LOG_CUDA "test warning\n");
	LogInfo(LOG_CUDA "test info\n");
	LogVerbose(LOG_CUDA "test verbose\n");
	LogDebug(LOG_CUDA "test debug\n");

	Log::SetLevel(Log::INFO);
	
	LogError(LOG_CUDA "test error\n");
	LogWarning(LOG_CUDA "test warning\n");
	LogInfo(LOG_CUDA "test info\n");
	LogVerbose(LOG_CUDA "test verbose\n");
	LogDebug(LOG_CUDA "test debug\n");

	#define IMAGE_TYPE_TEST(type)	LogInfo(#type " type => %i\n", (int)imageFormatFromType<type>())
	
	IMAGE_TYPE_TEST(uchar3);
	IMAGE_TYPE_TEST(uchar4);
	
	IMAGE_TYPE_TEST(float3);
	IMAGE_TYPE_TEST(float4);
	
	//IMAGE_TYPE_TEST(float);

	/*
	 * open video file
	 */
	videoOptions options;

	//options.resource = "/media/nvidia/WD_NVME/datasets/test_videos/jellyfish-15-mbps-hd-h264.mkv";
	options.resource = "rtp://5000";	
	options.codec = videoOptions::CODEC_H264;
	options.numBuffers = 16;
	//options.width = 1280;
	//options.height = 720;
	options.flipMethod = videoOptions::FLIP_NONE;

	options.resource.print();
	options.print();

	gstDecoder* inputStream = gstDecoder::Create(options);

	if( !inputStream )
	{
		LogError(LOG_GSTREAMER "failed to open gstDecoder\n");
		return 0;
	}

	/*
	 * create openGL window
	 */
	glDisplay* display = glDisplay::Create();
	
	if( !display )
		printf("camera-viewer:  failed to create openGL display\n");
	

	uint32_t numFrames = 0;

	while( !signal_recieved )
	{
		float4* nextFrame = NULL;

		if( !inputStream->Capture(&nextFrame, 1000) )
		{
			LogError(LOG_GSTREAMER "failed to capture next video frame\n");
			continue;
		}

		numFrames++;

		LogInfo("captured %u frames (%u x %u)\n", numFrames, inputStream->GetWidth(), inputStream->GetHeight());

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

		if( inputStream->IsEOS() )
			signal_recieved = true;
	}

	SAFE_DELETE(inputStream);
}

