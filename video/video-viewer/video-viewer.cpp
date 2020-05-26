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
}
