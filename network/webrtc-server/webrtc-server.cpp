/*
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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

#include "WebRTCServer.h"

#include "logging.h"
#include "commandLine.h"

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


int usage()
{
	printf("usage: webrtc-server [--help] --port PORT\n\n");
	printf("See below for additional arguments that may not be shown above.\n\n");
	printf("%s", Log::Usage());

	return 0;
}



int main( int argc, char** argv )
{
	/*
	 * parse command line
	 */
	commandLine cmdLine(argc, argv);

	if( cmdLine.GetFlag("help") )
		return usage();

	Log::ParseCmdLine(cmdLine);
	
	
	/*
	 * attach signal handler
	 */
	if( signal(SIGINT, sig_handler) == SIG_ERR )
		LogError("can't catch SIGINT\n");
	
	
	/*
	 * create server
	 */
	const uint32_t port = cmdLine.GetUnsignedInt("port", WEBRTC_DEFAULT_PORT);
	WebRTCServer* server = WebRTCServer::Create(port);
	
	if( !server )
		return 1;
	
	
	/*
	 * main loop
	 */
	while( !signal_recieved )
	{
		server->ProcessRequests();
	}
	
	
	/*
	 * destroy resources
	 */
	printf("webrtc-server:  shutting down...\n");
	
	server->Release();

	printf("webrtc-server:  shutdown complete\n");
}

