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
 
#ifndef __RTSP_SERVER_H__
#define __RTSP_SERVER_H__

#include <stdint.h>


// forward declarations
class Thread;

struct _GMainLoop;
struct _GstRTSPServer;
struct _GstElement;


/**
 * Default port used by RTSP server.
 * @ingroup network
 */
#define RTSP_DEFAULT_PORT 8554

/**
 * RTSP logging prefix
 * @ingroup network
 */
#define LOG_RTSP "[rtsp]   "


/**
 * @ingroup network
 */
class RTSPServer
{
public:
	static RTSPServer* Create( uint16_t port=RTSP_DEFAULT_PORT );
	
	void Release();
	
	//bool AddRoute( const char* path, _GstElement* pipeline );
	bool AddRoute( const char* path, const char* pipeline );
	
protected:
	RTSPServer( uint16_t port );
	~RTSPServer();
	
	bool init();
	
	static void* runThread( void* user_data );
	
	uint16_t mPort;
	uint32_t mRefCount;
	
	Thread* mThread;
	bool    mRunning;
	
	_GMainLoop* mMainLoop;
	_GstRTSPServer* mServer;
};

#endif
