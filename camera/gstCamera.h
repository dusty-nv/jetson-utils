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

#ifndef __GSTREAMER_CAMERA_H__
#define __GSTREAMER_CAMERA_H__

#include <gst/gst.h>
#include <string>

#include "Mutex.h"
#include "Event.h"

struct _GstAppSink;


/**
 * Enumeration of camera input source methods
 * @ingroup util
 */
enum gstCameraSrc
{
	GST_SOURCE_NVCAMERA,	/* use nvcamerasrc element */
	GST_SOURCE_NVARGUS,		/* use nvargussrc element */
	GST_SOURCE_V4L2		/* use v4l2src element */
};

const char* gstCameraSrcToString( gstCameraSrc src );	/**< Text version of gstCameraSrc enum */


/**
 * gstreamer CSI camera using nvcamerasrc (or optionally v4l2src)
 * @ingroup util
 */
class gstCamera
{
public:
	// Create camera
	static gstCamera* Create( int v4l2_device=-1 );	// use onboard camera by default (>=0 for V4L2)
	static gstCamera* Create( uint32_t width, uint32_t height, int v4l2_device=-1 );
	
	// Destroy
	~gstCamera();

	// Start/stop streaming
	bool Open();
	void Close();
	
	// Is open for streaming
	inline bool IsStreaming() const	   { return mStreaming; }

	// Capture YUV (NV12)
	// If timeout is UINT64_MAX, the calling thread will wait indefinetly for a new frame
	// If timeout is 0, the calling thread will return false if a new frame isn't immediately ready
	// Otherwise the timeout is in millseconds before returning if a new frame isn't ready
	bool Capture( void** cpu, void** cuda, uint64_t timeout=UINT64_MAX );

	// Capture a camera image and convert to it float4 RGBA
	// If you want to capture in a different thread than CUDA, use the Capture() and ConvertRGBA() functions.
	// Set zeroCopy to true if you need to access the image from CPU, otherwise it will be CUDA only.
	bool CaptureRGBA( float** output, uint64_t timeout=UINT64_MAX, bool zeroCopy=false );
	
	// Takes in captured YUV-NV12 CUDA image, converts to float4 RGBA (with pixel intensity 0-255)
	// Set zeroCopy to true if you need to access ConvertRGBA from CPU, otherwise it will be CUDA only.
	bool ConvertRGBA( void* input, float** output, bool zeroCopy=false );

	// Takes in captured YUV-NV12 CUDA image, converts to uint8 BGR (with pixel intensity 0-255)
	// Set zeroCopy to true if you need to access ConvertBGR8 from CPU, otherwise it will be CUDA only.
	bool ConvertBGR8( void* input, void** output, bool zeroCopy=false );	
	
	// Image dimensions
	inline uint32_t GetWidth() const	   { return mWidth; }
	inline uint32_t GetHeight() const	   { return mHeight; }
	inline uint32_t GetPixelDepth() const { return mDepth; }
	inline uint32_t GetSize() const	   { return mSize; }
	
	// Default resolution, unless otherwise specified during Create()
	static const uint32_t DefaultWidth  = 1280;
	static const uint32_t DefaultHeight = 720;
	
private:
	static void onEOS(_GstAppSink* sink, void* user_data);
	static GstFlowReturn onPreroll(_GstAppSink* sink, void* user_data);
	static GstFlowReturn onBuffer(_GstAppSink* sink, void* user_data);

	gstCamera();

	bool init( gstCameraSrc src );
	bool buildLaunchStr( gstCameraSrc src );
	void checkMsgBus();
	void checkBuffer();
	
	_GstBus*     mBus;
	_GstAppSink* mAppSink;
	_GstElement* mPipeline;
	gstCameraSrc mSource;

	std::string  mLaunchStr;
	
	uint32_t mWidth;
	uint32_t mHeight;
	uint32_t mDepth;
	uint32_t mSize;

	static const uint32_t NUM_RINGBUFFERS = 16;
	
	void* mRingbufferCPU[NUM_RINGBUFFERS];
	void* mRingbufferGPU[NUM_RINGBUFFERS];
	
	Event mWaitEvent;
	Mutex mWaitMutex;
	Mutex mRingMutex;
	
	uint32_t mLatestRGBA;
	uint32_t mLatestBGR8;
	uint32_t mLatestRingbuffer;
	bool     mLatestRetrieved;
	
	void*  mRGBA[NUM_RINGBUFFERS];
	void*  mBGR8[NUM_RINGBUFFERS];
	bool   mRGBAZeroCopy; // were the RGBA buffers allocated with zeroCopy?
	bool   mBGR8ZeroCopy;
	bool   mStreaming;	  // true if the device is currently open
	int    mV4L2Device;	  // -1 for onboard, >=0 for V4L2 device

	inline bool onboardCamera() const		{ return (mV4L2Device < 0); }
};

#endif
