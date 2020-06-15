/*
 * Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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

#ifndef __GSTREAMER_DECODER_H__
#define __GSTREAMER_DECODER_H__

#include "gstUtility.h"
#include "videoSource.h"

#include "Event.h"
#include "RingBuffer.h"


struct _GstAppSink;


/**
 * Hardware-accelerated H.264/H.265 video decoder for Jetson using GStreamer.
 * @ingroup codec
 */
class gstDecoder : public videoSource
{
public:
	/**
	 * Create a decoder from the provided video options.
	 */
	static gstDecoder* Create( const videoOptions& options );

	/**
	 * Create a decoder instance from resource URI and codec.
	 */
	static gstDecoder* Create( const URI& resource, videoOptions::Codec codec );
	
	/**
	 * Destructor
	 */
	~gstDecoder();
	
	/**
	 * Capture
	 */
	template<typename T> bool Capture( T** image, uint64_t timeout=UINT64_MAX )		{ return Capture((void**)image, imageFormatFromType<T>(), timeout); }
	
	/**
	 * Capture
	 */
	virtual bool Capture( void** image, imageFormat format, uint64_t timeout=UINT64_MAX );

	/**
	 * Open
	 */
	virtual bool Open();

	/**
	 * Close
	 */
	virtual void Close();

	/**
	 * IsEOS
	 */
	inline bool IsEOS() const				{ return mEOS; }

	/**
	 *
	 */
	virtual inline uint32_t GetType() const		{ return Type; }

	/**
	 *
	 */
	static const uint32_t Type = (1 << 1);

	/**
	 *
	 */
	static const char* SupportedExtensions[];

	/**
	 *
	 */
	static bool IsSupportedExtension( const char* ext );

protected:
	gstDecoder( const videoOptions& options );
	
	void checkMsgBus();
	void checkBuffer();
	bool buildLaunchStr();
	
	bool init();
	bool discover();
	
	inline bool isLooping() const { return (mOptions.loop < 0) || ((mOptions.loop > 0) && (mLoopCount < mOptions.loop)); }

	static void onEOS(_GstAppSink* sink, void* user_data);
	static GstFlowReturn onPreroll(_GstAppSink* sink, void* user_data);
	static GstFlowReturn onBuffer(_GstAppSink* sink, void* user_data);

	_GstBus*     mBus;
	_GstAppSink* mAppSink;
	_GstElement* mPipeline;
	//gstCodec     mCodec;
	
	Event	   mWaitEvent;
	RingBuffer   mBufferYUV;
	RingBuffer   mBufferRGB;

	std::string  mLaunchStr;
	bool         mCustomSize;
	bool         mEOS;
	size_t	   mLoopCount;
	size_t	   mFrameCount;
	imageFormat  mFormatYUV;
};
  
#endif
