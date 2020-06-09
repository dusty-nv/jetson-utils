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

#ifndef __GSTREAMER_ENCODER_H__
#define __GSTREAMER_ENCODER_H__

#include "gstUtility.h"
#include "videoOutput.h"
#include "RingBuffer.h"


/**
 * Hardware-accelerated H.264/H.265 video encoder for Jetson using GStreamer.
 * The encoder can write the encoded video to disk in .mkv or .h264/.h265 formats,
 * or handle streaming network transmission to remote host(s) via RTP/RTSP protocol.
 * @ingroup codec
 */
class gstEncoder : public videoOutput
{
public:
	/**
	 * Create an encoder from the provided video options.
	 */
	static gstEncoder* Create( const videoOptions& options );

	/**
	 * Create an encoder instance from resource URI and codec.
	 */
	static gstEncoder* Create( const URI& resource, videoOptions::Codec codec );
	
	/**
	 * Destructor
	 */
	~gstEncoder();
	
	/**
	 *
	 */
	template<typename T> bool Render( T* image, uint32_t width, uint32_t height )		{ return Render((void**)image, width, height, imageFormatFromType<T>()); }
	
	/**
	 *
	 */
	virtual bool Render( void* image, uint32_t width, uint32_t height, imageFormat format );

	/**
	 * 
	 */
	virtual bool Open();

	/**
	 * 
	 */
	virtual void Close();

	/**
	 *
	 */
	virtual inline uint32_t GetType() const		{ return Type; }

	/**
	 *
	 */
	static const uint32_t Type = (1 << 2);

	/**
	 *
	 */
	static const char* SupportedExtensions[];

	/**
	 *
	 */
	static bool IsSupportedExtension( const char* ext );

protected:
	gstEncoder( const videoOptions& options );
	
	bool init();

	void checkMsgBus();
	bool buildCapsStr();
	bool buildLaunchStr();
	
	bool encodeYUV( void* buffer, size_t size );

	static void onNeedData( _GstElement* pipeline, uint32_t size, void* user_data );
	static void onEnoughData( _GstElement* pipeline, void* user_data );

	_GstBus*     mBus;
	_GstCaps*    mBufferCaps;
	_GstElement* mAppSrc;
	_GstElement* mPipeline;
	bool         mNeedData;
	
	std::string  mCapsStr;
	std::string  mLaunchStr;
	std::string  mOutputPath;
	std::string  mOutputIP;
	uint16_t     mOutputPort;

	RingBuffer mBufferYUV;
};
 
 
#endif
