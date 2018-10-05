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


/**
 * HW-accelerated H.264/H.265 video encoder for Jetson using GStreamer.
 * The encoder can write the encoded video to disk in .mkv or .h264/.h265 formats,
 * or handle streaming network transmission to remote host(s) via RTP/RTSP protocol.
 */
class gstEncoder
{
public:
	/**
	 * Create an encoder instance that outputs to a file on disk.
	 */
	static gstEncoder* Create( gstCodec codec, uint32_t width, uint32_t height, const char* filename );
	
	/**
	 * Create an encoder instance that streams over the network.
	 */
	static gstEncoder* Create( gstCodec codec, uint32_t width, uint32_t height, const char* ipAddress, uint16_t port );
	
	/**
	 * Create an encoder instance that outputs to a file on disk and streams over the network.
	 */
	static gstEncoder* Create( gstCodec codec, uint32_t width, uint32_t height, const char* filename, const char* ipAddress, uint16_t port );
	
	/**
	 * Destructor
	 */
	~gstEncoder();
	
	/**
	 * Encode the next fixed-point RGBA frame.
	 * Expects 8-bit per channel, 32-bit per pixel unsigned image, range 0-255.
	 * It is assumed the width of the buffer is equal to GetWidth(),
	 * and that the height of the buffer is equal to GetHeight().
	 * This function performs colorspace conversion using CUDA, so the
	 * buffer pointer is expected to be CUDA memory allocated on the GPU.
	 * @param buffer CUDA pointer to the RGBA image.
	 */
	bool EncodeRGBA( uint8_t* buffer );

	/**
	 * Encode the next floating-point RGBA frame.
	 * It is assumed the width of the buffer is equal to GetWidth(),
	 * and that the height of the buffer is equal to GetHeight().
	 * This function performs colorspace conversion using CUDA, so the
	 * buffer pointer is expected to be CUDA memory allocated on the GPU.
	 * @param buffer CUDA pointer to the RGBA image.
	 * @param maxPixelValue indicates the maximum pixel intensity (typically 255.0f or 1.0f)
	 */
	bool EncodeRGBA( float* buffer, float maxPixelValue=255.0f );

	/**
	 * Encode the next I420 frame provided by the user.
	 * Expects 12-bpp (bit per pixel) image in YUV I420 format.
	 * This image is passed to GStreamer, so CPU pointer should be used.
	 * @param buffer CPU pointer to the I420 image
	 */
	bool EncodeI420( void* buffer, size_t size );
	
	/**
	 * Retrieve the width that the encoder was created for, in pixels.
	 */
	inline uint32_t GetWidth() const			{ return mWidth; }

	/**
	 * Retrieve the height that the encoder was created for, in pixels.
	 */
	inline uint32_t GetHeight() const			{ return mHeight; }

protected:
	gstEncoder();
	
	bool buildCapsStr();
	bool buildLaunchStr();
	
	bool init( gstCodec codec, uint32_t width, uint32_t height, const char* filename, const char* ipAddress, uint16_t port );
	
	static void onNeedData( _GstElement* pipeline, uint32_t size, void* user_data );
	static void onEnoughData( _GstElement* pipeline, void* user_data );

	_GstBus*     mBus;
	_GstCaps*    mBufferCaps;
	_GstElement* mAppSrc;
	_GstElement* mPipeline;
	gstCodec     mCodec;
	bool         mNeedData;
	uint32_t     mWidth;
	uint32_t     mHeight;
	
	std::string  mCapsStr;
	std::string  mLaunchStr;
	std::string  mOutputPath;
	std::string  mOutputIP;
	uint16_t     mOutputPort;

	// format conversion buffers
	void* mCpuRGBA;
	void* mGpuRGBA;
	void* mCpuI420;
	void* mGpuI420;
};
 
 
#endif
