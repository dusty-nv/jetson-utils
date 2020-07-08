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
 * Hardware-accelerated video decoder for Jetson using GStreamer.
 *
 * gstDecoder supports loading video files from disk (MKV, MP4, AVI, FLV)
 * and RTP/RTSP network streams over UDP/IP. The supported decoder codecs
 * are H.264, H.265, VP8, VP9, MPEG-2, MPEG-4, and MJPEG.
 *
 * @note gstDecoder implements the videoSource interface and is intended to
 * be used through that as opposed to directly.  videoSource implements
 * additional command-line parsing of videoOptions to construct instances.
 *
 * @see videoSource
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
	 * Capture the next decoded frame.
	 * @see videoSource::Capture()
	 */
	template<typename T> bool Capture( T** image, uint64_t timeout=UINT64_MAX )		{ return Capture((void**)image, imageFormatFromType<T>(), timeout); }
	
	/**
	 * Capture the next decoded frame.
	 * @see videoSource::Capture()
	 */
	virtual bool Capture( void** image, imageFormat format, uint64_t timeout=UINT64_MAX );

	/**
	 * Open the stream.
	 * @see videoSource::Open()
	 */
	virtual bool Open();

	/**
	 * Close the stream.
	 * @see videoSource::Close()
	 */
	virtual void Close();

	/**
	 * Return true if End Of Stream (EOS) has been reached.
	 * In the context of gstDecoder, EOS means that playback 
	 * has reached the end of the file, and looping is either
	 * disabled or all loops have already been run.  In the case
	 * of RTP/RTSP, it means that the stream has terminated.
	 */
	inline bool IsEOS() const				{ return mEOS; }

	/**
	 * Return the interface type (gstDecoder::Type)
	 */
	virtual inline uint32_t GetType() const		{ return Type; }

	/**
	 * Unique type identifier of gstDecoder class.
	 */
	static const uint32_t Type = (1 << 1);

	/**
	 * String array of supported video file extensions, terminated
	 * with a NULL sentinel value.  The supported extension are:
	 *
	 *    - MKV
	 *    - MP4 / QT
	 *    - AVI
	 *    - FLV
	 *
	 * @see IsSupportedExtension() to check a string against this list.
	 */
	static const char* SupportedExtensions[];

	/**
	 * Return true if the extension is in the list of SupportedExtensions.
	 * @param ext string containing the extension to be checked (should not contain leading dot)
	 * @see SupportedExtensions for the list of supported video file extensions.
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
	bool		   mCustomRate;
	bool         mEOS;
	size_t	   mLoopCount;
	size_t	   mFrameCount;
	imageFormat  mFormatYUV;
};
  
#endif
