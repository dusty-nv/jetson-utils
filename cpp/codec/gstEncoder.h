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


// Forward declarations
class RTSPServer;
class WebRTCServer;
struct WebRTCPeer;


/**
 * Hardware-accelerated video encoder for Jetson using GStreamer.
 *
 * The encoder can write the encoded video to disk in (MKV, MP4, AVI, FLV),
 * or stream over the network to a remote host via RTP/RTSP using UDP/IP.
 * The supported encoder codecs are H.264, H.265, VP8, VP9, and MJPEG.
 *
 * @note gstEncoder implements the videoOutput interface and is intended to
 * be used through that as opposed to directly.  videoOutput implements
 * additional command-line parsing of videoOptions to construct instances.
 *
 * @see videoOutput
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
	 * Encode the next frame.
	 * @see videoOutput::Render()
	 */
	template<typename T> bool Render( T* image, uint32_t width, uint32_t height, cudaStream_t stream=0 )		{ return Render((void**)image, width, height, imageFormatFromType<T>(), stream); }
	
	/**
	 * Encode the next frame.
	 * @see videoOutput::Render()
	 */
	virtual bool Render( void* image, uint32_t width, uint32_t height, imageFormat format, cudaStream_t stream=0 );

	/**
	 * Open the stream.
	 * @see videoOutput::Open()
	 */
	virtual bool Open();

	/**
	 * Close the stream.
	 * @see videoOutput::Open()
	 */
	virtual void Close();

	/**
	 * Return the GStreamer pipeline object.
	 */
	inline GstPipeline* GetPipeline() const			{ return GST_PIPELINE(mPipeline); }
	
	/**
	 * Return the WebRTC server (only used when the protocol is "webrtc://")
	 */
	inline WebRTCServer* GetWebRTCServer() const 	{ return mWebRTCServer; }
	
	/**
	 * Return the interface type (gstEncoder::Type)
	 */
	virtual inline uint32_t GetType() const			{ return Type; }

	/**
	 * Unique type identifier of gstEncoder class.
	 */
	static const uint32_t Type = (1 << 2);

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
	gstEncoder( const videoOptions& options );
	
	bool init();
	bool initPipeline();
	void destroyPipeline();
	
	void checkMsgBus();
	bool buildCapsStr();
	bool buildLaunchStr();
	bool encodeYUV( void* buffer, size_t size );
	
	// appsrc callbacks
	static void onNeedData( GstElement* pipeline, uint32_t size, void* user_data );
	static void onEnoughData( GstElement* pipeline, void* user_data );

	// WebRTC callbacks
	static void onWebsocketMessage( WebRTCPeer* peer, const char* message, size_t message_size, void* user_data );

	GstBus*     mBus;
	GstCaps*    mBufferCaps;
	GstElement* mAppSrc;
	GstElement* mPipeline;
	bool        mNeedData;
	
	std::string  mCapsStr;
	std::string  mLaunchStr;

	RingBuffer mBufferYUV;
	
	RTSPServer*   mRTSPServer;
	WebRTCServer* mWebRTCServer;
};
 
 
#endif
