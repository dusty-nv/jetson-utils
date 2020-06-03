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

#include "gstDecoder.h"

#include "filesystem.h"
#include "timespec.h"

#include <gst/gst.h>
#include <gst/app/gstappsink.h>

#include <sstream>
#include <unistd.h>
#include <string.h>
#include <strings.h>

#include "cudaYUV.h"


// 
// RTP test source pipeline:
//  $ gst-launch-1.0 -v videotestsrc ! video/x-raw,framerate=30/1 ! videoscale ! videoconvert ! x264enc tune=zerolatency bitrate=500 speed-preset=superfast ! rtph264pay ! udpsink host=127.0.0.1 port=5000
//  rtp://@:5000
//
// RSTP test server installation:
//  $ git clone https://github.com/GStreamer/gst-rtsp-server && cd gst-rtsp-server
//  $ git checkout 1.14.5
//  $ ./autogen.sh --noconfigure && ./configure && make
//  $ cd examples && ./test-launch "( videotestsrc ! x264enc ! rtph264pay name=pay0 pt=96 )"
//  rtsp://127.0.0.1:8554/test
//


// constructor
gstDecoder::gstDecoder( const videoOptions& options ) : videoSource(options)
{	
	mAppSink    = NULL;
	mBus        = NULL;
	mPipeline   = NULL;
	mEOS        = false;

	mBufferColor.SetThreaded(false);
}


// destructor
gstDecoder::~gstDecoder()
{
	Close();
}


// Create
gstDecoder* gstDecoder::Create( const videoOptions& options )
{
	gstDecoder* dec = new gstDecoder(options);

	if( !dec )
		return NULL;

	if( !dec->init() )
	{
		printf(LOG_GSTREAMER "gstDecoder -- failed to create decoder engine\n");
		return NULL;
	}
	
	return dec;
}


// Create
gstDecoder* gstDecoder::Create( const URI& resource, videoOptions::Codec codec )
{
	videoOptions opt;

	opt.resource = resource;
	opt.codec    = codec;
	opt.ioType   = videoOptions::INPUT;

	return Create(opt);
}
	

// init
bool gstDecoder::init()
{
	GError* err  = NULL;
	
	if( !gstreamerInit() )
	{
		printf(LOG_GSTREAMER "failed to initialize gstreamer API\n");
		return NULL;
	}

	// build pipeline string
	if( !buildLaunchStr() )
	{
		printf(LOG_GSTREAMER "gstDecoder -- failed to build pipeline string\n");
		return false;
	}

	// create pipeline
	mPipeline = gst_parse_launch(mLaunchStr.c_str(), &err);

	if( err != NULL )
	{
		printf(LOG_GSTREAMER "gstDecoder -- failed to create pipeline\n");
		printf(LOG_GSTREAMER "   (%s)\n", err->message);
		g_error_free(err);
		return false;
	}

	GstPipeline* pipeline = GST_PIPELINE(mPipeline);

	if( !pipeline )
	{
		printf(LOG_GSTREAMER "gstDecoder -- failed to cast GstElement into GstPipeline\n");
		return false;
	}	

	// retrieve pipeline bus
	/*GstBus**/ mBus = gst_pipeline_get_bus(pipeline);

	if( !mBus )
	{
		printf(LOG_GSTREAMER "gstDecoder -- failed to retrieve GstBus from pipeline\n");
		return false;
	}

	// add watch for messages (disabled when we poll the bus ourselves, instead of gmainloop)
	//gst_bus_add_watch(mBus, (GstBusFunc)gst_message_print, NULL);

	// get the appsrc
	GstElement* appsinkElement = gst_bin_get_by_name(GST_BIN(pipeline), "mysink");
	GstAppSink* appsink = GST_APP_SINK(appsinkElement);

	if( !appsinkElement || !appsink)
	{
		printf(LOG_GSTREAMER "gstDecoder -- failed to retrieve AppSink element from pipeline\n");
		return false;
	}
	
	mAppSink = appsink;

	// setup callbacks
	GstAppSinkCallbacks cb;
	memset(&cb, 0, sizeof(GstAppSinkCallbacks));
	
	cb.eos         = onEOS;
	cb.new_preroll = onPreroll;
#if GST_CHECK_VERSION(1,0,0)
	cb.new_sample  = onBuffer;
#else
	cb.new_buffer  = onBuffer;
#endif
	
	gst_app_sink_set_callbacks(mAppSink, &cb, (void*)this, NULL);
	
	return true;
}



// buildLaunchStr
bool gstDecoder::buildLaunchStr()
{
	std::ostringstream ss;

	// determine the requested protocol to use
	const URI& uri = GetResource();

	if( uri.protocol == "file" )
	{
		ss << "filesrc location=" << mOptions.resource.path << " ! ";

		if( uri.extension == "mkv" )
			ss << "matroskademux ! ";
		else if( uri.extension == "mp4" || uri.extension == "qt" )
			ss << "qtdemux ! ";
		else if( uri.extension == "flv" )
			ss << "flvdemux ! ";
		else if( uri.extension == "avi" )
			ss << "avidemux ! ";
		else if( uri.extension != "h264" && uri.extension != "h265" )
		{
			printf(LOG_GSTREAMER "gstDecoder -- unsupported video file extension (%s)\n", uri.extension.c_str());
			printf(LOG_GSTREAMER "              supported video extensions are:\n");
			printf(LOG_GSTREAMER "                 * mkv\n");
			printf(LOG_GSTREAMER "                 * mp4, qt\n");
			printf(LOG_GSTREAMER "                 * flv\n");
			printf(LOG_GSTREAMER "                 * avi\n");
			printf(LOG_GSTREAMER "                 * h264, h265\n");

			return false;
		}

		ss << "queue ! ";
		
		if( mOptions.codec == videoOptions::CODEC_H264 )
			ss << "h264parse ! ";
		else if( mOptions.codec == videoOptions::CODEC_H265 )
			ss << "h265parse ! ";

		mOptions.deviceType = videoOptions::DEVICE_FILE;
	}
	else if( uri.protocol == "rtp" )
	{
		if( uri.port <= 0 )
		{
			printf(LOG_GSTREAMER "gstDecoder -- invalid RTP port (%i)\n", uri.port);
			return false;
		}

		ss << "udpsrc port=" << uri.port;
		ss << " multicast-group=" << uri.path << " auto-multicast=true";

		ss << " caps=\"" << "application/x-rtp,media=(string)video,clock-rate=(int)90000,encoding-name=(string)";
		
		if( mOptions.codec == videoOptions::CODEC_H264 )
			ss << "H264\" ! rtph264depay ! h264parse ! ";
		else if( mOptions.codec == videoOptions::CODEC_H265 )
			ss << "H265\" ! rtph265depay ! h265parse ! ";
		else if( mOptions.codec == videoOptions::CODEC_VP8 )
			ss << "VP8\" ! rtpvp8depay ! ";
		else if( mOptions.codec == videoOptions::CODEC_VP9 )
			ss << "VP9\" ! rtpvp9depay ! ";

		mOptions.deviceType = videoOptions::DEVICE_IP;
	}
	else if( uri.protocol == "rtsp" )
	{
		ss << "rtspsrc location=" << uri.string;
		//ss << " latency=200 drop-on-latency=true";
		ss << " ! queue ! ";
		
		if( mOptions.codec == videoOptions::CODEC_H264 )
			ss << "rtph264depay ! h264parse ! ";
		else if( mOptions.codec == videoOptions::CODEC_H265 )
			ss << "rtph265depay ! h265parse ! ";
		else if( mOptions.codec == videoOptions::CODEC_VP8 )
			ss << "rtpvp8depay ! ";
		else if( mOptions.codec == videoOptions::CODEC_VP9 )
			ss << "rtpvp9depay ! ";

		mOptions.deviceType = videoOptions::DEVICE_IP;
	}
	else
	{
		printf(LOG_GSTREAMER "gstDecoder -- unsupported protocol (%s)\n", uri.protocol.c_str());
		printf(LOG_GSTREAMER "              supported protocols are:\n");
		printf(LOG_GSTREAMER "                 * file://\n");
		printf(LOG_GSTREAMER "                 * rtp://\n");
		printf(LOG_GSTREAMER "                 * rtsp://\n");

		return false;
	}

#if GST_CHECK_VERSION(1,0,0)
	if( mOptions.codec == videoOptions::CODEC_H264 )
		ss << "omxh264dec ! ";
	else if( mOptions.codec == videoOptions::CODEC_H265 )
		ss << "omxh265dec ! ";
	else if( mOptions.codec == videoOptions::CODEC_VP8 )
		ss << "omxvp8dec ! ";
	else if( mOptions.codec == videoOptions::CODEC_VP9 )
		ss << "omxvp9dec ! ";
#else
	if( mOptions.codec == videoOptions::CODEC_H264 )
		ss << "nv_omx_h264dec ! ";
	else if( mOptions.codec == videoOptions::CODEC_H265 )
		ss << "nv_omx_h265dec ! ";
	else if( mOptions.codec == videoOptions::CODEC_VP8 )
		ss << "nv_omx_vp8dec ! ";
	else if( mOptions.codec == videoOptions::CODEC_VP9 )
		ss << "nv_omx_vp9dec ! ";
#endif
	else
	{
		printf(LOG_GSTREAMER "gstDecoder -- unsupported codec requested (should be H.264/H.265/VP8/VP9)\n");
		return false;
	}

	// resize if requested
	if( mOptions.width != 0 && mOptions.height != 0 || mOptions.flipMethod != videoOptions::FLIP_NONE )
	{
		ss << "nvvidconv";

		if( mOptions.flipMethod != videoOptions::FLIP_NONE )
			ss << " flip-method=" << (int)mOptions.flipMethod;

		ss << " ! video/x-raw";

		if( mOptions.width != 0 && mOptions.height != 0 )
			ss << ", width=(int)" << mOptions.width << ", height=(int)" << mOptions.height << ", format=(string)NV12";

		ss <<" ! ";
	}
	else
		ss << "video/x-raw ! ";

	ss << "appsink name=mysink";

	mLaunchStr = ss.str();

	printf(LOG_GSTREAMER "gstDecoder -- pipeline string:\n");
	printf("%s\n", mLaunchStr.c_str());
	return true;
}


// onEOS
void gstDecoder::onEOS( _GstAppSink* sink, void* user_data )
{
	printf(LOG_GSTREAMER "gstDecoder -- onEOS()\n");

	if( !user_data )
		return;

	gstDecoder* dec = (gstDecoder*)user_data;

	dec->mEOS = true;	
	dec->mStreaming = false;
	//dec->Close();
}


// onPreroll
GstFlowReturn gstDecoder::onPreroll( _GstAppSink* sink, void* user_data )
{
	printf(LOG_GSTREAMER "gstDecoder -- onPreroll()\n");
	return GST_FLOW_OK;
}


// onBuffer
GstFlowReturn gstDecoder::onBuffer(_GstAppSink* sink, void* user_data)
{
	//printf(LOG_GSTREAMER "gstDecoder -- onBuffer()\n");
	
	if( !user_data )
		return GST_FLOW_OK;
		
	gstDecoder* dec = (gstDecoder*)user_data;
	
	dec->checkBuffer();
	dec->checkMsgBus();
	return GST_FLOW_OK;
}

#if GST_CHECK_VERSION(1,0,0)
#define release_return { gst_sample_unref(gstSample); return; }
#else
#define release_return { gst_buffer_unref(gstBuffer); return; }
#endif

// checkBuffer
void gstDecoder::checkBuffer()
{
	if( !mAppSink )
		return;

	
#if GST_CHECK_VERSION(1,0,0)
	// block waiting for the sample
	GstSample* gstSample = gst_app_sink_pull_sample(mAppSink);
	
	if( !gstSample )
	{
		printf(LOG_GSTREAMER "gstDecoder -- app_sink_pull_sample() returned NULL...\n");
		return;
	}
	
	// retrieve sample caps
	GstCaps* gstCaps = gst_sample_get_caps(gstSample);
	
	if( !gstCaps )
	{
		printf(LOG_GSTREAMER "gstDecoder -- gst_sample had NULL caps...\n");
		release_return;
	}
	
	// retrieve the buffer from the sample
	GstBuffer* gstBuffer = gst_sample_get_buffer(gstSample);
	
	if( !gstBuffer )
	{
		printf(LOG_GSTREAMER "gstDecoder -- app_sink_pull_sample() returned NULL...\n");
		release_return;
	}
	
	// map the buffer memory for read access
	GstMapInfo map; 
	
	if( !gst_buffer_map(gstBuffer, &map, GST_MAP_READ) ) 
	{ 
		printf(LOG_GSTREAMER "gstDecoder -- failed to map gstreamer buffer memory\n");
		release_return;
	}
	
	const void* gstData = map.data;
	const guint gstSize = map.size;
#else
	// block waiting for the buffer
	GstBuffer* gstBuffer = gst_app_sink_pull_buffer(mAppSink);
	
	if( !gstBuffer )
	{
		printf(LOG_GSTREAMER "gstDecoder -- app_sink_pull_buffer() returned NULL...\n");
		return;
	}
	
	// retrieve data pointer
	void* gstData = GST_BUFFER_DATA(gstBuffer);
	const guint gstSize = GST_BUFFER_SIZE(gstBuffer);
	
	if( !gstData )
	{
		printf(LOG_GSTREAMER "gstDecoder -- gst_buffer had NULL data pointer...\n");
		release_return;
	}
	
	// retrieve caps
	GstCaps* gstCaps = gst_buffer_get_caps(gstBuffer);
	
	if( !gstCaps )
	{
		printf(LOG_GSTREAMER "gstDecoder -- gst_buffer had NULL caps...\n");
		release_return;
	}
#endif
	// retrieve caps structure
	GstStructure* gstCapsStruct = gst_caps_get_structure(gstCaps, 0);
	
	if( !gstCapsStruct )
	{
		printf(LOG_GSTREAMER "gstDecoder -- gst_caps had NULL structure...\n");
		release_return;
	}
	
	// retrieve the width and height of the buffer
	int width  = 0;
	int height = 0;
	
	if( !gst_structure_get_int(gstCapsStruct, "width", &width) ||
		!gst_structure_get_int(gstCapsStruct, "height", &height) )
	{
		printf(LOG_GSTREAMER "gstDecoder -- gst_caps missing width/height...\n");
		release_return;
	}
	
	//printf(LOG_GSTREAMER "gstDecoder -- recieved %ix%i frame\n", width, height);
		
	if( width < 1 || height < 1 )
		release_return;
	
	mOptions.width = width;
	mOptions.height = height;

	// allocate ringbuffer
	if( !mBufferRaw.Alloc(mOptions.numBuffers, gstSize, RingBuffer::ZeroCopy) )
	{
		printf(LOG_GSTREAMER "gstDecoder -- failed to allocate %u buffers of %u bytes\n", mOptions.numBuffers, gstSize);
		release_return;
	}

	// copy to next ringbuffer
	void* nextBuffer = mBufferRaw.Peek(RingBuffer::Write);

	if( !nextBuffer )
	{
		printf(LOG_GSTREAMER "gstDecoder -- failed to retrieve next ringbuffer for writing\n");
		release_return;
	}

	memcpy(nextBuffer, gstData, gstSize);
	mBufferRaw.Next(RingBuffer::Write);
	mWaitEvent.Wake();

#if GST_CHECK_VERSION(1,0,0)
	gst_buffer_unmap(gstBuffer, &map);
#endif
	
	release_return;
}


// Capture
bool gstDecoder::Capture( void** output, imageFormat format, uint64_t timeout )
{
	// verify the output pointer exists
	if( !output )
		return false;

	// confirm the camera is streaming
	if( !mStreaming )
	{
		if( !Open() )
			return false;
	}

	// wait until a new frame is recieved
	if( !mWaitEvent.Wait(timeout) )
		return false;

	// get the latest ringbuffer
	void* latestRaw = mBufferRaw.Next(RingBuffer::ReadLatestOnce);

	if( !latestRaw )
		return false;

	// allocate ringbuffer for colorspace conversion
	const size_t colorBufferSize = imageFormatSize(format, GetWidth(), GetHeight());

	if( !mBufferColor.Alloc(mOptions.numBuffers, colorBufferSize, mOptions.zeroCopy ? RingBuffer::ZeroCopy : 0) )
	{
		printf(LOG_GSTREAMER "gstDecoder -- failed to allocate %u buffers of %zu bytes\n", mOptions.numBuffers, colorBufferSize);
		return false;
	}

	// perform colorspace conversion
	void* nextColor = mBufferColor.Next(RingBuffer::Write);

	if( format == FORMAT_RGBA32 )
	{
		if( CUDA_FAILED(cudaNV12ToRGBA((uint8_t*)latestRaw, (float4*)nextColor, GetWidth(), GetHeight())) )
			return false;
	}
	else if( format == FORMAT_RGBA8 )
	{
		if( CUDA_FAILED(cudaNV12ToRGBA((uint8_t*)latestRaw, (uchar4*)nextColor, GetWidth(), GetHeight())) )
			return false;
	}
	else
	{
		printf(LOG_GSTREAMER "gstCamera -- unsupported format requested for Capture()\n");
		return false;
	}

	*output = nextColor;
	return true;
}


// Open
bool gstDecoder::Open()
{
	if( mEOS )
	{
		printf(LOG_GSTREAMER "gstDecoder -- End of Stream (EOS) has been reached, stream has been closed\n");
		return false;
	}

	if( mStreaming )
		return true;

	// transition pipline to STATE_PLAYING
	printf(LOG_GSTREAMER "opening gstDecoder for streaming, transitioning pipeline to GST_STATE_PLAYING\n");
	
	const GstStateChangeReturn result = gst_element_set_state(mPipeline, GST_STATE_PLAYING);

	if( result == GST_STATE_CHANGE_ASYNC )
	{
#if 0
		GstMessage* asyncMsg = gst_bus_timed_pop_filtered(mBus, 5 * GST_SECOND, 
    	 					      (GstMessageType)(GST_MESSAGE_ASYNC_DONE|GST_MESSAGE_ERROR)); 

		if( asyncMsg != NULL )
		{
			gst_message_print(mBus, asyncMsg, this);
			gst_message_unref(asyncMsg);
		}
		else
			printf(LOG_GSTREAMER "gstDecoder -- NULL message after transitioning pipeline to PLAYING...\n");
#endif
	}
	else if( result != GST_STATE_CHANGE_SUCCESS )
	{
		printf(LOG_GSTREAMER "gstDecoder -- failed to set pipeline state to PLAYING (error %u)\n", result);
		return false;
	}

	checkMsgBus();
	usleep(100 * 1000);
	checkMsgBus();

	mStreaming = true;
	return true;
}


// Close
void gstDecoder::Close()
{
	if( !mStreaming )
		return;

	// stop pipeline
	printf(LOG_GSTREAMER "closing gstDecoder for streaming, transitioning pipeline to GST_STATE_NULL\n");

	const GstStateChangeReturn result = gst_element_set_state(mPipeline, GST_STATE_NULL);

	if( result != GST_STATE_CHANGE_SUCCESS )
		printf(LOG_GSTREAMER "gstDecoder -- failed to stop pipeline (error %u)\n", result);

	usleep(250*1000);
	mStreaming = false;
	printf(LOG_GSTREAMER "gstDecoder -- pipeline stopped\n");
}


// checkMsgBus
void gstDecoder::checkMsgBus()
{
	while(true)
	{
		GstMessage* msg = gst_bus_pop(mBus);

		if( !msg )
			break;

		gst_message_print(mBus, msg, this);
		gst_message_unref(msg);
	}
}



