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
#include <string.h>
#include <strings.h>



// constructor
gstDecoder::gstDecoder()
{	
	mAppSink    = NULL;
	mBus        = NULL;
	mPipeline   = NULL;
	mPort       = 0;
	mCodec      = GST_CODEC_H264;
}


// destructor
gstDecoder::~gstDecoder()
{
	// stop pipeline
	printf(LOG_GSTREAMER "gstDecoder - shutting down pipeline\n");
	printf(LOG_GSTREAMER "gstDecoder - transitioning pipeline to GST_STATE_NULL\n");

	const GstStateChangeReturn result = gst_element_set_state(mPipeline, GST_STATE_NULL);

	if( result != GST_STATE_CHANGE_SUCCESS )
		printf(LOG_GSTREAMER "gstDecoder - failed to stop pipeline (error %u)\n", result);

	sleepMs(250);
	
	printf(LOG_GSTREAMER "gstDecoder - pipeline shutdown complete\n");
}


// Create
gstDecoder* gstDecoder::Create( gstCodec codec, const char* filename )
{
	gstDecoder* dec = new gstDecoder();
	
	if( !dec )
		return NULL;
	
	if( !dec->init(codec, filename, NULL, 0) )
	{
		printf(LOG_GSTREAMER "gstDecoder::Create() failed\n");
		return NULL;
	}
	
	return dec;
}
	

// Create
gstDecoder* gstDecoder::Create( gstCodec codec, uint16_t port )
{
	return Create(codec, NULL, port);
}
	

// Create
gstDecoder* gstDecoder::Create( gstCodec codec, const char* multicastIP, uint16_t port )
{
	gstDecoder* dec = new gstDecoder();
	
	if( !dec )
		return NULL;
	
	if( !dec->init(codec, NULL, multicastIP, port) )
	{
		printf(LOG_GSTREAMER "gstDecoder::Create() failed\n");
		return NULL;
	}
	
	return dec;
}


// init
bool gstDecoder::init( gstCodec codec, const char* filename, const char* multicastIP, uint16_t port )
{
	mCodec 		 = codec;
	mInputPath 	 = filename;
	mMulticastIP = multicastIP;
	mPort 		 = port;
	GError* err  = NULL;
	
	if( !filename && !multicastIP )
		return false;

	// build pipeline string
	if( !buildLaunchStr() )
	{
		printf(LOG_GSTREAMER "gstDecoder - failed to build pipeline string\n");
		return false;
	}

	// create pipeline
	mPipeline = gst_parse_launch(mLaunchStr.c_str(), &err);

	if( err != NULL )
	{
		printf(LOG_GSTREAMER "gstDecoder - failed to create pipeline\n");
		printf(LOG_GSTREAMER "   (%s)\n", err->message);
		g_error_free(err);
		return false;
	}

	GstPipeline* pipeline = GST_PIPELINE(mPipeline);

	if( !pipeline )
	{
		printf(LOG_GSTREAMER "gstDecoder - failed to cast GstElement into GstPipeline\n");
		return false;
	}	

	// retrieve pipeline bus
	/*GstBus**/ mBus = gst_pipeline_get_bus(pipeline);

	if( !mBus )
	{
		printf(LOG_GSTREAMER "gstDecoder - failed to retrieve GstBus from pipeline\n");
		return false;
	}

	// add watch for messages (disabled when we poll the bus ourselves, instead of gmainloop)
	//gst_bus_add_watch(mBus, (GstBusFunc)gst_message_print, NULL);

	// get the appsrc
	GstElement* appsinkElement = gst_bin_get_by_name(GST_BIN(pipeline), "mysink");
	GstAppSink* appsink = GST_APP_SINK(appsinkElement);

	if( !appsinkElement || !appsink)
	{
		printf(LOG_GSTREAMER "gstDecoder - failed to retrieve AppSink element from pipeline\n");
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


	// transition pipline to STATE_PLAYING
	printf(LOG_GSTREAMER "gstDecoder - transitioning pipeline to GST_STATE_PLAYING\n");
	
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
			printf(LOG_GSTREAMER "gstDecoder - NULL message after transitioning pipeline to PLAYING...\n");
#endif
	}
	else if( result != GST_STATE_CHANGE_SUCCESS )
	{
		printf(LOG_GSTREAMER "gstDecoder - failed to set pipeline state to PLAYING (error %u)\n", result);
		return false;
	}

	checkMsgBus();
	sleepMs(100);
	checkMsgBus();
	
	return true;
}


// buildLaunchStr
bool gstDecoder::buildLaunchStr()
{
	const size_t fileLen = mInputPath.size();
	
	if( fileLen > 0 && mPort != 0 )
	{
		printf(LOG_GSTREAMER "gstDecoder - can only use port %u or %s as input\n", mPort, mInputPath.c_str());
		return false;
	}
	
	std::ostringstream ss;
	
	if( fileLen > 0 )
	{
		ss << "filesrc location=" << mInputPath << " ! matroskademux ! queue ! ";
		
		if( mCodec == GST_CODEC_H264 )
			ss << "h264parse ! ";
		else if( mCodec == GST_CODEC_H265 )
			ss << "h265parse ! ";
	}
	else if( mPort != 0 )
	{
		ss << "udpsrc port=" << mPort;

		if( mMulticastIP.length() > 0 )
			ss << " multicast-group=" << mMulticastIP << " auto-multicast=true";

		ss << " caps=\"" << "application/x-rtp,media=(string)video,clock-rate=(int)90000,encoding-name=(string)";
		
		if( mCodec == GST_CODEC_H264 )
			ss << "H264\" ! rtph264depay ! ";
		else if( mCodec == GST_CODEC_H265 )
			ss << "H265\" ! rtph265depay ! ";
	}
	else
		return false;
	
#if GST_CHECK_VERSION(1,0,0)
	if( mCodec == GST_CODEC_H264 )
		ss << "omxh264dec ! ";
	else if( mCodec == GST_CODEC_H265 )
		ss << "omxh265dec ! ";
#else
	if( mCodec == GST_CODEC_H264 )
		ss << "nv_omx_h264dec ! ";
	else if( mCodec == GST_CODEC_H265 )
		ss << "nv_omx_h265dec ! ";
#endif

#define CAPS_STR "video/x-raw,format=(string)RGBA"
//#define CAPS_STR "video/x-raw-yuv,format=(fourcc)NV12"

	ss << "nvvidconv ! \"" << CAPS_STR << "\" ! ";
	ss << "appsink name=mysink caps=\"" << CAPS_STR << "\"";
	
	mLaunchStr = ss.str();

	printf(LOG_GSTREAMER "gstDecoder - pipeline string:\n");
	printf("%s\n", mLaunchStr.c_str());
	return true;
}


// onEOS
void gstDecoder::onEOS( _GstAppSink* sink, void* user_data )
{
	printf(LOG_GSTREAMER "gstDecoder - onEOS()\n");
}


// onPreroll
GstFlowReturn gstDecoder::onPreroll( _GstAppSink* sink, void* user_data )
{
	printf(LOG_GSTREAMER "gstDecoder - onPreroll()\n");
	return GST_FLOW_OK;
}


// onBuffer
GstFlowReturn gstDecoder::onBuffer(_GstAppSink* sink, void* user_data)
{
	printf(LOG_GSTREAMER "gstDecoder - onBuffer()\n");
	
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
		printf(LOG_GSTREAMER "gstDecoder - app_sink_pull_sample() returned NULL...\n");
		return;
	}
	
	// retrieve sample caps
	GstCaps* gstCaps = gst_sample_get_caps(gstSample);
	
	if( !gstCaps )
	{
		printf(LOG_GSTREAMER "gstDecoder - gst_sample had NULL caps...\n");
		release_return;
	}
	
	// retrieve the buffer from the sample
	GstBuffer* gstBuffer = gst_sample_get_buffer(gstSample);
	
	if( !gstBuffer )
	{
		printf(LOG_GSTREAMER "gstDecoder - app_sink_pull_sample() returned NULL...\n");
		release_return;
	}
	
	// map the buffer memory for read access
	GstMapInfo map; 
	
	if( !gst_buffer_map(gstBuffer, &map, GST_MAP_READ) ) 
	{ 
		printf(LOG_GSTREAMER "gstDecoder - failed to map gstreamer buffer memory\n");
		release_return;
	}
	
	const void* gstData = map.data;
	const guint gstSize = map.size;
#else
	// block waiting for the buffer
	GstBuffer* gstBuffer = gst_app_sink_pull_buffer(mAppSink);
	
	if( !gstBuffer )
	{
		printf(LOG_GSTREAMER "gstDecoder - app_sink_pull_buffer() returned NULL...\n");
		return;
	}
	
	// retrieve data pointer
	void* gstData = GST_BUFFER_DATA(gstBuffer);
	const guint gstSize = GST_BUFFER_SIZE(gstBuffer);
	
	if( !gstData )
	{
		printf(LOG_GSTREAMER "gstDecoder - gst_buffer had NULL data pointer...\n");
		release_return;
	}
	
	// retrieve caps
	GstCaps* gstCaps = gst_buffer_get_caps(gstBuffer);
	
	if( !gstCaps )
	{
		printf(LOG_GSTREAMER "gstDecoder - gst_buffer had NULL caps...\n");
		release_return;
	}
#endif
	// retrieve caps structure
	GstStructure* gstCapsStruct = gst_caps_get_structure(gstCaps, 0);
	
	if( !gstCapsStruct )
	{
		printf(LOG_GSTREAMER "gstDecoder - gst_caps had NULL structure...\n");
		release_return;
	}
	
	// retrieve the width and height of the buffer
	int width  = 0;
	int height = 0;
	
	if( !gst_structure_get_int(gstCapsStruct, "width", &width) ||
		!gst_structure_get_int(gstCapsStruct, "height", &height) )
	{
		printf(LOG_GSTREAMER "gstDecoder - gst_caps missing width/height...\n");
		release_return;
	}
	
	printf(LOG_GSTREAMER "gstDecoder - recieved %ix%i frame\n", width, height);
		
	if( width < 1 || height < 1 )
		release_return;
	
	/*// alloc ringbuffer
	const DataType type(12, 1, false, false);	// NV12
	
	if( !AllocRingbuffer2D(width, height, type) )
	{
		printf(LOG_GASKET "gstreamer decoder -- failed to alloc %ix%i ringbuffer\n", width, height);
		release_return;
	}*/
	
#if GST_CHECK_VERSION(1,0,0)
	gst_buffer_unmap(gstBuffer, &map);
#endif
	
	release_return;
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

