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

#include "gstEncoder.h"

#include "filesystem.h"
#include "timespec.h"

#include "cudaColorspace.h"

#include <gst/gst.h>
#include <gst/app/gstappsrc.h>

#include <sstream>
#include <string.h>
#include <strings.h>
#include <unistd.h>


// constructor
gstEncoder::gstEncoder( const videoOptions& options ) : videoOutput(options)
{	
	mAppSrc     = NULL;
	mBus        = NULL;
	mBufferCaps = NULL;
	mPipeline   = NULL;
	mNeedData   = false;
	mOutputPort = 0;

	mCpuRGBA    = NULL;
	mGpuRGBA    = NULL;
	mCpuI420    = NULL;
	mGpuI420    = NULL;

	mBufferYUV.SetThreaded(false);
}


// destructor	
gstEncoder::~gstEncoder()
{
	// send EOS
	mNeedData = false;
	
	printf(LOG_GSTREAMER "gstEncoder -- shutting down pipeline, sending EOS\n");
	GstFlowReturn eos_result = gst_app_src_end_of_stream(GST_APP_SRC(mAppSrc));

	if( eos_result != 0 )
		printf(LOG_GSTREAMER "gstEncoder -- failed sending appsrc EOS (result %u)\n", eos_result);

	sleep(1);

	// stop pipeline
	printf(LOG_GSTREAMER "gstEncoder -- transitioning pipeline to GST_STATE_NULL\n");

	const GstStateChangeReturn result = gst_element_set_state(mPipeline, GST_STATE_NULL);

	if( result != GST_STATE_CHANGE_SUCCESS )
		printf(LOG_GSTREAMER "gstEncoder -- failed to set pipeline state to NULL (error %u)\n", result);

	sleep(1);
	
	printf(LOG_GSTREAMER "gstEncoder -- pipeline shutdown complete\n");	
}


// Create
gstEncoder* gstEncoder::Create( const videoOptions& options )
{
	gstEncoder* enc = new gstEncoder(options);
	
	if( !enc )
		return NULL;
	
	if( !enc->init() )
	{
		printf(LOG_GSTREAMER "gstEncoder -- failed to create encoder engine\n");
		return NULL;
	}
	
	return enc;
}


// Create
gstEncoder* gstEncoder::Create( const URI& resource, videoOptions::Codec codec )
{
	videoOptions opt;

	opt.resource = resource;
	opt.codec    = codec;
	opt.ioType   = videoOptions::OUTPUT;

	return Create(opt);
}
	

// init
bool gstEncoder::init()
{
	// initialize GStreamer libraries
	if( !gstreamerInit() )
	{
		printf(LOG_GSTREAMER "failed to initialize gstreamer API\n");
		return NULL;
	}

	// build caps string
	if( !buildCapsStr() )
	{
		printf(LOG_GSTREAMER "gstEncoder -- failed to build caps string\n");
		return false;
	}
	
	mBufferCaps = gst_caps_from_string(mCapsStr.c_str());

	if( !mBufferCaps )
	{
		printf(LOG_GSTREAMER "gstEncoder -- failed to parse caps from string\n");
		return false;
	}

	// build pipeline string
	if( !buildLaunchStr() )
	{
		printf(LOG_GSTREAMER "gstEncoder -- failed to build pipeline string\n");
		return false;
	}
	
	// create the pipeline
	GError* err = NULL;
	mPipeline   = gst_parse_launch(mLaunchStr.c_str(), &err);

	if( err != NULL )
	{
		printf(LOG_GSTREAMER "gstEncoder -- failed to create pipeline\n");
		printf(LOG_GSTREAMER "   (%s)\n", err->message);
		g_error_free(err);
		return false;
	}
	
	GstPipeline* pipeline = GST_PIPELINE(mPipeline);

	if( !pipeline )
	{
		printf(LOG_GSTREAMER "gstEncoder -- failed to cast GstElement into GstPipeline\n");
		return false;
	}	
	
	// retrieve pipeline bus
	mBus = gst_pipeline_get_bus(pipeline);

	if( !mBus )
	{
		printf(LOG_GSTREAMER "gstEncoder -- failed to retrieve GstBus from pipeline\n");
		return false;
	}
	
	// add watch for messages (disabled when we poll the bus ourselves, instead of gmainloop)
	//gst_bus_add_watch(mBus, (GstBusFunc)gst_message_print, NULL);

	// get the appsrc element
	GstElement* appsrcElement = gst_bin_get_by_name(GST_BIN(pipeline), "mysource");
	GstAppSrc* appsrc = GST_APP_SRC(appsrcElement);

	if( !appsrcElement || !appsrc )
	{
		printf(LOG_GSTREAMER "gstEncoder -- failed to retrieve AppSrc element from pipeline\n");
		return false;
	}
	
	mAppSrc = appsrcElement;

	g_signal_connect(appsrcElement, "need-data", G_CALLBACK(onNeedData), this);
	g_signal_connect(appsrcElement, "enough-data", G_CALLBACK(onEnoughData), this);

#if GST_CHECK_VERSION(1,0,0)
	//gst_app_src_set_caps(appsrc, mBufferCaps);
#endif
	
	// set stream properties
	gst_app_src_set_stream_type(appsrc, GST_APP_STREAM_TYPE_STREAM);
	
	g_object_set(G_OBJECT(mAppSrc), "is-live", TRUE, NULL); 
	g_object_set(G_OBJECT(mAppSrc), "do-timestamp", TRUE, NULL); 
	
	// transition pipline to STATE_PLAYING
	printf(LOG_GSTREAMER "gstEncoder -- transitioning pipeline to GST_STATE_PLAYING\n");
	
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
			printf(LOG_GSTREAMER "gstEncoder -- NULL message after transitioning pipeline to PLAYING...\n");
#endif
	}
	else if( result != GST_STATE_CHANGE_SUCCESS )
	{
		printf(LOG_GSTREAMER "gstEncoder -- failed to set pipeline state to PLAYING (error %u)\n", result);
		return false;
	}
	
	return true;
}
	

// buildCapsStr
bool gstEncoder::buildCapsStr()
{
	std::ostringstream ss;

#if GST_CHECK_VERSION(1,0,0)
	ss << "video/x-raw";
	ss << ",width=" << GetWidth();
	ss << ",height=" << GetHeight();
	ss << ",format=(string)I420";
	ss << ",framerate=30/1";
#else
	ss << "video/x-raw-yuv";
	ss << ",width=" << GetWidth();
	ss << ",height=" << GetHeight();
	ss << ",format=(fourcc)I420";
	ss << ",framerate=30/1";
#endif
	
	mCapsStr = ss.str();
	
	printf(LOG_GSTREAMER "gstEncoder -- buffer caps string:\n");
	printf("%s\n", mCapsStr.c_str());
	return true;
}
	
	
// buildLaunchStr
bool gstEncoder::buildLaunchStr()
{
	std::ostringstream ss;
	ss << "appsrc name=mysource ! ";
	
	// determine the requested protocol to use
	const URI& uri = GetResource();

#if GST_CHECK_VERSION(1,0,0)
	ss << mCapsStr << " ! ";

	if( mOptions.codec == videoOptions::CODEC_H264 )
		ss << "omxh264enc ! video/x-h264 !  ";	// TODO:  investigate quality-level setting
	else if( mOptions.codec == videoOptions::CODEC_H265 )
		ss << "omxh265enc ! video/x-h265 ! ";
	else if( mOptions.codec == videoOptions::CODEC_VP8 )
		ss << "omxvp8enc ! video/x-vp8 ! ";
	else if( mOptions.codec == videoOptions::CODEC_VP9 )
		ss << "omxvp9enc ! video/x-vp9 ! ";
#else
	if( mOptions.codec == videoOptions::CODEC_H264 )
		ss << "nv_omx_h264enc quality-level=2 ! video/x-h264 ! ";
	else if( mOptions.codec == videoOptions::CODEC_H265 )
		ss << "nv_omx_h265enc quality-level=2 ! video/x-h265 ! ";
	else if( mOptions.codec == videoOptions::CODEC_VP8 )
		ss << "nv_omx_vp8enc quality-level=2 ! video/x-vp8 ! ";
	else if( mOptions.codec == videoOptions::CODEC_VP9 )
		ss << "nv_omx_vp9enc quality-level=2 ! video/x-vp9 ! ";
#endif

	//if( fileLen > 0 && ipLen > 0 )
	//	ss << "nvtee name=t ! ";

	if( uri.protocol == "file" )
	{
		if( uri.extension == "mkv" )
		{
			//ss << "matroskamux ! queue ! ";
			ss << "matroskamux ! ";
		}
		else if( uri.extension == "avi" )
		{
			ss << "avimux ! ";
		}
		else if( uri.extension == "flv" )
		{
			ss << "flvmux ! ";
		}
		else if( uri.extension == "mp4" || uri.extension == "qt" )
		{
			if( mOptions.codec == videoOptions::CODEC_H264 )
				ss << "h264parse ! ";
			else if( mOptions.codec == videoOptions::CODEC_H265 )
				ss << "h265parse ! ";

			ss << "qtmux ! ";
		}
		else if( uri.extension != "h264" && uri.extension != "h265" )
		{
			printf(LOG_GSTREAMER "gstEncoder -- unsupported video file extension (%s)\n", uri.extension.c_str());
			printf(LOG_GSTREAMER "              supported video extensions are:\n");
			printf(LOG_GSTREAMER "                 * mkv\n");
			printf(LOG_GSTREAMER "                 * mp4, qt\n");
			printf(LOG_GSTREAMER "                 * flv\n");
			printf(LOG_GSTREAMER "                 * avi\n");
			printf(LOG_GSTREAMER "                 * h264, h265\n");

			return false;
		}

		ss << "filesink location=" << uri.path;

		//if( ipLen > 0 )
		//	ss << " t. ! ";	// begin the second tee
	}
	else if( uri.protocol == "rtp" )
	{
		ss << "rtph264pay config-interval=1 ! udpsink host=";
		ss << uri.path << " ";

		if( uri.port != 0 )
			ss << "port=" << uri.port;

		ss << " auto-multicast=true";
	}

	mLaunchStr = ss.str();

	printf(LOG_GSTREAMER "gstEncoder -- pipeline launch string:\n");
	printf("%s\n", mLaunchStr.c_str());
	return true;
}


// onNeedData
void gstEncoder::onNeedData( GstElement* pipeline, guint size, gpointer user_data )
{
	printf(LOG_GSTREAMER "gstEncoder -- AppSrc requesting data (%u bytes)\n", size);
	
	if( !user_data )
		return;

	gstEncoder* enc = (gstEncoder*)user_data;
	enc->mNeedData  = true;
}
 

// onEnoughData
void gstEncoder::onEnoughData( GstElement* pipeline, gpointer user_data )
{
	printf(LOG_GSTREAMER "gstEncoder -- AppSrc signalling enough data\n");

	if( !user_data )
		return;

	gstEncoder* enc = (gstEncoder*)user_data;
	enc->mNeedData  = false;
}


// encodeYUV
bool gstEncoder::encodeYUV( void* buffer, size_t size )
{
	if( !buffer || size == 0 )
		return false;
	
	if( !mNeedData )
	{
		printf(LOG_GSTREAMER "gstEncoder -- pipeline full, skipping frame (%zu bytes)\n", size);
		return true;
	}

	
#if GST_CHECK_VERSION(1,0,0)
	// allocate gstreamer buffer memory
	GstBuffer* gstBuffer = gst_buffer_new_allocate(NULL, size, NULL);
	
	// map the buffer for write access
	GstMapInfo map; 

	if( gst_buffer_map(gstBuffer, &map, GST_MAP_WRITE) ) 
	{ 
		if( map.size != size )
		{
			printf(LOG_GSTREAMER "gstEncoder -- gst_buffer_map() size mismatch, got %zu bytes, expected %zu bytes\n", map.size, size);
			gst_buffer_unref(gstBuffer);
			return false;
		}
		
		memcpy(map.data, buffer, size);
		gst_buffer_unmap(gstBuffer, &map); 
	} 
	else
	{
		printf(LOG_GSTREAMER "gstEncoder -- failed to map gstreamer buffer memory (%zu bytes)\n", size);
		gst_buffer_unref(gstBuffer);
		return false;
	}

#else
	// convert memory to GstBuffer
	GstBuffer* gstBuffer = gst_buffer_new();	

	GST_BUFFER_MALLOCDATA(gstBuffer) = (guint8*)g_malloc(size);
	GST_BUFFER_DATA(gstBuffer) = GST_BUFFER_MALLOCDATA(gstBuffer);
	GST_BUFFER_SIZE(gstBuffer) = size;
	
		//static size_t num_frame = 0;
	//GST_BUFFER_TIMESTAMP(gstBuffer) = (GstClockTime)((num_frame / 30.0) * 1e9);	// for 1.0, use GST_BUFFER_PTS or GST_BUFFER_DTS instead
	//num_frame++;

	if( mBufferCaps != NULL )
		gst_buffer_set_caps(gstBuffer, mBufferCaps);
	
	memcpy(GST_BUFFER_DATA(gstBuffer), buffer, size);
#endif

	// queue buffer to gstreamer
	GstFlowReturn ret;	
	g_signal_emit_by_name(mAppSrc, "push-buffer", gstBuffer, &ret);
	gst_buffer_unref(gstBuffer);

	if( ret != 0 )
		printf(LOG_GSTREAMER "gstEncoder -- AppSrc pushed buffer abnormally (result %u)\n", ret);

	// check for any messages
	while(true)
	{
		GstMessage* msg = gst_bus_pop(mBus);

		if( !msg )
			break;

		gst_message_print(mBus, msg, this);
		gst_message_unref(msg);
	}
	
	return true;
}


// Render
bool gstEncoder::Render( void* image, imageFormat format, uint32_t width, uint32_t height )
{
	if( !image )
		return false;

	if( mOptions.width != width || mOptions.height != height )
		printf(LOG_GSTREAMER "gstEncoder::Render() -- warning, input dimensions (%ux%u) are different than expected (%ux%u)\n", width, height, mOptions.width, mOptions.height);

	// error checking / return
	bool enc_success = false;

	#define render_end()	\
		const bool substreams_success = videoOutput::Render(image, format, width, height); \
		return enc_success & substreams_success;

	// allocate color conversion buffer
	const size_t i420Size = imageFormatSize(FORMAT_I420, width, height);

	if( !mBufferYUV.Alloc(2, i420Size, RingBuffer::ZeroCopy) )
	{
		printf(LOG_GSTREAMER "gstEncoder -- failed to allocate buffers of %zu bytes\n", i420Size);
		enc_success = false;
		render_end();
	}

	// perform colorspace conversion
	void* nextYUV = mBufferYUV.Next(RingBuffer::Write);

	if( CUDA_FAILED(cudaConvertColor(image, format, nextYUV, FORMAT_I420, width, height)) )
	{
		printf(LOG_GSTREAMER "gstEncoder::Render() -- unsupported image format (%s)\n", imageFormatToStr(format));
		printf(LOG_GSTREAMER "                        supported formats are:\n");
		printf(LOG_GSTREAMER "                            * rgba32\n");
		
		enc_success = false;
		render_end();
	}

	CUDA(cudaDeviceSynchronize());
	
	// encode YUV buffer
	enc_success = encodeYUV(nextYUV, i420Size);

	// render sub-streams
	render_end();	
}

#if 0
// EncodeRGBA
bool gstEncoder::EncodeRGBA( uint8_t* buffer )
{
	if( !buffer )
		return false;

	const size_t i420Size = (mWidth * mHeight * 12) / 8;

	if( !mCpuI420 || !mGpuI420 )
	{
		if( !cudaAllocMapped(&mCpuI420, &mGpuI420, i420Size) )
		{
			printf(LOG_GSTREAMER "gstEncoder -- failed to allocate CUDA memory for YUV I420 conversion\n");
			return false;
		}
	}

	if( CUDA_FAILED(cudaRGBAToI420((uchar4*)buffer, (uint8_t*)mGpuI420, mWidth, mHeight)) )
	{
		printf(LOG_GSTREAMER "gstEncoder -- failed convert RGBA image to I420\n");
		return false;
	}

	CUDA(cudaDeviceSynchronize());

	return EncodeI420(mCpuI420, i420Size);
}


// EncodeRGBA
bool gstEncoder::EncodeRGBA( float* buffer, float maxPixelValue )
{
	if( !buffer )
		return false;

	if( !mCpuRGBA || !mGpuRGBA )
	{
		if( !cudaAllocMapped(&mCpuRGBA, &mGpuRGBA, mWidth * mHeight * 4 * sizeof(uint8_t)) )
		{
			printf(LOG_GSTREAMER "gstEncoder -- failed to allocate CUDA memory for RGBA8 conversion\n");
			return false;
		}
	}

	if( CUDA_FAILED(cudaRGBA32ToRGBA8((float4*)buffer, (uchar4*)mGpuRGBA, mWidth, mHeight, make_float2(0.0f, maxPixelValue))) )
	{
		printf(LOG_GSTREAMER "gstEncoder -- failed convert RGBA32f image to RGBA8\n");
		return false;
	}

	return EncodeRGBA((uint8_t*)mGpuRGBA);
}
#endif


