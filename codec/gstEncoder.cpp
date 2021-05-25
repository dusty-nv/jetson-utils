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
#include "logging.h"

#include "cudaColorspace.h"

#include <gst/gst.h>
#include <gst/app/gstappsrc.h>

#include <sstream>
#include <string.h>
#include <strings.h>
#include <unistd.h>


// supported video file extensions
const char* gstEncoder::SupportedExtensions[] = { "mkv", "mp4", "qt", 
										"flv", "avi", "h264", 
										"h265", NULL };

bool gstEncoder::IsSupportedExtension( const char* ext )
{
	if( !ext )
		return false;

	uint32_t extCount = 0;

	while(true)
	{
		if( !SupportedExtensions[extCount] )
			break;

		if( strcasecmp(SupportedExtensions[extCount], ext) == 0 )
			return true;

		extCount++;
	}

	return false;
}


// constructor
gstEncoder::gstEncoder( const videoOptions& options ) : videoOutput(options)
{	
	mAppSrc     = NULL;
	mBus        = NULL;
	mBufferCaps = NULL;
	mPipeline   = NULL;
	mNeedData   = false;
	mOutputPort = 0;

	mBufferYUV.SetThreaded(false);
}


// destructor	
gstEncoder::~gstEncoder()
{
	Close();

	if( mAppSrc != NULL )
	{
		gst_object_unref(mAppSrc);
		mAppSrc = NULL;
	}

	if( mBus != NULL )
	{
		gst_object_unref(mBus);
		mBus = NULL;
	}

	if( mPipeline != NULL )
	{
		gst_object_unref(mPipeline);
		mPipeline = NULL;
	}
}


// Create
gstEncoder* gstEncoder::Create( const videoOptions& options )
{
	gstEncoder* enc = new gstEncoder(options);
	
	if( !enc )
		return NULL;
	
	if( !enc->init() )
	{
		LogError(LOG_GSTREAMER "gstEncoder -- failed to create encoder engine\n");
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
		LogError(LOG_GSTREAMER "failed to initialize gstreamer API\n");
		return NULL;
	}

	// check for default codec
	if( mOptions.codec == videoOptions::CODEC_UNKNOWN )
	{
		LogWarning(LOG_GSTREAMER "gstEncoder -- codec not specified, defaulting to H.264\n");
		mOptions.codec = videoOptions::CODEC_H264;
	}

	// check if default framerate is needed
	if( mOptions.frameRate <= 0 )
		mOptions.frameRate = 30;

	// build pipeline string
	if( !buildLaunchStr() )
	{
		LogError(LOG_GSTREAMER "gstEncoder -- failed to build pipeline string\n");
		return false;
	}
	
	// create the pipeline
	GError* err = NULL;
	mPipeline   = gst_parse_launch(mLaunchStr.c_str(), &err);

	if( err != NULL )
	{
		LogError(LOG_GSTREAMER "gstEncoder -- failed to create pipeline\n");
		LogError(LOG_GSTREAMER "   (%s)\n", err->message);
		g_error_free(err);
		return false;
	}
	
	GstPipeline* pipeline = GST_PIPELINE(mPipeline);

	if( !pipeline )
	{
		LogError(LOG_GSTREAMER "gstEncoder -- failed to cast GstElement into GstPipeline\n");
		return false;
	}	
	
	// retrieve pipeline bus
	mBus = gst_pipeline_get_bus(pipeline);

	if( !mBus )
	{
		LogError(LOG_GSTREAMER "gstEncoder -- failed to retrieve GstBus from pipeline\n");
		return false;
	}
	
	// add watch for messages (disabled when we poll the bus ourselves, instead of gmainloop)
	//gst_bus_add_watch(mBus, (GstBusFunc)gst_message_print, NULL);

	// get the appsrc element
	GstElement* appsrcElement = gst_bin_get_by_name(GST_BIN(pipeline), "mysource");
	GstAppSrc* appsrc = GST_APP_SRC(appsrcElement);

	if( !appsrcElement || !appsrc )
	{
		LogError(LOG_GSTREAMER "gstEncoder -- failed to retrieve appsrc element from pipeline\n");
		return false;
	}
	
	mAppSrc = appsrcElement;

	g_signal_connect(appsrcElement, "need-data", G_CALLBACK(onNeedData), this);
	g_signal_connect(appsrcElement, "enough-data", G_CALLBACK(onEnoughData), this);

#if GST_CHECK_VERSION(1,0,0)
	//gst_app_src_set_caps(appsrc, mBufferCaps);
#endif
	
#if 0
	// set stream properties (note: these are now set in the pipeline)
	gst_app_src_set_stream_type(appsrc, GST_APP_STREAM_TYPE_STREAM);
	
	g_object_set(G_OBJECT(mAppSrc), "is-live", TRUE, NULL); 
	g_object_set(G_OBJECT(mAppSrc), "do-timestamp", TRUE, NULL); 
#endif

	return true;
}
	

// buildCapsStr
bool gstEncoder::buildCapsStr()
{
	std::ostringstream ss;

#if GST_CHECK_VERSION(1,0,0)
	ss << "video/x-raw";
	ss << ", width=" << GetWidth();
	ss << ", height=" << GetHeight();
	ss << ", format=(string)I420";
	ss << ", framerate=" << (int)mOptions.frameRate << "/1";
#else
	ss << "video/x-raw-yuv";
	ss << ",width=" << GetWidth();
	ss << ",height=" << GetHeight();
	ss << ",format=(fourcc)I420";
	ss << ",framerate=" << (int)mOptions.frameRate << "/1";
#endif
	
	mCapsStr = ss.str();
	LogInfo(LOG_GSTREAMER "gstEncoder -- new caps: %s\n", mCapsStr.c_str());
	return true;
}
	
	
// buildLaunchStr
bool gstEncoder::buildLaunchStr()
{
	std::ostringstream ss;

	// setup appsrc input element
	ss << "appsrc name=mysource is-live=true do-timestamp=true format=3 ! ";

	// set default bitrate (if needed)
	if( mOptions.bitRate == 0 )
		mOptions.bitRate = 4000000; 
	
	// determine the requested protocol to use
	const URI& uri = GetResource();

#if GST_CHECK_VERSION(1,0,0)
	//ss << mCapsStr << " ! ";

	if( mOptions.codec == videoOptions::CODEC_H264 )
		ss << "omxh264enc bitrate=" << mOptions.bitRate << " ! video/x-h264 !  ";	// TODO:  investigate quality-level setting
	else if( mOptions.codec == videoOptions::CODEC_H265 )
		ss << "omxh265enc bitrate=" << mOptions.bitRate << " ! video/x-h265 ! ";
	else if( mOptions.codec == videoOptions::CODEC_VP8 )
		ss << "omxvp8enc bitrate=" << mOptions.bitRate << " ! video/x-vp8 ! ";
	else if( mOptions.codec == videoOptions::CODEC_VP9 )
		ss << "omxvp9enc bitrate=" << mOptions.bitRate << " ! video/x-vp9 ! ";
	else if( mOptions.codec == videoOptions::CODEC_MJPEG )
		ss << "nvjpegenc ! image/jpeg ! ";
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
	else
	{
		LogError(LOG_GSTREAMER "gstEncoder -- unsupported codec requested (%s)\n", videoOptions::CodecToStr(mOptions.codec));
		LogError(LOG_GSTREAMER "              supported encoder codecs are:\n");
		LogError(LOG_GSTREAMER "                 * h264\n");
		LogError(LOG_GSTREAMER "                 * h265\n");
		LogError(LOG_GSTREAMER "                 * vp8\n");
		LogError(LOG_GSTREAMER "                 * vp9\n");
		LogError(LOG_GSTREAMER "                 * mjpeg\n");
	}

	//if( fileLen > 0 && ipLen > 0 )
	//	ss << "nvtee name=t ! ";

	if( uri.protocol == "file" )
	{
		if( uri.extension == "mkv" )
		{
			ss << "matroskamux ! ";
		}
		else if( uri.extension == "flv" )
		{
			ss << "flvmux ! ";
		}
		else if( uri.extension == "avi" )
		{
			if( mOptions.codec == videoOptions::CODEC_H265 || mOptions.codec == videoOptions::CODEC_VP9 )
			{
				LogError(LOG_GSTREAMER "gstEncoder -- AVI format doesn't support codec %s\n", videoOptions::CodecToStr(mOptions.codec));
				LogError(LOG_GSTREAMER "              supported AVI codecs are:\n");
				LogError(LOG_GSTREAMER "                 * h264\n");
				LogError(LOG_GSTREAMER "                 * vp8\n");
				LogError(LOG_GSTREAMER "                 * mjpeg\n");

				return false;
			}

			ss << "avimux ! ";
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

		ss << "filesink location=" << uri.location;

		mOptions.deviceType = videoOptions::DEVICE_FILE;
	}
	else if( uri.protocol == "rtp" )
	{
		if( mOptions.codec == videoOptions::CODEC_H264 )
			ss << "rtph264pay";
		else if( mOptions.codec == videoOptions::CODEC_H265 )
			ss << "rtph265pay";
		else if( mOptions.codec == videoOptions::CODEC_VP8 )
			ss << "rtpvp8pay";
		else if( mOptions.codec == videoOptions::CODEC_VP9 )
			ss << "rtpvp9pay";
		else if( mOptions.codec == videoOptions::CODEC_MJPEG )
			ss << "rtpjpegpay";

		if (mOptions.codec == videoOptions::CODEC_H264 || mOptions.codec == videoOptions::CODEC_H265) {
				ss << " config-interval=1 ! udpsink host=";
		} else {
				ss << " ! udpsink host=";
		}
		ss << uri.location << " ";

		if( uri.port != 0 )
			ss << "port=" << uri.port;

		ss << " auto-multicast=true";

		mOptions.deviceType = videoOptions::DEVICE_IP;
	}
	else if( uri.protocol == "rtmp" )
	{
		ss << "flvmux streamable=true ! queue ! rtmpsink location=";
		ss << uri.string << " ";

		mOptions.deviceType = videoOptions::DEVICE_IP;
	}

	mLaunchStr = ss.str();

	LogInfo(LOG_GSTREAMER "gstEncoder -- pipeline launch string:\n");
	LogInfo(LOG_GSTREAMER "%s\n", mLaunchStr.c_str());

	return true;
}


// onNeedData
void gstEncoder::onNeedData( GstElement* pipeline, guint size, gpointer user_data )
{
	LogDebug(LOG_GSTREAMER "gstEncoder -- appsrc requesting data (%u bytes)\n", size);
	
	if( !user_data )
		return;

	gstEncoder* enc = (gstEncoder*)user_data;
	enc->mNeedData  = true;
}
 

// onEnoughData
void gstEncoder::onEnoughData( GstElement* pipeline, gpointer user_data )
{
	LogDebug(LOG_GSTREAMER "gstEncoder -- appsrc signalling enough data\n");

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
	
	// confirm the stream is open
	if( !mStreaming )
	{
		if( !Open() )
			return false;
	}

	// check to see if data can be accepted
	if( !mNeedData )
	{
		LogVerbose(LOG_GSTREAMER "gstEncoder -- pipeline full, skipping frame (%zu bytes)\n", size);
		return true;
	}

	// construct the buffer caps for this size image
	if( !mBufferCaps )
	{
		if( !buildCapsStr() )
		{
			LogError(LOG_GSTREAMER "gstEncoder -- failed to build caps string\n");
			return false;
		}

		mBufferCaps = gst_caps_from_string(mCapsStr.c_str());

		if( !mBufferCaps )
		{
			LogError(LOG_GSTREAMER "gstEncoder -- failed to parse caps from string:\n");
			LogError(LOG_GSTREAMER "   %s\n", mCapsStr.c_str());
			return false;
		}

	#if GST_CHECK_VERSION(1,0,0)
		gst_app_src_set_caps(GST_APP_SRC(mAppSrc), mBufferCaps);
	#endif
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
			LogError(LOG_GSTREAMER "gstEncoder -- gst_buffer_map() size mismatch, got %zu bytes, expected %zu bytes\n", map.size, size);
			gst_buffer_unref(gstBuffer);
			return false;
		}
		
		memcpy(map.data, buffer, size);
		gst_buffer_unmap(gstBuffer, &map); 
	} 
	else
	{
		LogError(LOG_GSTREAMER "gstEncoder -- failed to map gstreamer buffer memory (%zu bytes)\n", size);
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
		LogError(LOG_GSTREAMER "gstEncoder -- appsrc pushed buffer abnormally (result %u)\n", ret);
	
	checkMsgBus();
	return true;
}


// Render
bool gstEncoder::Render( void* image, uint32_t width, uint32_t height, imageFormat format )
{
	if( !image || width == 0 || height == 0 )
		return false;

	if( mOptions.width != width || mOptions.height != height )
	{
		if( mOptions.width != 0 || mOptions.height != 0 )
			LogWarning(LOG_GSTREAMER "gstEncoder::Render() -- warning, input dimensions (%ux%u) are different than expected (%ux%u)\n", width, height, mOptions.width, mOptions.height);
		
		mOptions.width  = width;
		mOptions.height = height;

		if( mBufferCaps != NULL )
		{
			gst_object_unref(mBufferCaps);
			mBufferCaps = NULL;
		}
	}

	// error checking / return
	bool enc_success = false;

	#define render_end()	\
		const bool substreams_success = videoOutput::Render(image, width, height, format); \
		return enc_success & substreams_success;

	// allocate color conversion buffer
	const size_t i420Size = imageFormatSize(IMAGE_I420, width, height);

	if( !mBufferYUV.Alloc(2, i420Size, RingBuffer::ZeroCopy) )
	{
		LogError(LOG_GSTREAMER "gstEncoder -- failed to allocate buffers (%zu bytes each)\n", i420Size);
		enc_success = false;
		render_end();
	}

	// perform colorspace conversion
	void* nextYUV = mBufferYUV.Next(RingBuffer::Write);

	if( CUDA_FAILED(cudaConvertColor(image, format, nextYUV, IMAGE_I420, width, height)) )
	{
		LogError(LOG_GSTREAMER "gstEncoder::Render() -- unsupported image format (%s)\n", imageFormatToStr(format));
		LogError(LOG_GSTREAMER "                        supported formats are:\n");
		LogError(LOG_GSTREAMER "                            * rgb8\n");		
		LogError(LOG_GSTREAMER "                            * rgba8\n");		
		LogError(LOG_GSTREAMER "                            * rgb32f\n");		
		LogError(LOG_GSTREAMER "                            * rgba32f\n");
		
		enc_success = false;
		render_end();
	}

	CUDA(cudaDeviceSynchronize());	// TODO replace with cudaStream?
	
	// encode YUV buffer
	enc_success = encodeYUV(nextYUV, i420Size);

	// render sub-streams
	render_end();	
}


// Open
bool gstEncoder::Open()
{
	if( mStreaming )
		return true;

	// transition pipline to STATE_PLAYING
	LogInfo(LOG_GSTREAMER "gstEncoder-- starting pipeline, transitioning to GST_STATE_PLAYING\n");

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
		LogError(LOG_GSTREAMER "gstEncoder -- failed to set pipeline state to PLAYING (error %u)\n", result);
		return false;
	}

	checkMsgBus();
	usleep(100 * 1000);
	checkMsgBus();

	mStreaming = true;
	return true;
}
	

// Close
void gstEncoder::Close()
{
	if( !mStreaming )
		return;

	// send EOS
	mNeedData = false;
	
	LogInfo(LOG_GSTREAMER "gstEncoder -- shutting down pipeline, sending EOS\n");
	GstFlowReturn eos_result = gst_app_src_end_of_stream(GST_APP_SRC(mAppSrc));

	if( eos_result != 0 )
		LogError(LOG_GSTREAMER "gstEncoder -- failed sending appsrc EOS (result %u)\n", eos_result);

	sleep(1);

	// stop pipeline
	LogInfo(LOG_GSTREAMER "gstEncoder -- transitioning pipeline to GST_STATE_NULL\n");

	const GstStateChangeReturn result = gst_element_set_state(mPipeline, GST_STATE_NULL);

	if( result != GST_STATE_CHANGE_SUCCESS )
		LogError(LOG_GSTREAMER "gstEncoder -- failed to set pipeline state to NULL (error %u)\n", result);

	sleep(1);
	checkMsgBus();	
	mStreaming = false;
	LogInfo(LOG_GSTREAMER "gstEncoder -- pipeline stopped\n");
}


// checkMsgBus
void gstEncoder::checkMsgBus()
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



