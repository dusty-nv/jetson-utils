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
#include "gstWebRTC.h"

#include "RTSPServer.h"
#include "WebRTCServer.h"

#include "filesystem.h"
#include "timespec.h"
#include "logging.h"

#include "cudaColorspace.h"

#define GST_USE_UNSTABLE_API
#include <gst/webrtc/webrtc.h>
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
	mAppSrc       = NULL;
	mBus          = NULL;
	mBufferCaps   = NULL;
	mPipeline     = NULL;
	mRTSPServer   = NULL;
	mWebRTCServer = NULL;
	mNeedData     = false;

	mBufferYUV.SetThreaded(false);
}


// destructor	
gstEncoder::~gstEncoder()
{
	Close();

	if( mRTSPServer != NULL )
	{
		mRTSPServer->Release();
		mRTSPServer = NULL;
	}
	
	if( mWebRTCServer != NULL )
	{
		mWebRTCServer->Release();
		mWebRTCServer = NULL;
	}
	
	destroyPipeline();
}


// destroyPipeline
void gstEncoder::destroyPipeline()
{
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
		gst_element_set_state(mPipeline, GST_STATE_NULL);
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
	

// initPipeline
bool gstEncoder::initPipeline()
{
	// check for default codec
	if( mOptions.codec == videoOptions::CODEC_UNKNOWN )
	{
		LogWarning(LOG_GSTREAMER "gstEncoder -- codec not specified, defaulting to H.264\n");
		mOptions.codec = videoOptions::CODEC_H264;
	}

	// check if default framerate is needed
	if( mOptions.frameRate <= 0 )
		mOptions.frameRate = 30;

	// set default bitrate if needed
	if( mOptions.bitRate == 0 )
		mOptions.bitRate = 4000000; 
	
	// build pipeline string
	if( !buildLaunchStr() )
	{
		LogError(LOG_GSTREAMER "gstEncoder -- failed to build pipeline string\n");
		return false;
	}
	
	// create the pipeline
	GError* err = NULL;
	mPipeline = gst_parse_launch(mLaunchStr.c_str(), &err);

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
	
	return true;
}


// init
bool gstEncoder::init()
{
	// initialize GStreamer libraries
	if( !gstreamerInit() )
	{
		LogError(LOG_GSTREAMER "failed to initialize gstreamer API\n");
		return false;
	}

	// create GStreamer pipeline
	if( !initPipeline() )
	{
		LogError(LOG_GSTREAMER "failed to create encoder pipeline\n");
		return false;
	}

	// create servers for RTSP/WebRTC streams
	if( mOptions.resource.protocol == "rtsp" )
	{
		mRTSPServer = RTSPServer::Create(mOptions.resource.port);
		
		if( !mRTSPServer )
			return false;
		
		mRTSPServer->AddRoute(mOptions.resource.path.c_str(), mPipeline);
	}
	else if( mOptions.resource.protocol == "webrtc" )
	{
		mWebRTCServer = WebRTCServer::Create(mOptions.resource.port, mOptions.stunServer.c_str(),
									  mOptions.sslCert.c_str(), mOptions.sslKey.c_str());
		
		if( !mWebRTCServer )
			return false;
		
		mWebRTCServer->AddRoute(mOptions.resource.path.c_str(), onWebsocketMessage, this, WEBRTC_VIDEO|WEBRTC_SEND|WEBRTC_PUBLIC|WEBRTC_MULTI_CLIENT);
	}		

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
	ss << "appsrc name=mysource is-live=true do-timestamp=true format=3 ! ";  // setup appsrc input element
	
	const URI& uri = GetResource();
	std::string encoderOptions = "";

	// select the encoder
	const char* encoder = gst_select_encoder(mOptions.codec, mOptions.codecType);
	
	if( !encoder )
	{
		LogError(LOG_GSTREAMER "gstEncoder -- unsupported codec requested (%s)\n", videoOptions::CodecToStr(mOptions.codec));
		LogError(LOG_GSTREAMER "              supported encoder codecs are:\n");
		LogError(LOG_GSTREAMER "                 * h264\n");
		LogError(LOG_GSTREAMER "                 * h265\n");
		LogError(LOG_GSTREAMER "                 * vp8\n");
		LogError(LOG_GSTREAMER "                 * vp9\n");
		LogError(LOG_GSTREAMER "                 * mjpeg\n");
		
		return false;
	}
	
	// the V4L2 encoders expect NVMM memory, so use nvvidconv to convert it
	if( mOptions.codecType == videoOptions::CODEC_V4L2 && mOptions.codec != videoOptions::CODEC_MJPEG )
		ss << "nvvidconv name=vidconv ! video/x-raw(memory:NVMM) ! ";
	
	// setup the encoder and options
	ss << encoder << " name=encoder ";
	
	if( mOptions.codecType == videoOptions::CODEC_CPU )
	{
		if( mOptions.codec == videoOptions::CODEC_H264 || mOptions.codec == videoOptions::CODEC_H265 )
		{
			ss << "bitrate=" << mOptions.bitRate / 1000 << " ";	// x264enc/x265enc bitrates are in kbits
			ss << "speed-preset=ultrafast tune=zerolatency ";
			
			if( mOptions.deviceType == videoOptions::DEVICE_IP )
				ss << "key-int-max=30 insert-vui=1 ";			// send keyframes/I-frames more frequently for network streams
		}
		else if( mOptions.codec == videoOptions::CODEC_VP8 || mOptions.codec == videoOptions::CODEC_VP9 )
		{
			ss << "target-bitrate=" << mOptions.bitRate << " ";
			
			if( mOptions.deviceType == videoOptions::DEVICE_IP )
				ss << "keyframe-max-dist=30 ";
		}
	}
	else if( mOptions.codec != videoOptions::CODEC_MJPEG )
	{
		ss << "bitrate=" << mOptions.bitRate << " ";
		
		if( mOptions.deviceType == videoOptions::DEVICE_IP )
		{
			if( mOptions.codecType == videoOptions::CODEC_V4L2 )
				ss << "insert-sps-pps=1 insert-vui=1 idrinterval=30 ";
			else if( mOptions.codecType == videoOptions::CODEC_OMX )
				ss << "insert-sps-pps=1 insert-vui=1 ";
		}
		
		if( mOptions.codecType == videoOptions::CODEC_V4L2 )
			ss << "maxperf-enable=1 ";
	}

	if( mOptions.codec == videoOptions::CODEC_H264 )
		ss << "! video/x-h264 ! ";
	else if( mOptions.codec == videoOptions::CODEC_H265 )
		ss << "! video/x-h265 ! ";
	else if( mOptions.codec == videoOptions::CODEC_VP8 )
		ss << "! video/x-vp8 ! ";
	else if( mOptions.codec == videoOptions::CODEC_VP9 )
		ss << "! video/x-vp9 ! ";
	else if( mOptions.codec == videoOptions::CODEC_MJPEG )
		ss << "! image/jpeg ! ";
	
	if( mOptions.save.path.length() > 0 )
	{
		ss << "tee name=savetee savetee. ! queue ! ";
		
		if( !gst_build_filesink(mOptions.save, mOptions.codec, ss) )
			return false;

		ss << "savetee. ! queue ! ";
	}
	
	if( uri.protocol == "file" )
	{
		if( !gst_build_filesink(uri, mOptions.codec, ss) )
			return false;
	}
	else if( uri.protocol == "rtp" || uri.protocol == "rtsp" || uri.protocol == "webrtc" )
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

		if( mOptions.codec == videoOptions::CODEC_H264 || mOptions.codec == videoOptions::CODEC_H265 ) 
			ss << " config-interval=1";	// aggregate-mode=zero-latency";
		
		if( uri.protocol == "rtsp" )
			ss << " name=pay0";	 // GstRTSPServer expects the payloaders to be named pay0, pay1, ect
		else
			ss << " ! ";
		
		if( uri.protocol == "rtp" )
		{
			ss << "udpsink host=" << uri.location << " ";

			if( uri.port != 0 )
				ss << "port=" << uri.port;

			ss << " auto-multicast=true";
		}
		else if( uri.protocol == "webrtc" )
		{
			ss << "application/x-rtp,media=video,encoding-name=" << videoOptions::CodecToStr(mOptions.codec) << ",clock-rate=90000,payload=96 ! ";
			ss << "tee name=videotee ! queue ! fakesink";  // webrtcbin's will be added when clients connect
		}
	}
	else if( uri.protocol == "rtpmp2ts" )
	{
		// https://forums.developer.nvidia.com/t/gstreamer-udp-to-vlc/215349/5
		if( mOptions.codec == videoOptions::CODEC_H264 ) 
			ss << "h264parse config-interval=1 ! mpegtsmux ! rtpmp2tpay ! udpsink host=";
		else if (mOptions.codec == videoOptions::CODEC_H265 )
			ss << "h265parse config-interval=1 ! mpegtsmux ! rtpmp2tpay ! udpsink host=";
		else
		{
			LogError(LOG_GSTREAMER "gstEncoder -- rtpmp2ts output only supports h264 and h265. Unsupported codec (%s)\n", uri.extension.c_str());
			return false;
		}
 		
		ss << uri.location << " ";

		if( uri.port != 0 )
			ss << "port=" << uri.port;

		ss << " auto-multicast=true";
	}
	else if( uri.protocol == "rtmp" )
	{
		ss << "flvmux streamable=true ! queue ! rtmpsink location=";
		ss << uri.string << " ";
	}
	else
	{
		LogError(LOG_GSTREAMER "gstEncoder -- invalid protocol (%s)\n", uri.protocol.c_str());
		return false;
	}

	mLaunchStr = ss.str();

	LogInfo(LOG_GSTREAMER "gstEncoder -- pipeline launch string:\n");
	LogInfo(LOG_GSTREAMER "%s\n", mLaunchStr.c_str());

	return true;
}


// onNeedData
void gstEncoder::onNeedData( GstElement* pipeline, guint size, gpointer user_data )
{
	//LogDebug(LOG_GSTREAMER "gstEncoder -- appsrc requesting data (%u bytes)\n", size);
	
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
	// 20240307 - disabling this because with WebRTC sometimes it gets stuck in 'pipeline full' state
	/*if( !mNeedData )
	{
		if( mOptions.frameCount % 25 == 0 )
			LogVerbose(LOG_GSTREAMER "gstEncoder -- pipeline full, skipping frame %zu (%ux%u, %zu bytes)\n", mOptions.frameCount, mOptions.width, mOptions.height, size);
		
		return true;
	}*/

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
	while( true )
	{
		GstFlowReturn ret;	
		g_signal_emit_by_name(mAppSrc, "push-buffer", gstBuffer, &ret);
		
		if( ret >= 0 )
		{
			gst_buffer_unref(gstBuffer);
			break;
		}
		
		LogError(LOG_GSTREAMER "gstEncoder -- an error occurred pushing appsrc buffer (result=%i '%s')\n", (int)ret, gst_flow_get_name(ret));
		
		// check to make sure the pipeline is still playing (some pipelines like RTSP server may disconnect)
		GstState state = GST_STATE_VOID_PENDING;
		gst_element_get_state(mPipeline, &state, NULL, GST_CLOCK_TIME_NONE);
	
		if( state != GST_STATE_PLAYING )
		{
			LogError(LOG_GSTREAMER "gstEncoder -- pipeline is in the '%s' state, restarting pipeline...\n", gst_element_state_get_name(state));
			
			mStreaming = false;
			
			if( !Open() )
			{
				gst_buffer_unref(gstBuffer);
				return false;
			}
		}
	}
	
	checkMsgBus();
	return true;
}


// Render
bool gstEncoder::Render( void* image, uint32_t width, uint32_t height, imageFormat format, cudaStream_t stream )
{	
	// update the webrtc server if needed
	if( mWebRTCServer != NULL && !mWebRTCServer->IsThreaded() )
		mWebRTCServer->ProcessRequests();	
	
	// increment frame counter
	mOptions.frameCount += 1;
		
	// verify image dimensions
	if( !image || width == 0 || height == 0 )
		return false;

	if( mOptions.width != width || mOptions.height != height )
	{
		if( mOptions.width != 0 || mOptions.height != 0 )
			LogWarning(LOG_GSTREAMER "gstEncoder -- resolution changing from (%ux%u) to (%ux%u)\n", mOptions.width, mOptions.height, width, height);
		
		mOptions.width  = width;
		mOptions.height = height;

		if( mBufferCaps != NULL )
		{
			gst_object_unref(mBufferCaps);
			mBufferCaps = NULL;
			
			destroyPipeline();
		
			mStreaming = false;
			
			if( !initPipeline() || !Open() )
			{
				LogError(LOG_GSTREAMER "failed to re-initialize encoder with new dimensions (%ux%u)\n", width, height);
				return false;
			}
		}
		
		/*// nvbufsurface: NvBufSurfaceCopy: buffer param mismatch
		GstElement* vidconv = gst_bin_get_by_name(GST_BIN(mPipeline), "vidconv");
		GstElement* encoder = gst_bin_get_by_name(GST_BIN(mPipeline), "encoder");
		
		if( vidconv != NULL && encoder != NULL )
		{
			gst_element_set_state(mAppSrc, GST_STATE_NULL);
			gst_element_set_state(vidconv, GST_STATE_NULL);
			gst_element_set_state(encoder, GST_STATE_NULL);
			gst_element_set_state(mAppSrc, GST_STATE_PLAYING);
			gst_element_set_state(vidconv, GST_STATE_PLAYING);
			gst_element_set_state(encoder, GST_STATE_PLAYING);
			gst_object_unref(vidconv);
			gst_object_unref(encoder);
			usleep(500*1000);
			checkMsgBus();
		}*/
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

	if( CUDA_FAILED(cudaConvertColor(image, format, nextYUV, IMAGE_I420, width, height, stream)) )
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

    if( stream != 0 )
        CUDA(cudaStreamSynchronize(stream));
    else
	    CUDA(cudaDeviceSynchronize());
	
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
	LogInfo(LOG_GSTREAMER "gstEncoder -- starting pipeline, transitioning to GST_STATE_PLAYING\n");

	const GstStateChangeReturn result = gst_element_set_state(mPipeline, GST_STATE_PLAYING);

	if( result == GST_STATE_CHANGE_ASYNC )
	{
		LogDebug(LOG_GSTREAMER "gstEncoder -- queued state to GST_STATE_PLAYING => GST_STATE_CHANGE_ASYNC\n");
		
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


// onWebsocketMessage
void gstEncoder::onWebsocketMessage( WebRTCPeer* peer, const char* message, size_t message_size, void* user_data )
{
	if( !user_data )
		return;
	
	gstEncoder* encoder = (gstEncoder*)user_data;
	gstWebRTC::PeerContext* peer_context = (gstWebRTC::PeerContext*)peer->user_data;
	
	if( peer->flags & WEBRTC_PEER_CONNECTING )
	{
		LogVerbose(LOG_WEBRTC "new WebRTC peer connecting (%s, peer_id=%u)\n", peer->ip_address.c_str(), peer->ID);
		
		// new peer context
		peer_context = new gstWebRTC::PeerContext();
		peer->user_data = peer_context;
		
		// create a new queue element
		gchar* tmp = g_strdup_printf("queue-%u", peer->ID);
		peer_context->queue = gst_element_factory_make("queue", tmp);
		g_assert_nonnull(peer_context->queue);
		gst_object_ref(peer_context->queue);
		g_free(tmp);
		
		// create a new webrtcbin element
		tmp = g_strdup_printf("webrtcbin-%u", peer->ID);
		peer_context->webrtcbin = gst_element_factory_make("webrtcbin", tmp);
		g_assert_nonnull(peer_context->webrtcbin);
		gst_object_ref(peer_context->webrtcbin);
		g_free(tmp);
		
		// set webrtcbin properties
		const char* stun_server = peer->server->GetSTUNServer();
		
		if( stun_server != NULL && strlen(stun_server) > 0 )
		{
		    std::string stun_url = std::string("stun://") + stun_server;
		    g_object_set(peer_context->webrtcbin, "stun-server", stun_url.c_str(), NULL);
		}
		
		g_object_set(peer_context->webrtcbin, "latency", encoder->mOptions.latency, NULL);   // this doesn't seem to have an impact?
	
		// set latency on the rtpbin (https://github.com/centricular/gstwebrtc-demos/issues/102#issuecomment-575157321)
		GstElement* rtpbin = gst_bin_get_by_name(GST_BIN(peer_context->webrtcbin), "rtpbin");
		g_assert_nonnull(rtpbin);
		g_object_set(rtpbin, "latency", encoder->mOptions.latency, NULL);
		gst_object_unref(rtpbin);
		
		// add queue and webrtcbin elements to the pipeline
		gst_bin_add_many(GST_BIN(encoder->mPipeline), peer_context->queue, peer_context->webrtcbin, NULL);
		
		// link the queue to webrtc bin
		GstPad* srcpad = gst_element_get_static_pad(peer_context->queue, "src");
		g_assert_nonnull(srcpad);
		GstPad* sinkpad = gst_element_get_request_pad(peer_context->webrtcbin, "sink_%u");
		g_assert_nonnull(sinkpad);
		int ret = gst_pad_link(srcpad, sinkpad);
		g_assert_cmpint(ret, ==, GST_PAD_LINK_OK);
		gst_object_unref(srcpad);
		gst_object_unref(sinkpad);
		
		// link the queue to the tee
		GstElement* tee = gst_bin_get_by_name(GST_BIN(encoder->mPipeline), "videotee");
		g_assert_nonnull(tee);
		srcpad = gst_element_get_request_pad(tee, "src_%u");
		g_assert_nonnull(srcpad);
		gst_object_unref(tee);
		sinkpad = gst_element_get_static_pad(peer_context->queue, "sink");
		g_assert_nonnull(sinkpad);
		ret = gst_pad_link(srcpad, sinkpad);
		g_assert_cmpint(ret, ==, GST_PAD_LINK_OK);
		gst_object_unref(srcpad);
		gst_object_unref(sinkpad);
		
		// set transciever to send-only mode
		GArray* transceivers = NULL;
		
		g_signal_emit_by_name(peer_context->webrtcbin, "get-transceivers", &transceivers);
		g_assert(transceivers != NULL && transceivers->len > 0);
		
		GstWebRTCRTPTransceiver* transceiver = g_array_index(transceivers, GstWebRTCRTPTransceiver*, 0);
		g_object_set(transceiver, "direction", GST_WEBRTC_RTP_TRANSCEIVER_DIRECTION_SENDONLY, NULL);
		g_array_unref(transceivers);
		
		// subscribe to callbacks
		g_signal_connect(peer_context->webrtcbin, "on-negotiation-needed", G_CALLBACK(gstWebRTC::onNegotiationNeeded), peer);
		g_signal_connect(peer_context->webrtcbin, "on-ice-candidate", G_CALLBACK(gstWebRTC::onIceCandidate), peer);
		
		// Set to pipeline branch to PLAYING
		ret = gst_element_sync_state_with_parent(peer_context->queue);
		g_assert_true(ret);
		ret = gst_element_sync_state_with_parent(peer_context->webrtcbin);
		g_assert_true(ret);
		
		return;
	}
	else if( peer->flags & WEBRTC_PEER_CLOSED )
	{
		LogVerbose(LOG_WEBRTC "WebRTC peer disconnected (%s, peer_id=%u)\n", peer->ip_address.c_str(), peer->ID);
		
		// remove webrtcbin from pipeline
		gst_bin_remove(GST_BIN(encoder->mPipeline), peer_context->webrtcbin);
		gst_element_set_state(peer_context->webrtcbin, GST_STATE_NULL);
		gst_object_unref(peer_context->webrtcbin);

		// disconnect queue pads
		GstPad* sinkpad = gst_element_get_static_pad(peer_context->queue, "sink");
		g_assert_nonnull(sinkpad);
		GstPad* srcpad = gst_pad_get_peer(sinkpad);
		g_assert_nonnull(srcpad);
		gst_object_unref(sinkpad);
  
		// remove queue from pipeline
		gst_bin_remove(GST_BIN(encoder->mPipeline), peer_context->queue);
		gst_element_set_state(peer_context->queue, GST_STATE_NULL);
		gst_object_unref(peer_context->queue);

		// free encoder-specific context
		delete peer_context;
		peer->user_data = NULL;
		
		return;
	}
	
	gstWebRTC::onWebsocketMessage(peer, message, message_size, user_data);
}
