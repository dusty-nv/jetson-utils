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
#include "WebRTCServer.h"

#include "filesystem.h"
#include "timespec.h"
#include "logging.h"

#include "cudaColorspace.h"

#define GST_USE_UNSTABLE_API

#include <gst/gst.h>
#include <gst/app/gstappsrc.h>
#include <gst/webrtc/webrtc.h>

#include <json-glib/json-glib.h>

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
	mWebRTCServer = NULL;
	mNeedData     = false;

	mBufferYUV.SetThreaded(false);
}


// destructor	
gstEncoder::~gstEncoder()
{
	Close();

	if( mWebRTCServer != NULL )
	{
		mWebRTCServer->Release();
		mWebRTCServer = NULL;
	}
	
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

	// create server for WebRTC streams
	if( mOptions.resource.protocol == "webrtc" )
	{
		mWebRTCServer = WebRTCServer::Create(mOptions.resource.port);
		
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

	// setup appsrc input element
	ss << "appsrc name=mysource is-live=true do-timestamp=true format=3 ! ";

	// set default bitrate (if needed)
	if( mOptions.bitRate == 0 )
		mOptions.bitRate = 4000000; 
	
	// determine the requested protocol to use
	const URI& uri = GetResource();
	
	//ss << mCapsStr << " ! ";

#ifdef GST_CODECS_V4L2
	// the V4L2 encoders expect NVMM memory, so use nvvidconv to convert it
	if( mOptions.codec != videoOptions::CODEC_MJPEG )
		ss << "nvvidconv ! video/x-raw(memory:NVMM) ! ";
#endif
	
	// select hardware codec to use
	if( mOptions.codec == videoOptions::CODEC_H264 )
		ss << GST_ENCODER_H264 << " bitrate=" << mOptions.bitRate << " ! video/x-h264 !  ";	// TODO:  investigate quality-level setting
	else if( mOptions.codec == videoOptions::CODEC_H265 )
		ss << GST_ENCODER_H265 << " bitrate=" << mOptions.bitRate << " ! video/x-h265 ! ";
	else if( mOptions.codec == videoOptions::CODEC_VP8 )
		ss << GST_ENCODER_VP8 << " bitrate=" << mOptions.bitRate << " ! video/x-vp8 ! ";
	else if( mOptions.codec == videoOptions::CODEC_VP9 )
		ss << GST_ENCODER_VP9 << " bitrate=" << mOptions.bitRate << " ! video/x-vp9 ! ";
	else if( mOptions.codec == videoOptions::CODEC_MJPEG )
		ss << GST_ENCODER_MJPEG << " ! image/jpeg ! ";
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
	else if( uri.protocol == "rtp" || uri.protocol == "webrtc" )
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
			ss << " config-interval=1";
		
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
			ss << "application/x-rtp,media=video,encoding-name=" << videoOptions::CodecToStr(mOptions.codec) << ",payload=96 ! ";
			//ss << "webrtcbin name=webrtcbin stun-server=" << STUN_SERVER;
			ss << "tee name=videotee ! queue ! fakesink";  // webrtcbin's will be added when clients connect
		}
		
		mOptions.deviceType = videoOptions::DEVICE_IP;
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

		mOptions.deviceType = videoOptions::DEVICE_IP;
	}
	else if( uri.protocol == "rtmp" )
	{
		ss << "flvmux streamable=true ! queue ! rtmpsink location=";
		ss << uri.string << " ";

		mOptions.deviceType = videoOptions::DEVICE_IP;
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
	if( mWebRTCServer != NULL )
		mWebRTCServer->ProcessRequests();	// update the webrtc server
	
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


// gstreamer-specific context for each WebRTC peer
struct gstWebRTCPeerContext
{
	GstElement* webrtcbin;
	GstElement* queue;
	gstEncoder* encoder;
};


// onWebsocketMessage
void gstEncoder::onWebsocketMessage( WebRTCPeer* peer, const char* message, size_t message_size, void* user_data )
{
	if( !user_data )
		return;
	
	gstEncoder* encoder = (gstEncoder*)user_data;
	gstWebRTCPeerContext* peer_context = (gstWebRTCPeerContext*)peer->user_data;
	
	if( peer->flags & WEBRTC_PEER_CONNECTING )
	{
		LogVerbose(LOG_WEBRTC "new WebRTC peer connecting (%s, peer_id=%u)\n", peer->ip_address.c_str(), peer->ID);
		
		// new peer context
		peer_context = new gstWebRTCPeerContext();
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
		std::string stun_server = std::string("stun://") + peer->server->GetSTUNServer();
		g_object_set(peer_context->webrtcbin, "stun-server", stun_server.c_str(), NULL);
		//g_object_set(peer_context->webrtcbin, "latency", 40, NULL);   // this doesn't seem to have an impact
	
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
		g_signal_connect(peer_context->webrtcbin, "on-negotiation-needed", G_CALLBACK(onNegotiationNeeded), peer);
		g_signal_connect(peer_context->webrtcbin, "on-ice-candidate", G_CALLBACK(onIceCandidate), peer);
		
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
	
	#define cleanup() { \
		if( json_parser != NULL ) \
			g_object_unref(G_OBJECT(json_parser)); \
		return; } \

	#define unknown_message() { \
		LogWarning(LOG_WEBRTC "gstEncoder -- unknown message, ignoring...\n%s\n", message); \
		cleanup(); }

	// parse JSON data string
	JsonParser* json_parser = json_parser_new();
	
	if( !json_parser_load_from_data(json_parser, message, -1, NULL) )
		unknown_message();

	JsonNode* root_json = json_parser_get_root(json_parser);
	
	if( !JSON_NODE_HOLDS_OBJECT(root_json) )
		unknown_message();

	JsonObject* root_json_object = json_node_get_object(root_json);

	// retrieve type string
	if( !json_object_has_member(root_json_object, "type") ) 
	{
		LogError(LOG_WEBRTC "received JSON message without 'type' field\n");
		cleanup();
	}
	
	const gchar* type_string = json_object_get_string_member(root_json_object, "type");

	// retrieve data object
	if( !json_object_has_member(root_json_object, "data") ) 
	{
		LogError(LOG_WEBRTC "received JSON message without 'data' field\n");
		cleanup();
	}
	
	JsonObject* data_json_object = json_object_get_object_member(root_json_object, "data");

	// handle message types
	if( g_strcmp0(type_string, "sdp") == 0 ) 
	{
		// validate SDP message
		if( !json_object_has_member(data_json_object, "type") ) 
		{
			LogError(LOG_WEBRTC "received SDP message without 'type' field\n");
			cleanup();
		}
		
		const gchar* sdp_type_string = json_object_get_string_member(data_json_object, "type");

		if( g_strcmp0(sdp_type_string, "answer") != 0 ) 
		{
			LogError(LOG_WEBRTC "expected SDP message type 'answer', got '%s'\n", sdp_type_string);
			cleanup();
		}

		if( !json_object_has_member(data_json_object, "sdp") )
		{
			LogError(LOG_WEBRTC "received SDP message without 'sdp' field\n");
			cleanup();
		}
		
		const gchar* sdp_string = json_object_get_string_member(data_json_object, "sdp");
		LogVerbose(LOG_WEBRTC "received SDP message for %s from %s (peer_id=%u)\n%s\n", peer->path.c_str(), peer->ip_address.c_str(), peer->ID, sdp_string);
		
		// parse SDP string
		GstSDPMessage* sdp = NULL;
		int ret = gst_sdp_message_new(&sdp);
		g_assert_cmphex(ret, ==, GST_SDP_OK);

		ret = gst_sdp_message_parse_buffer((guint8*)sdp_string, strlen(sdp_string), sdp);
		
		if( ret != GST_SDP_OK )
		{
			LogError(LOG_WEBRTC "failed to parse SDP string\n");
			cleanup();
		}

		// provide the SDP to webrtcbin
		GstWebRTCSessionDescription* answer = gst_webrtc_session_description_new(GST_WEBRTC_SDP_TYPE_ANSWER, sdp);
		g_assert_nonnull(answer);

		GstPromise* promise = gst_promise_new();
		g_signal_emit_by_name(peer_context->webrtcbin, "set-remote-description", answer, promise);
		gst_promise_interrupt(promise);
		gst_promise_unref(promise);
		gst_webrtc_session_description_free(answer);
		
	} 
	else if( g_strcmp0(type_string, "ice") == 0 )
	{
		// validate ICE message
		if( !json_object_has_member(data_json_object, "sdpMLineIndex") )
		{
			LogError(LOG_WEBRTC "received ICE message without 'sdpMLineIndex' field\n");
			cleanup();
		}
		
		const uint32_t mline_index = json_object_get_int_member(data_json_object, "sdpMLineIndex");

		// extract the ICE candidate
		if( !json_object_has_member(data_json_object, "candidate") ) 
		{
			LogError(LOG_WEBRTC "received ICE message without 'candidate' field\n");
			cleanup();
		}
		
		const gchar* candidate_string = json_object_get_string_member(data_json_object, "candidate");

		LogVerbose(LOG_WEBRTC "received ICE message on %s from %s (peer_id=%u) with mline index %u; candidate: \n%s\n", peer->path.c_str(), peer->ip_address.c_str(), peer->ID, mline_index, candidate_string);

		// provide the ICE candidate to webrtcbin
		g_signal_emit_by_name(peer_context->webrtcbin, "add-ice-candidate", mline_index, candidate_string);
	} 
	else
		unknown_message();

	cleanup();
}


// get_string_from_json_object
static gchar* get_string_from_json_object(JsonObject* object)
{
	JsonNode* root = json_node_init_object (json_node_alloc (), object);
	JsonGenerator* generator = json_generator_new ();
	json_generator_set_root (generator, root);
	gchar* text = json_generator_to_data (generator, NULL);

	g_object_unref(generator);
	json_node_free(root);
	
	return text;
}


// onNegotiationNeeded
void gstEncoder::onNegotiationNeeded( GstElement* webrtcbin, void* user_data )
{
	LogDebug(LOG_WEBRTC "gstEncoder -- onNegotiationNeeded()\n");
	
	if( !user_data )
		return;
	
	WebRTCPeer* peer = (WebRTCPeer*)user_data;
	gstWebRTCPeerContext* peer_context = (gstWebRTCPeerContext*)peer->user_data;
	
	// setup offer created callback
	GstPromise* promise = gst_promise_new_with_change_func(onCreateOffer, peer, NULL);
	g_signal_emit_by_name(G_OBJECT(peer_context->webrtcbin), "create-offer", NULL, promise);
}


// onOfferCreated
void gstEncoder::onCreateOffer( GstPromise* promise, void* user_data )
{
	LogDebug(LOG_WEBRTC "gstEncoder -- onCreateOffer()\n");
	
	if( !user_data )
		return;
	
	WebRTCPeer* peer = (WebRTCPeer*)user_data;
	gstWebRTCPeerContext* peer_context = (gstWebRTCPeerContext*)peer->user_data;

	// send the SDP offer
	const GstStructure* reply = gst_promise_get_reply(promise);
	
	GstWebRTCSessionDescription* offer = NULL;
	gst_structure_get(reply, "offer", GST_TYPE_WEBRTC_SESSION_DESCRIPTION, &offer, NULL);
	gst_promise_unref(promise);

	GstPromise* local_desc_promise = gst_promise_new();
	g_signal_emit_by_name(peer_context->webrtcbin, "set-local-description", offer, local_desc_promise);
	gst_promise_interrupt(local_desc_promise);
	gst_promise_unref(local_desc_promise);

	gchar* sdp_string = gst_sdp_message_as_text(offer->sdp);
	LogVerbose(LOG_WEBRTC "negotiation offer created:\n%s\n", sdp_string);

	JsonObject* sdp_json = json_object_new();
	json_object_set_string_member(sdp_json, "type", "sdp");

	JsonObject* sdp_data_json = json_object_new ();
	json_object_set_string_member(sdp_data_json, "type", "offer");
	json_object_set_string_member(sdp_data_json, "sdp", sdp_string);
	json_object_set_object_member(sdp_json, "data", sdp_data_json);

	gchar* json_string = get_string_from_json_object(sdp_json);
	json_object_unref(sdp_json);

	LogVerbose(LOG_WEBRTC "sending offer for %s to %s (peer_id=%u): \n%s\n", peer->path.c_str(), peer->ip_address.c_str(), peer->ID, json_string);
	
	soup_websocket_connection_send_text(peer->connection, json_string);
	
	//g_free(json_string);
	g_free(sdp_string);
	gst_webrtc_session_description_free(offer);
}


// onIceCandidate
void gstEncoder::onIceCandidate( GstElement* webrtcbin, uint32_t mline_index, char* candidate, void* user_data )
{
	LogDebug(LOG_WEBRTC "gstEncoder -- onIceCandidate()\n");
	
	if( !user_data )
		return;
	
	WebRTCPeer* peer = (WebRTCPeer*)user_data;
	gstWebRTCPeerContext* peer_context = (gstWebRTCPeerContext*)peer->user_data;

	// send the ICE candidate
	JsonObject* ice_json = json_object_new();
	json_object_set_string_member(ice_json, "type", "ice");

	JsonObject* ice_data_json = json_object_new();
	json_object_set_int_member(ice_data_json, "sdpMLineIndex", mline_index);
	json_object_set_string_member(ice_data_json, "candidate", candidate);
	json_object_set_object_member(ice_json, "data", ice_data_json);

	gchar* json_string = get_string_from_json_object(ice_json);
	json_object_unref(ice_json);

	LogVerbose(LOG_WEBRTC "sending ICE candidate for %s to %s (peer_id=%u): \n%s\n", peer->path.c_str(), peer->ip_address.c_str(), peer->ID, json_string);

	soup_websocket_connection_send_text(peer->connection, json_string);
	
	g_free(json_string);
}

