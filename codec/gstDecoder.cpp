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
#include "cudaColorspace.h"

#include "logging.h"
#include "filesystem.h"

#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <gst/pbutils/pbutils.h>

#include <sstream>
#include <unistd.h>
#include <string.h>
#include <strings.h>


// 
// RTP test source pipeline:
//  (from PC)     $ gst-launch-1.0 -v videotestsrc ! video/x-raw,width=300,height=300,framerate=30/1 ! x264enc ! rtph264pay ! udpsink host=127.0.0.1 port=5000
//  (from Jetson) $ ./video-viewer test.mkv rtp://@:5000
// 
// RTP test recieve pipeline:
//  (from PC)     $ gst-launch-1.0 -v udpsrc port=5000 caps = "application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)H264, payload=(int)96" ! rtph264depay ! decodebin ! videoconvert ! autovideosink
//  (from Jetson) $ ./video-viewer rtp://@:5000
//  (from VLC)    SDP file contents:
//                c=IN IP4 127.0.0.1
//                m=video 5000 RTP/AVP 96
//                a=rtpmap:96 H264/90000
//
// RSTP test server installation:
//  $ git clone https://github.com/GStreamer/gst-rtsp-server && cd gst-rtsp-server
//  $ git checkout 1.14.5
//  $ ./autogen.sh --noconfigure && ./configure && make
//  $ cd examples && ./test-launch "( videotestsrc ! x264enc ! rtph264pay name=pay0 pt=96 )"
//  > rtsp://127.0.0.1:8554/test
//
// RTSP authentication test:
//  $ ./test-auth
//  > rtsp://user:password@127.0.0.1:8554/test
//  > rtsp://admin:power@127.0.0.1:8554/test
//


// supported image file extensions
const char* gstDecoder::SupportedExtensions[] = { "mkv", "mp4", "qt", 
										"flv", "avi", "h264", 
										"h265", "mov", "webm", NULL };

bool gstDecoder::IsSupportedExtension( const char* ext )
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
gstDecoder::gstDecoder( const videoOptions& options ) : videoSource(options)
{	
	mAppSink    = NULL;
	mBus        = NULL;
	mPipeline   = NULL;
	mCustomSize = false;
	mCustomRate = false;
	mEOS        = false;
	mLoopCount  = 1;

	mBufferManager = new gstBufferManager(&mOptions);
}


// destructor
gstDecoder::~gstDecoder()
{
	Close();

	if( mAppSink != NULL )
	{
		gst_object_unref(mAppSink);
		mAppSink = NULL;
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
	
	SAFE_DELETE(mBufferManager);
}


// Create
gstDecoder* gstDecoder::Create( const videoOptions& options )
{
	gstDecoder* dec = new gstDecoder(options);

	if( !dec )
		return NULL;

	if( !dec->init() )
	{
		LogError(LOG_GSTREAMER "gstDecoder -- failed to create decoder for %s\n", dec->mOptions.resource.string.c_str());
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
		LogError(LOG_GSTREAMER "failed to initialize gstreamer API\n");
		return NULL;
	}

	// first, check that the file exists
	if( mOptions.resource.protocol == "file" )
	{
		if( !fileExists(mOptions.resource.location) )
		{
			LogError(LOG_GSTREAMER "gstDecoder -- couldn't find file '%s'\n", mOptions.resource.location.c_str());
			return false;
		}
	}
	
	LogInfo(LOG_GSTREAMER "gstDecoder -- creating decoder for %s\n", mOptions.resource.location.c_str());

	// flag if the user wants a specific resolution and framerate
	if( mOptions.width != 0 || mOptions.height != 0 )
		mCustomSize = true;

	if( mOptions.frameRate != 0 )
		mCustomRate = true;

	// discover resource stats
	if( !discover() )
	{
		if( mOptions.resource.protocol == "rtp" )
			LogWarning(LOG_GSTREAMER "gstDecoder -- resource discovery not supported for RTP streams\n");		
		else
			LogError(LOG_GSTREAMER "gstDecoder -- resource discovery and auto-negotiation failed\n");

		if( mOptions.codec == videoOptions::CODEC_UNKNOWN )
		{
			LogError(LOG_GSTREAMER "gstDecoder -- try manually setting the codec with the --input-codec option\n");
			return false;
		}
	}
	
	// build pipeline string
	if( !buildLaunchStr() )
	{
		LogError(LOG_GSTREAMER "gstDecoder -- failed to build pipeline string\n");
		return false;
	}

	// create pipeline
	mPipeline = gst_parse_launch(mLaunchStr.c_str(), &err);

	if( err != NULL )
	{
		LogError(LOG_GSTREAMER "gstDecoder -- failed to create pipeline\n");
		LogError(LOG_GSTREAMER "   (%s)\n", err->message);
		g_error_free(err);
		return false;
	}

	GstPipeline* pipeline = GST_PIPELINE(mPipeline);

	if( !pipeline )
	{
		LogError(LOG_GSTREAMER "gstDecoder -- failed to cast GstElement into GstPipeline\n");
		return false;
	}	

	// retrieve pipeline bus
	/*GstBus**/ mBus = gst_pipeline_get_bus(pipeline);

	if( !mBus )
	{
		LogError(LOG_GSTREAMER "gstDecoder -- failed to retrieve GstBus from pipeline\n");
		return false;
	}

	// add watch for messages (disabled when we poll the bus ourselves, instead of gmainloop)
	//gst_bus_add_watch(mBus, (GstBusFunc)gst_message_print, NULL);

	// get the appsrc
	GstElement* appsinkElement = gst_bin_get_by_name(GST_BIN(pipeline), "mysink");
	GstAppSink* appsink = GST_APP_SINK(appsinkElement);

	if( !appsinkElement || !appsink)
	{
		LogError(LOG_GSTREAMER "gstDecoder -- failed to retrieve AppSink element from pipeline\n");
		return false;
	}
	
	mAppSink = appsink;

	// setup callbacks
	GstAppSinkCallbacks cb;
	memset(&cb, 0, sizeof(GstAppSinkCallbacks));
	
	cb.eos         = onEOS;
	cb.new_preroll = onPreroll;	// disabled b/c preroll sometimes occurs during Close() and crashes
#if GST_CHECK_VERSION(1,0,0)
	cb.new_sample  = onBuffer;
#else
	cb.new_buffer  = onBuffer;
#endif
	
	gst_app_sink_set_callbacks(mAppSink, &cb, (void*)this, NULL);
	
	return true;
}


// findVideoStreamInfo
static GstDiscovererVideoInfo* findVideoStreamInfo( GstDiscovererStreamInfo* info )
{
	if( !info )
		return NULL;
	
	//printf("stream type -- %s\n", gst_discoverer_stream_info_get_stream_type_nick(info));
	
	if( GST_IS_DISCOVERER_VIDEO_INFO(info) )
	{
		return GST_DISCOVERER_VIDEO_INFO(info);
	}
	else if( GST_IS_DISCOVERER_CONTAINER_INFO(info) )
	{
		GstDiscovererContainerInfo* containerInfo = GST_DISCOVERER_CONTAINER_INFO(info);
	
		if( !containerInfo )
			return NULL;

		GList* containerStreams = gst_discoverer_container_info_get_streams(containerInfo);
			
		for( GList* n=containerStreams; n; n = n->next )
		{
			GstDiscovererVideoInfo* videoStream = findVideoStreamInfo(GST_DISCOVERER_STREAM_INFO(n->data));
			
			if( videoStream != NULL )
				return videoStream;
		}
	}
	
	return findVideoStreamInfo(gst_discoverer_stream_info_get_next(info));
}


// discover
bool gstDecoder::discover()
{
	// RTP streams can't be discovered
	if( mOptions.resource.protocol == "rtp" )
		return false;

	// create a new discovery interface
	GError* err = NULL;
	GstDiscoverer* discoverer = gst_discoverer_new(5 * GST_SECOND, &err);
	
	if( !discoverer )
	{
		LogError(LOG_GSTREAMER "gstDecoder -- failed to create gstreamer discovery instance:  %s\n", err->message);
		return false;
	}
	
	GstDiscovererInfo* info = gst_discoverer_discover_uri(discoverer,
                             mOptions.resource.string.c_str(), &err);
    
	if( !info || err != NULL )
	{
		LogError(LOG_GSTREAMER "gstDecoder -- %s\n", err->message);
		return false;
	}
	
	GstDiscovererStreamInfo* rootStream = gst_discoverer_info_get_stream_info(info);
	
	if( !rootStream )
	{
		LogError(LOG_GSTREAMER "gstDecoder -- failed to discover stream info\n");
		return false;
	}

	GstDiscovererVideoInfo* videoInfo = findVideoStreamInfo(rootStream);
	GstDiscovererStreamInfo* streamInfo = GST_DISCOVERER_STREAM_INFO(videoInfo);
	
	if( !videoInfo )
	{
		LogError(LOG_GSTREAMER "gstDecoder -- failed to discover any video streams\n");
		return false;
	}
	
	// retrieve video resolution
	guint width  = gst_discoverer_video_info_get_width(videoInfo);
	guint height = gst_discoverer_video_info_get_height(videoInfo);
	if( mOptions.flipMethod == videoOptions::FLIP_CLOCKWISE || mOptions.flipMethod == videoOptions::FLIP_COUNTERCLOCKWISE
		|| mOptions.flipMethod == videoOptions::FLIP_UPPER_LEFT_DIAGONAL || mOptions.flipMethod == videoOptions::FLIP_UPPER_RIGHT_DIAGONAL )
	{
		const guint prevWidth = width;
		width = height;
		height = prevWidth;
	}
	
	const float framerate_num   = gst_discoverer_video_info_get_framerate_num(videoInfo);
	const float framerate_denom = gst_discoverer_video_info_get_framerate_denom(videoInfo);
	const float framerate       = framerate_num / framerate_denom;

	LogVerbose(LOG_GSTREAMER "gstDecoder -- discovered video resolution: %ux%u  (framerate %f Hz)\n", width, height, framerate);
	
	// disable re-scaling if the user's custom size matches the feed's
	if( mCustomSize && mOptions.width == width && mOptions.height == height )
		mCustomSize = false;

	if( mOptions.width == 0 )
		mOptions.width = width;

	if( mOptions.height == 0 )
		mOptions.height = height;
	
	// confirm the desired framerate against what the stream provides
	if( mCustomRate )
	{
		// disable rate-limiting if the user's custom rate matches the feed's
		if( mOptions.frameRate == framerate )
			mCustomRate = false;
	}
	else
	{
		// otherwise adopt the feed's framerate
		mOptions.frameRate = framerate;
	}

	// retrieve video caps
	GstCaps* caps = gst_discoverer_stream_info_get_caps(streamInfo);
	
	if( !caps )
	{
		LogError(LOG_GSTREAMER "gstDecoder -- failed to discover video caps\n");
		return false;
	}
	
	const std::string videoCaps = gst_caps_to_string(caps);
	
	LogVerbose(LOG_GSTREAMER "gstDecoder -- discovered video caps:  %s\n", videoCaps.c_str());

	// parse codec
	if( videoCaps.find("video/x-h264") != std::string::npos )
		mOptions.codec = videoOptions::CODEC_H264;
	else if( videoCaps.find("video/x-h265") != std::string::npos )
		mOptions.codec = videoOptions::CODEC_H265;
	else if( videoCaps.find("video/x-vp8") != std::string::npos )
		mOptions.codec = videoOptions::CODEC_VP8;
	else if( videoCaps.find("video/x-vp9") != std::string::npos )
		mOptions.codec = videoOptions::CODEC_VP9;
	else if( videoCaps.find("image/jpeg") != std::string::npos )
		mOptions.codec = videoOptions::CODEC_MJPEG;
	else if( videoCaps.find("video/mpeg") != std::string::npos )
	{
		if( videoCaps.find("mpegversion=(int)4") != std::string::npos )
			mOptions.codec = videoOptions::CODEC_MPEG4;
		else if( videoCaps.find("mpegversion=(int)2") != std::string::npos )
			mOptions.codec = videoOptions::CODEC_MPEG2;
	}

	if( mOptions.codec == videoOptions::CODEC_UNKNOWN )
	{
		LogError(LOG_GSTREAMER "gstDecoder -- unsupported codec, supported codecs are:\n");
		LogError(LOG_GSTREAMER "                 * h264\n");
		LogError(LOG_GSTREAMER "                 * h265\n");
		LogError(LOG_GSTREAMER "                 * vp8\n");
		LogError(LOG_GSTREAMER "                 * vp9\n");
		LogError(LOG_GSTREAMER "                 * mpeg2\n");
		LogError(LOG_GSTREAMER "                 * mpeg4\n");
		LogError(LOG_GSTREAMER "                 * mjpeg\n");

		return false;
	}

	// TODO free other resources
	//g_free(discoverer);
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
		ss << "filesrc location=" << mOptions.resource.location << " ! ";

		if( uri.extension == "mkv" || uri.extension == "webm" )
			ss << "matroskademux ! ";
		else if( uri.extension == "mp4" || uri.extension == "qt" || uri.extension == "mov" )
			ss << "qtdemux ! ";
		else if( uri.extension == "flv" )
			ss << "flvdemux ! ";
		else if( uri.extension == "avi" )
			ss << "avidemux ! ";
		else if( uri.extension != "h264" && uri.extension != "h265" )
		{
			LogError(LOG_GSTREAMER "gstDecoder -- unsupported video file extension (%s)\n", uri.extension.c_str());
			LogError(LOG_GSTREAMER "              supported video extensions are:\n");
			LogError(LOG_GSTREAMER "                 * mkv, webm\n");
			LogError(LOG_GSTREAMER "                 * mp4, qt, mov\n");
			LogError(LOG_GSTREAMER "                 * flv\n");
			LogError(LOG_GSTREAMER "                 * avi\n");
			LogError(LOG_GSTREAMER "                 * h264, h265\n");

			return false;
		}

		ss << "queue ! ";
		
		if( mOptions.codec == videoOptions::CODEC_H264 )
			ss << "h264parse ! ";
		else if( mOptions.codec == videoOptions::CODEC_H265 )
			ss << "h265parse ! ";
		else if( mOptions.codec == videoOptions::CODEC_MPEG2 )
			ss << "mpegvideoparse ! ";
		else if( mOptions.codec == videoOptions::CODEC_MPEG4 )
			ss << "mpeg4videoparse ! ";

		mOptions.deviceType = videoOptions::DEVICE_FILE;
	}
	else if( uri.protocol == "rtp" )
	{
		if( uri.port <= 0 )
		{
			LogError(LOG_GSTREAMER "gstDecoder -- invalid RTP port (%i)\n", uri.port);
			return false;
		}

		ss << "udpsrc port=" << uri.port;
		ss << " multicast-group=" << uri.location << " auto-multicast=true";

		ss << " caps=\"" << "application/x-rtp,media=(string)video,clock-rate=(int)90000,encoding-name=(string)";
		
		if( mOptions.codec == videoOptions::CODEC_H264 )
			ss << "H264\" ! rtph264depay ! h264parse ! ";
		else if( mOptions.codec == videoOptions::CODEC_H265 )
			ss << "H265\" ! rtph265depay ! h265parse ! ";
		else if( mOptions.codec == videoOptions::CODEC_VP8 )
			ss << "VP8\" ! rtpvp8depay ! ";
		else if( mOptions.codec == videoOptions::CODEC_VP9 )
			ss << "VP9\" ! rtpvp9depay ! ";
		else if( mOptions.codec == videoOptions::CODEC_MPEG2 )
			ss << "MP2T\" ! rtpmp2tdepay ! ";		// MP2T-ES
		else if( mOptions.codec == videoOptions::CODEC_MPEG4 )
			ss << "MP4V-ES\" ! rtpmp4vdepay ! ";	// MPEG4-GENERIC\" ! rtpmp4gdepay
		else if( mOptions.codec == videoOptions::CODEC_MJPEG )
			ss << "JPEG\" ! rtpjpegdepay ! ";

		mOptions.deviceType = videoOptions::DEVICE_IP;
	}
	else if( uri.protocol == "rtsp" )
	{
		ss << "rtspsrc location=" << uri.string;
		ss << " latency=" << mOptions.rtspLatency;
		ss << " ! queue ! ";
		
		if( mOptions.codec == videoOptions::CODEC_H264 )
			ss << "rtph264depay ! h264parse ! ";
		else if( mOptions.codec == videoOptions::CODEC_H265 )
			ss << "rtph265depay ! h265parse ! ";
		else if( mOptions.codec == videoOptions::CODEC_VP8 )
			ss << "rtpvp8depay ! ";
		else if( mOptions.codec == videoOptions::CODEC_VP9 )
			ss << "rtpvp9depay ! ";
		else if( mOptions.codec == videoOptions::CODEC_MPEG2 )
			ss << "rtpmp2tdepay ! ";		// MP2T-ES
		else if( mOptions.codec == videoOptions::CODEC_MPEG4 )
			ss << "rtpmp4vdepay ! ";	// rtpmp4gdepay
		else if( mOptions.codec == videoOptions::CODEC_MJPEG )
			ss << "rtpjpegdepay ! ";

		mOptions.deviceType = videoOptions::DEVICE_IP;
	}
	else
	{
		LogError(LOG_GSTREAMER "gstDecoder -- unsupported protocol (%s)\n", uri.protocol.c_str());
		LogError(LOG_GSTREAMER "              supported protocols are:\n");
		LogError(LOG_GSTREAMER "                 * file://\n");
		LogError(LOG_GSTREAMER "                 * rtp://\n");
		LogError(LOG_GSTREAMER "                 * rtsp://\n");

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
	else if( mOptions.codec == videoOptions::CODEC_MPEG2 )
		ss << "omxmpeg2videodec ! ";
	else if( mOptions.codec == videoOptions::CODEC_MPEG4 )
		ss << "omxmpeg4videodec ! ";
	else if( mOptions.codec == videoOptions::CODEC_MJPEG )
		ss << "nvjpegdec ! ";
#else
	if( mOptions.codec == videoOptions::CODEC_H264 )
		ss << "nv_omx_h264dec ! ";
	else if( mOptions.codec == videoOptions::CODEC_H265 )
		ss << "nv_omx_h265dec ! ";
	else if( mOptions.codec == videoOptions::CODEC_VP8 )
		ss << "nv_omx_vp8dec ! ";
	else if( mOptions.codec == videoOptions::CODEC_VP9 )
		ss << "nv_omx_vp9dec ! ";
	else if( mOptions.codec == videoOptions::CODEC_MPEG2 )
		ss << "nx_omx_mpeg2videodec ! ";
	else if( mOptions.codec == videoOptions::CODEC_MPEG4 )
		ss << "nx_omx_mpeg4videodec ! ";
#endif
	else
	{
		LogError(LOG_GSTREAMER "gstDecoder -- unsupported codec requested (%s)\n", videoOptions::CodecToStr(mOptions.codec));
		LogError(LOG_GSTREAMER "              supported decoder codecs are:\n");
		LogError(LOG_GSTREAMER "                 * h264\n");
		LogError(LOG_GSTREAMER "                 * h265\n");
		LogError(LOG_GSTREAMER "                 * vp8\n");
		LogError(LOG_GSTREAMER "                 * vp9\n");
		LogError(LOG_GSTREAMER "                 * mpeg2\n");
		LogError(LOG_GSTREAMER "                 * mpeg4\n");
		LogError(LOG_GSTREAMER "                 * mjpeg\n");

		return false;
	}

	// resize if requested
	if( mCustomSize || mOptions.flipMethod != videoOptions::FLIP_NONE )
	{
		ss << "nvvidconv";

		if( mOptions.flipMethod != videoOptions::FLIP_NONE )
			ss << " flip-method=" << (int)mOptions.flipMethod;

		ss << " ! video/x-raw";

	#ifdef ENABLE_NVMM
		ss << "(" << GST_CAPS_FEATURE_MEMORY_NVMM << ")";
	#endif
	
		if( mOptions.width != 0 && mOptions.height != 0 )
			ss << ", width=(int)" << mOptions.width << ", height=(int)" << mOptions.height << ", format=(string)NV12";

		ss <<" ! ";
	}
	else
	{
		ss << "video/x-raw";
		
	#ifdef ENABLE_NVMM
		ss << "(" << GST_CAPS_FEATURE_MEMORY_NVMM << ")";
	#endif
	
		ss << " ! ";
	}

	// rate-limit if requested
	if( mCustomRate )
		ss << "videorate drop-only=true max-rate=" << (int)mOptions.frameRate << " ! ";

	// add the app sink
	ss << "appsink name=mysink"; // wait-on-eos=false;

	mLaunchStr = ss.str();

	LogInfo(LOG_GSTREAMER "gstDecoder -- pipeline string:\n");
	LogInfo(LOG_GSTREAMER "%s\n", mLaunchStr.c_str());

	return true;
}



// onEOS
void gstDecoder::onEOS( _GstAppSink* sink, void* user_data )
{
	LogWarning(LOG_GSTREAMER "gstDecoder -- end of stream (EOS)\n");

	if( !user_data )
		return;

	gstDecoder* dec = (gstDecoder*)user_data;

	dec->mEOS = true;	
	dec->mStreaming = dec->isLooping();
}


// onPreroll
GstFlowReturn gstDecoder::onPreroll( _GstAppSink* sink, void* user_data )
{
	LogVerbose(LOG_GSTREAMER "gstDecoder -- onPreroll()\n");

	if( !user_data )
		return GST_FLOW_OK;
		
	gstDecoder* dec = (gstDecoder*)user_data;
	
#if GST_CHECK_VERSION(1,0,0)
	// onPreroll gets called sometimes, just pull and free the buffer
	// otherwise the pipeline may hang during shutdown
	GstSample* gstSample = gst_app_sink_pull_preroll(dec->mAppSink);
	
	if( !gstSample )
	{
		LogError(LOG_GSTREAMER "gstDecoder -- app_sink_pull_sample() returned NULL...\n");
		return GST_FLOW_OK;
	}

	gst_sample_unref(gstSample);
#endif

	dec->checkMsgBus();
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
		LogError(LOG_GSTREAMER "gstDecoder -- app_sink_pull_sample() returned NULL...\n");
		return;
	}
	
	// retrieve sample caps
	GstCaps* gstCaps = gst_sample_get_caps(gstSample);
	
	if( !gstCaps )
	{
		LogError(LOG_GSTREAMER "gstDecoder -- gst_sample had NULL caps...\n");
		release_return;
	}
	
	// retrieve the buffer from the sample
	GstBuffer* gstBuffer = gst_sample_get_buffer(gstSample);
	
	if( !gstBuffer )
	{
		LogError(LOG_GSTREAMER "gstDecoder -- app_sink_pull_sample() returned NULL...\n");
		release_return;
	}
#else
	// block waiting for the buffer
	GstBuffer* gstBuffer = gst_app_sink_pull_buffer(mAppSink);
	
	if( !gstBuffer )
	{
		LogError(LOG_GSTREAMER "gstDecoder -- app_sink_pull_buffer() returned NULL...\n");
		return;
	}
	
	// retrieve caps
	GstCaps* gstCaps = gst_buffer_get_caps(gstBuffer);
	
	if( !gstCaps )
	{
		LogError(LOG_GSTREAMER "gstDecoder -- gst_buffer had NULL caps...\n");
		release_return;
	}
#endif
	
	if( !mBufferManager->Enqueue(gstBuffer, gstCaps) )
		LogError(LOG_GSTREAMER "gstDecoder -- failed to handle incoming buffer\n");
	
	release_return;
}


// Capture
bool gstDecoder::Capture( void** output, imageFormat format, uint64_t timeout )
{
	// verify the output pointer exists
	if( !output )
		return false;

	// confirm the stream is open
	if( !mStreaming || mEOS )
	{
		if( !Open() )
			return false;
	}

	// wait until a new frame is recieved
	if( !mBufferManager->Dequeue(output, format, timeout) )
	{
		LogError(LOG_GSTREAMER "gstDecoder -- failed to retrieve next image buffer\n");
		return false;
	}
	
	return true;
}

#if 0
static void queryPipelineState( GstElement* pipeline )
{
	GstState state = GST_STATE_VOID_PENDING;
	GstState pending = GST_STATE_VOID_PENDING;

	GstStateChangeReturn result = gst_element_get_state (pipeline,
		                  &state, &pending,  GST_CLOCK_TIME_NONE);

	if( result == GST_STATE_CHANGE_FAILURE )
		printf("GST_STATE_CHANGE_FAILURE\n");

	printf("state - %s\n", gst_element_state_get_name(state));
	printf("pending - %s\n", gst_element_state_get_name(pending));
}
#endif

// Open
bool gstDecoder::Open()
{
	if( mEOS )
	{
		if( isLooping() )
		{
			// seek stream back to the beginning
			GstEvent *seek_event = NULL;

			const bool seek = gst_element_seek(mPipeline, 1.0, GST_FORMAT_TIME,
						                    (GstSeekFlags)(GST_SEEK_FLAG_FLUSH | GST_SEEK_FLAG_KEY_UNIT),
						                    GST_SEEK_TYPE_SET, 0LL,
						                    GST_SEEK_TYPE_NONE, GST_CLOCK_TIME_NONE );

			if( !seek )
			{
				LogError(LOG_GSTREAMER "gstDecoder -- failed to seek stream to beginning (loop %zu of %i)\n", mLoopCount+1, mOptions.loop);
				return false;
			}
	
			LogWarning(LOG_GSTREAMER "gstDecoder -- seeking stream to beginning (loop %zu of %i)\n", mLoopCount+1, mOptions.loop);

			mLoopCount++;
			mEOS = false;
		}
		else
		{
			LogWarning(LOG_GSTREAMER "gstDecoder -- end of stream (EOS) has been reached, stream has been closed\n");
			return false;
		}	
	}

	if( mStreaming )
		return true;

	// transition pipline to STATE_PLAYING
	LogInfo(LOG_GSTREAMER "opening gstDecoder for streaming, transitioning pipeline to GST_STATE_PLAYING\n");
	
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
		LogError(LOG_GSTREAMER "gstDecoder -- failed to set pipeline state to PLAYING (error %u)\n", result);
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
	if( !mStreaming && !mEOS )  // if EOS was set, the pipeline is actually open
		return;

	// stop pipeline
	LogInfo(LOG_GSTREAMER "gstDecoder -- stopping pipeline, transitioning to GST_STATE_NULL\n");

	const GstStateChangeReturn result = gst_element_set_state(mPipeline, GST_STATE_NULL);

	if( result != GST_STATE_CHANGE_SUCCESS )
		LogError(LOG_GSTREAMER "gstDecoder -- failed to stop pipeline (error %u)\n", result);

	usleep(250*1000);
	checkMsgBus();
	mStreaming = false;
	LogInfo(LOG_GSTREAMER "gstDecoder -- pipeline stopped\n");
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



