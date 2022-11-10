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
#include "WebRTCServer.h"

#include "logging.h"
#include "filesystem.h"

#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <gst/pbutils/pbutils.h>

#define GST_USE_UNSTABLE_API
#include <gst/webrtc/webrtc.h>
#include <json-glib/json-glib.h>

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
	
	mWebRTCServer = NULL;
	mWebRTCPeer = NULL;
	mWebRTCBin = NULL;
	mWebRTCConnected = false;
}


// destructor
gstDecoder::~gstDecoder()
{
	Close();

	if( mWebRTCBin != NULL )
	{
		gst_object_unref(mWebRTCBin);
		mWebRTCBin = NULL;
	}
	
	if( mWebRTCServer != NULL )
	{
		mWebRTCServer->Release();
		mWebRTCServer = NULL;
	}
	
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
		if( mOptions.resource.protocol == "rtp" || mOptions.resource.protocol == "webrtc" )
		{
			LogWarning(LOG_GSTREAMER "gstDecoder -- resource discovery not supported for RTP/WebRTC streams\n");	

			if( mOptions.codec == videoOptions::CODEC_UNKNOWN )
			{
				LogWarning(LOG_GSTREAMER "gstDecoder -- defaulting to H264 codec (you can change this with the --input-codec option)\n");
				mOptions.codec = videoOptions::CODEC_H264;
			}
		}
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
	
	// create server for WebRTC streams
	if( mOptions.resource.protocol == "webrtc" )
	{
		// connect webrtcbin callbacks
		mWebRTCBin = gst_bin_get_by_name(GST_BIN(mPipeline), "webrtcbin");
		g_assert_nonnull(mWebRTCBin);
		
		g_signal_connect(mWebRTCBin, "on-negotiation-needed", G_CALLBACK(onNegotiationNeeded), this);
		g_signal_connect(mWebRTCBin, "on-ice-candidate", G_CALLBACK(onIceCandidate), this);
		
		// create stream caps to advertise
		std::ostringstream ss;
		
		ss << "application/x-rtp,media=video,encoding-name=";
		ss << videoOptions::CodecToStr(mOptions.codec);
		ss << ",payload=96,clock-rate=90000";
		
		if( mOptions.codec == videoOptions::CODEC_H264 )
		{
			// https://www.rfc-editor.org/rfc/rfc6184#section-8.1
			// https://stackoverflow.com/questions/22960928/identify-h264-profile-and-level-from-profile-level-id-in-sdp
			// https://en.wikipedia.org/wiki/Advanced_Video_Coding#Levels
			//
			// profile_idc:
			//   0x42 = 66  => baseline
			//   0x4D = 77  => main
			//   0x64 = 100 => high
			//
			// profile_iop:
			//   0x80 = 100000 => constraint_set0_flag=1, constraint_set1_flag=0 => ???
			//   0xc0 = 110000 => constraint_set0_flag=1, constraint_set1_flag=1 => constrained
			//
			// levels_idc:  
			//   0x16 = 22 = 4mbps  (720×480@15.0)
			//   0x1E = 30 = 10mbps (720×480@15.0)
			//   0x1F = 31 = 14mbps (1280×720@30.0)
			ss << ",profile-level-id=(string)42c016";  // constrained baseline profile 2.2
			ss << ",packetization-mode=(string)1";
		}
		
		const std::string caps_str = ss.str();
		LogVerbose(LOG_WEBRTC "gstDecoder -- configuring recieve-only caps string: \n%s\n", caps_str.c_str());
		
		// add transciever in receive-only mode  (https://stackoverflow.com/questions/57430215/how-to-use-webrtcbin-create-offer-only-receive-video)
		GstWebRTCRTPTransceiver* transceiver = NULL;
		GstCaps* transceiver_caps = gst_caps_from_string(caps_str.c_str());
		g_signal_emit_by_name(mWebRTCBin, "add-transceiver", GST_WEBRTC_RTP_TRANSCEIVER_DIRECTION_RECVONLY, transceiver_caps, &transceiver);
		gst_caps_unref(transceiver_caps);
		gst_object_unref(transceiver);
		
		// create WebRTC server
		mWebRTCServer = WebRTCServer::Create(mOptions.resource.port, mOptions.stunServer.c_str(),
									  mOptions.sslCert.c_str(), mOptions.sslKey.c_str());
		
		if( !mWebRTCServer )
			return false;
		
		mWebRTCServer->AddRoute(mOptions.resource.path.c_str(), onWebsocketMessage, this, WEBRTC_VIDEO|WEBRTC_RECEIVE|WEBRTC_PUBLIC);
	}	
	
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
	// RTP streams and WebRTC connections can't be discovered
	if( mOptions.resource.protocol == "rtp" || mOptions.resource.protocol == "webrtc" )
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
	else if( uri.protocol == "rtsp" || uri.protocol == "webrtc" )
	{
		if( uri.protocol == "rtsp" )
		{
			ss << "rtspsrc location=" << uri.string;
			ss << " latency=" << mOptions.rtspLatency;
			ss << " ! queue ! ";
		}
		else
		{
			ss << "webrtcbin name=webrtcbin ";
			ss << "stun-server=stun://" << WEBRTC_DEFAULT_STUN_SERVER;
			ss << " ! queue ! ";
		}
		
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

	// select the decoder
	if( mOptions.codec == videoOptions::CODEC_H264 )
		ss << GST_DECODER_H264 << " ! ";
	else if( mOptions.codec == videoOptions::CODEC_H265 )
		ss << GST_DECODER_H265 << " ! ";
	else if( mOptions.codec == videoOptions::CODEC_VP8 )
		ss << GST_DECODER_VP8 << " ! ";
	else if( mOptions.codec == videoOptions::CODEC_VP9 )
		ss << GST_DECODER_VP9 << " ! ";
	else if( mOptions.codec == videoOptions::CODEC_MPEG2 )
		ss << GST_DECODER_MPEG2 << " ! ";
	else if( mOptions.codec == videoOptions::CODEC_MPEG4 )
		ss << GST_DECODER_MPEG4 << " ! ";
	else if( mOptions.codec == videoOptions::CODEC_MJPEG )
		ss << GST_DECODER_MJPEG << " ! ";

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
	#if defined(__aarch64__)
		ss << "nvvidconv";

		if( mOptions.flipMethod != videoOptions::FLIP_NONE )
			ss << " flip-method=" << (int)mOptions.flipMethod;
		
		ss << " ! video/x-raw";
		
	#elif defined(__x86_64__) || defined(__amd64__)
		if( mOptions.flipMethod != videoOptions::FLIP_NONE )
			ss << "videoflip method=" << videoOptions::FlipMethodToStr(mOptions.flipMethod) << " ! ";
		
		if( mOptions.width != 0 && mOptions.height != 0 )
			ss << "videoscale ! ";
		
		ss << "video/x-raw";
	#endif

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
		
	#if defined(ENABLE_NVMM) || defined(GST_CODECS_V4L2)
		// add NVMM caps when requested, or if using V4L2 codecs
		ss << "(" << GST_CAPS_FEATURE_MEMORY_NVMM << ")";
	#ifndef ENABLE_NVMM
		// V4L2 codecs only output NVMM memory
		// so if NVMM is disabled, put it through nvvidconv first
		ss << " ! nvvidconv ! video/x-raw";
	#endif
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
GstFlowReturn gstDecoder::onBuffer( _GstAppSink* sink, void* user_data )
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
	// update the webrtc server
	if( mWebRTCServer != NULL )
		mWebRTCServer->ProcessRequests();
	
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
	
	mLastTimestamp = mBufferManager->GetLastTimestamp();
	mRawFormat = mBufferManager->GetRawFormat();
	
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

	if( mStreaming || (mWebRTCServer != NULL && !mWebRTCConnected) )  // with WebRTC, don't start the pipeline until peer connected
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



// onWebsocketMessage
void gstDecoder::onWebsocketMessage( WebRTCPeer* peer, const char* message, size_t message_size, void* user_data )
{
	if( !user_data )
		return;
	
	gstDecoder* decoder = (gstDecoder*)user_data;

	if( peer->flags & WEBRTC_PEER_CONNECTING )
	{
		LogVerbose(LOG_WEBRTC "new WebRTC peer connecting (%s, peer_id=%u)\n", peer->ip_address.c_str(), peer->ID);
		
		if( decoder->mWebRTCConnected )
		{
			LogError(LOG_WEBRTC "another WebRTC peer is already connected to gstDecoder, ignoring incoming connection\n");
			return;
		}

		peer->user_data = decoder;
		
		decoder->mWebRTCPeer = peer;
		decoder->mWebRTCConnected = true;
		
		decoder->Open();
  
		return;
	}
	else if( peer->flags & WEBRTC_PEER_CLOSED )
	{
		LogVerbose(LOG_WEBRTC "WebRTC peer disconnected (%s, peer_id=%u)\n", peer->ip_address.c_str(), peer->ID);
		
		decoder->mWebRTCPeer = NULL;
		decoder->mWebRTCConnected = false;
		
		return;
	}
	
	#define cleanup() { \
		if( json_parser != NULL ) \
			g_object_unref(G_OBJECT(json_parser)); \
		return; } \

	#define unknown_message() { \
		LogWarning(LOG_WEBRTC "gstDecoder -- unknown message, ignoring...\n%s\n", message); \
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
		g_signal_emit_by_name(decoder->mWebRTCBin, "set-remote-description", answer, promise);
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
		g_signal_emit_by_name(decoder->mWebRTCBin, "add-ice-candidate", mline_index, candidate_string);
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
void gstDecoder::onNegotiationNeeded( GstElement* webrtcbin, void* user_data )
{
	LogDebug(LOG_WEBRTC "gstDecoder -- onNegotiationNeeded()\n");
	
	if( !user_data )
		return;
	
	gstDecoder* decoder = (gstDecoder*)user_data;

	// setup offer created callback
	GstPromise* promise = gst_promise_new_with_change_func(onCreateOffer, decoder, NULL);
	g_signal_emit_by_name(G_OBJECT(decoder->mWebRTCBin), "create-offer", NULL, promise);
}


// onOfferCreated
void gstDecoder::onCreateOffer( GstPromise* promise, void* user_data )
{
	LogDebug(LOG_WEBRTC "gstDecoder -- onCreateOffer()\n");
	
	if( !user_data )
		return;
	
	gstDecoder* decoder = (gstDecoder*)user_data;
	WebRTCPeer* peer = decoder->mWebRTCPeer;
	
	// send the SDP offer
	const GstStructure* reply = gst_promise_get_reply(promise);
	
	GstWebRTCSessionDescription* offer = NULL;
	gst_structure_get(reply, "offer", GST_TYPE_WEBRTC_SESSION_DESCRIPTION, &offer, NULL);
	gst_promise_unref(promise);

	GstPromise* local_desc_promise = gst_promise_new();
	g_signal_emit_by_name(decoder->mWebRTCBin, "set-local-description", offer, local_desc_promise);
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
void gstDecoder::onIceCandidate( GstElement* webrtcbin, uint32_t mline_index, char* candidate, void* user_data )
{
	LogDebug(LOG_WEBRTC "gstDecoder -- onIceCandidate()\n");
	
	if( !user_data )
		return;
	
	gstDecoder* decoder = (gstDecoder*)user_data;
	WebRTCPeer* peer = decoder->mWebRTCPeer;

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

