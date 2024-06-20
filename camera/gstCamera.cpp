/*
 * Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
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

#include "gstCamera.h"

#include "cudaColorspace.h"
#include "filesystem.h"
#include "logging.h"
#include "NvInfer.h"

#include <gst/app/gstappsink.h>

#include <sstream> 
#include <unistd.h>
#include <string.h>
#include <math.h>


// constructor
gstCamera::gstCamera( const videoOptions& options ) : videoSource(options)
{	
	mAppSink   = NULL;
	mBus       = NULL;
	mPipeline  = NULL;	
	mFormatYUV = IMAGE_UNKNOWN;
	
	mBufferManager = new gstBufferManager(&mOptions);
}


// destructor	
gstCamera::~gstCamera()
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
gstCamera* gstCamera::Create( uint32_t width, uint32_t height, const char* camera )
{
	videoOptions opt;

	if( !camera )
		camera = "csi://0";

	opt.resource   = camera;
	opt.width      = width;
	opt.height     = height;
	opt.ioType     = videoOptions::INPUT;
	opt.deviceType = videoOptions::DeviceTypeFromStr(opt.resource.protocol.c_str());
	
	return Create(opt);
}


// Create
gstCamera* gstCamera::Create( const videoOptions& options )
{
	if( !gstreamerInit() )
	{
		LogError(LOG_GSTREAMER "failed to initialize gstreamer API\n");
		return NULL;
	}

	// create camera instance
	gstCamera* cam = new gstCamera(options);
	
	if( !cam )
		return NULL;
	
	// initialize camera (with fallback)
	if( !cam->init() )
	{
		LogError(LOG_GSTREAMER "gstCamera -- failed to create device %s\n", cam->GetResource().c_str());
		return NULL;
	}
	
	LogInfo(LOG_GSTREAMER "gstCamera successfully created device %s\n", cam->GetResource().c_str()); 
	return cam;
}


// Create
gstCamera* gstCamera::Create( const char* camera )
{
	return Create( DefaultWidth, DefaultHeight, camera );
}


// buildLaunchStr
bool gstCamera::buildLaunchStr()
{
	std::ostringstream ss;

	#if defined(ENABLE_NVMM)
		const bool enable_nvmm = true;
	#else
		const bool enable_nvmm = false;
	#endif
	
	if( mOptions.save.path.length() > 0 && (mOptions.codec == videoOptions::CODEC_RAW || mOptions.codec == videoOptions::CODEC_UNKNOWN) )
	{
		LogError(LOG_GSTREAMER "can't use the --input-save option on a raw/uncompressed input stream\n");
		return false;
	}

	if( mOptions.resource.protocol == "csi" )
	{
	#if defined(__x86_64__) || defined(__amd64__)
		LogError(LOG_GSTREAMER "MIPI CSI camera isn't available on x86 - please use /dev/video (V4L2) instead");
		return false;
	#endif
	
	#if NV_TENSORRT_MAJOR > 4
		// on newer JetPack's, it's common for CSI camera to need flipped
		// so here we reverse FLIP_NONE with FLIP_ROTATE_180
		if( mOptions.flipMethod == videoOptions::FLIP_NONE )
			mOptions.flipMethod = videoOptions::FLIP_ROTATE_180;
		else if( mOptions.flipMethod == videoOptions::FLIP_ROTATE_180 )
			mOptions.flipMethod = videoOptions::FLIP_NONE;
	
		ss << "nvarguscamerasrc sensor-id=" << mOptions.resource.port << " ! video/x-raw(memory:NVMM), width=(int)" << GetWidth() << ", height=(int)" << GetHeight() << ", framerate=" << (int)mOptions.frameRate << "/1, format=(string)NV12 ! nvvidconv flip-method=" << mOptions.flipMethod << " ! ";
	#else
		// older JetPack versions use nvcamerasrc element instead of nvarguscamerasrc
		ss << "nvcamerasrc fpsRange=\"" << (int)mOptions.frameRate << " " << (int)mOptions.frameRate << "\" ! video/x-raw(memory:NVMM), width=(int)" << GetWidth() << ", height=(int)" << GetHeight() << ", format=(string)NV12 ! nvvidconv flip-method=" << mOptions.flipMethod << " ! "; //'video/x-raw(memory:NVMM), width=(int)1920, height=(int)1080, format=(string)I420, framerate=(fraction)30/1' ! ";
	#endif
	
		ss << (enable_nvmm ? "video/x-raw(memory:NVMM) ! " : "video/x-raw ! "); 
		ss << "appsink name=mysink";
	}
	else
	{
		ss << "v4l2src device=" << mOptions.resource.location << " do-timestamp=true ! ";
		
		if( mOptions.codec != videoOptions::CODEC_UNKNOWN )
		{
			ss << gst_codec_to_string(mOptions.codec) << ", ";
			
			if( mOptions.codec == videoOptions::CODEC_RAW )
				ss << "format=(string)" << gst_format_to_string(mFormatYUV) << ", ";
			
			ss << "width=(int)" << GetWidth() << ", height=(int)" << GetHeight() << ", framerate=" << (int)mOptions.frameRate << "/1 ! "; 
		}
		
		//ss << "queue max-size-buffers=16 ! ";

		if( mOptions.save.path.length() > 0 )
		{
			ss << "tee name=savetee savetee. ! queue ! ";
			
			if( !gst_build_filesink(mOptions.save, mOptions.codec, ss) )
				return false;

			ss << "savetee. ! queue ! ";
		}
	
		// select the decoder
		const char* decoder = gst_select_decoder(mOptions.codec, mOptions.codecType);
		
		if( !decoder && mOptions.codec != videoOptions::CODEC_RAW )
		{
			LogError(LOG_GSTREAMER "gstCamera -- unsupported codec requested (%s)\n", videoOptions::CodecToStr(mOptions.codec));
			LogError(LOG_GSTREAMER "             supported decoder codecs are:\n");
			LogError(LOG_GSTREAMER "                * h264\n");
			LogError(LOG_GSTREAMER "                * h265\n");
			LogError(LOG_GSTREAMER "                * vp8\n");
			LogError(LOG_GSTREAMER "                * vp9\n");
			LogError(LOG_GSTREAMER "                * mpeg2\n");
			LogError(LOG_GSTREAMER "                * mpeg4\n");
			LogError(LOG_GSTREAMER "                * mjpeg\n");
			
			return false;
		}

		if( mOptions.codec != videoOptions::CODEC_RAW )
		{
			if( mOptions.codecType != videoOptions::CODEC_V4L2 )
			{
				if( mOptions.codec == videoOptions::CODEC_H264 )
					ss << "h264parse ! ";  
				else if( mOptions.codec == videoOptions::CODEC_H265 )
					ss << "h265parse ! ";
				else if( mOptions.codec == videoOptions::CODEC_MPEG2 )
					ss << "mpegvideoparse ! ";
				else if( mOptions.codec == videoOptions::CODEC_MPEG4 )
					ss << "mpeg4videoparse ! ";
			}

			ss << decoder << " name=decoder ";  //ss << "nvjpegdec ! video/x-raw ! "; //ss << "jpegparse ! nvv4l2decoder mjpeg=1 ! video/x-raw(memory:NVMM) ! nvvidconv ! video/x-raw ! "; //
	
			if( mOptions.codecType == videoOptions::CODEC_V4L2 && mOptions.codec != videoOptions::CODEC_MJPEG )
				ss << "enable-max-performance=1 ";
			
			ss << "! ";
	
			if( (enable_nvmm && mOptions.codecType != videoOptions::CODEC_CPU) || mOptions.codecType == videoOptions::CODEC_V4L2 )
				ss << "video/x-raw(memory:NVMM) ! ";  // V4L2 codecs can only output NVMM
			else
				ss << "video/x-raw ! ";
		}

	#if defined(__aarch64__)
		// video flipping/rotating for V4L2 devices (use nvvidconv if a hw codec is used for decode)
		// V4L2 decoders can only output NVMM memory, if we aren't using NVMM have nvvidconv convert it 
		if( mOptions.flipMethod != videoOptions::FLIP_NONE || (mOptions.codecType == videoOptions::CODEC_V4L2 && !enable_nvmm) )
		{
			if( (enable_nvmm && mOptions.codecType != videoOptions::CODEC_CPU) || mOptions.codecType == videoOptions::CODEC_V4L2 )
				ss << "nvvidconv flip-method=" << mOptions.flipMethod << " ! " << (enable_nvmm ? "video/x-raw(memory:NVMM) ! " : "video/x-raw ! ");
			else
				ss << "videoflip method=" << videoOptions::FlipMethodToStr(mOptions.flipMethod) << " ! ";  // the videoflip enum varies slightly, but the strings are the same
		}
	#elif defined(__x86_64__) || defined(__amd64__)
		if( mOptions.flipMethod != videoOptions::FLIP_NONE )
			ss << "videoflip method=" << videoOptions::FlipMethodToStr(mOptions.flipMethod) << " ! ";
	#endif
		ss << "appsink name=mysink sync=false";
	}
	
	mLaunchStr = ss.str();

	LogInfo(LOG_GSTREAMER "gstCamera pipeline string:\n");
	LogInfo(LOG_GSTREAMER "%s\n", mLaunchStr.c_str());

	return true;
}


// printCaps
bool gstCamera::printCaps( GstCaps* device_caps ) const
{
	const uint32_t numCaps = gst_caps_get_size(device_caps);
	
	LogVerbose(LOG_GSTREAMER "gstCamera -- found %u caps for v4l2 device %s\n", numCaps, mOptions.resource.location.c_str());

	if( numCaps == 0 )
		return false;
	
	for( uint32_t n=0; n < numCaps; n++ )
	{
		GstStructure* caps = gst_caps_get_structure(device_caps, n);
		
		if( !caps )
			continue;
		
		LogVerbose(LOG_GSTREAMER "[%u] %s\n", n, gst_structure_to_string(caps));
	}
	
	return true;
}


// pick the closest framerate
float gstCamera::findFramerate( const std::vector<float>& frameRates, float frameRate ) const
{
	const uint32_t numRates = frameRates.size();
	
	if( numRates == 0 )
		return frameRate;
	
	float bestRate = 0.0f;
	float bestDiff = 10000.0f;
	
	for( uint32_t n=0; n < numRates; n++ )
	{
		const float diff = fabsf(frameRates[n] - frameRate);
		
		if( diff < bestDiff )
		{
			bestRate = frameRates[n];
			bestDiff = diff;
		}
	}
	
	return bestRate;
}


// parseCaps
bool gstCamera::parseCaps( GstStructure* caps, videoOptions::Codec* _codec, imageFormat* _format, uint32_t* _width, uint32_t* _height, float* _frameRate ) const
{
	std::vector<float> frameRates;
	
	if( !parseCaps(caps, _codec, _format, _width, _height, frameRates) )
		return false;
	
	*_frameRate = findFramerate(frameRates, *_frameRate);
	return true;
}


// parseCaps
bool gstCamera::parseCaps( GstStructure* caps, videoOptions::Codec* _codec, imageFormat* _format, uint32_t* _width, uint32_t* _height, std::vector<float>& frameRates ) const
{
	// parse codec/format
	const videoOptions::Codec codec = gst_parse_codec(caps);
	const imageFormat format = gst_parse_format(caps);
	
	if( codec == videoOptions::CODEC_UNKNOWN )
		return false;
	
	if( codec == videoOptions::CODEC_RAW && format == IMAGE_UNKNOWN )
		return false;
	
	// if the user is requesting a codec, check that it matches
	if( mOptions.codec != videoOptions::CODEC_UNKNOWN && mOptions.codec != codec )
		return false;
	
	// get width/height
	int width  = 0;
	int height = 0;
	
	if( !gst_structure_get_int(caps, "width", &width) || !gst_structure_get_int(caps, "height", &height) )
		return false;

	// get highest framerate
	int frameRateNum = 0;
	int frameRateDenom = 0;
	
	if( gst_structure_get_fraction(caps, "framerate", &frameRateNum, &frameRateDenom) )
	{
		frameRates.push_back(float(frameRateNum) / float(frameRateDenom));
	}
	else
	{
		GValueArray* frameRateList = NULL;

		if( gst_structure_get_list(caps, "framerate", &frameRateList) && frameRateList->n_values > 0 )
		{
			for( uint32_t n=0; n < frameRateList->n_values; n++ )
			{
				GValue* value = frameRateList->values + n;

				if( GST_VALUE_HOLDS_FRACTION(value) )
				{
					frameRateNum = gst_value_get_fraction_numerator(value);
					frameRateDenom = gst_value_get_fraction_denominator(value);

					if( frameRateNum > 0 && frameRateDenom > 0 )
						frameRates.push_back(float(frameRateNum) / float(frameRateDenom));
				}
			}
		}
	}
	
	if( frameRates.size() == 0 )
		LogWarning(LOG_GSTREAMER "gstCamera -- missing framerate in caps, ignoring\n");

	*_codec     = codec;
	*_format    = format;
	*_width     = width;
	*_height    = height;

	return true;
}


// matchCaps
bool gstCamera::matchCaps( GstCaps* device_caps )
{
	const uint32_t numCaps = gst_caps_get_size(device_caps);
	GstStructure* bestCaps = NULL;

	int bestResolution = 1000000;
	float bestFrameRate = 0.0f;
	
	videoOptions::Codec bestCodec = videoOptions::CODEC_UNKNOWN;

	for( uint32_t n=0; n < numCaps; n++ )
	{
		GstStructure* caps = gst_caps_get_structure(device_caps, n);
		
		if( !caps )
			continue;
		
		videoOptions::Codec codec;
		imageFormat format;
		uint32_t width, height;
		float frameRate = mOptions.frameRate;

		if( !parseCaps(caps, &codec, &format, &width, &height, &frameRate) )
			continue;
	
		const int resolutionDiff = abs(int(mOptions.width) - int(width)) + abs(int(mOptions.height) - int(height));
	
		// pick this one if the resolution is closer, or if the resolution is the same but the framerate is better
		// (or if the framerate is the same and previous codec was MJPEG, pick the new one because MJPEG isn't preferred)
		if( resolutionDiff < bestResolution || (resolutionDiff == bestResolution && (frameRate > bestFrameRate || bestCodec == videoOptions::CODEC_MJPEG)) )
		{
			bestResolution = resolutionDiff;
			bestFrameRate = frameRate;
			bestCodec = codec;
			bestCaps = caps;
		}
	}
		
	if( !bestCaps )
	{
		printf(LOG_GSTREAMER "gstCamera -- couldn't find a compatible codec/format for v4l2 device %s\n", mOptions.resource.location.c_str());
		return false;
	}
	
	if( !parseCaps(bestCaps, &mOptions.codec, &mFormatYUV, &mOptions.width, &mOptions.height, &mOptions.frameRate) )
		return false;
	
	return true;
}


// discover
bool gstCamera::discover()
{
	// check desired frame sizes
	if( mOptions.width == 0 )
		mOptions.width = DefaultWidth;

	if( mOptions.height == 0 )
		mOptions.height = DefaultHeight;
	
	if( mOptions.frameRate <= 0 )
		mOptions.frameRate = 30;
	
	// MIPI CSI cameras aren't enumerated
	if( mOptions.resource.protocol != "v4l2" )
	{
		mOptions.codec = videoOptions::CODEC_RAW;
		return true;
	}
	
	// create v4l2 device service
	GstDeviceProvider* deviceProvider = gst_device_provider_factory_get_by_name("v4l2deviceprovider");
	
	if( !deviceProvider )
	{
		LogError(LOG_GSTREAMER "gstCamera -- failed to create v4l2 device provider during discovery\n");
		return false;
	}
	
	// get list of v4l2 devices
	GList* deviceList = gst_device_provider_get_devices(deviceProvider);

	if( !deviceList )
	{
		LogError(LOG_GSTREAMER "gstCamera -- didn't discover any v4l2 devices\n");
		return false;
	}

	// find the requested /dev/video* device
	GstDevice* device = NULL;
	
	for( GList* n=deviceList; n; n = n->next )
	{
		GstDevice* d = GST_DEVICE(n->data);
		
		const char* deviceName = gst_device_get_display_name(d);
		
		LogVerbose(LOG_GSTREAMER "gstCamera -- found v4l2 device: %s\n", deviceName);
	
	#if NV_TENSORRT_MAJOR > 8 || (NV_TENSORRT_MAJOR == 8 && NV_TENSORRT_MINOR >= 4)
		// on JetPack >= 5.0.1, the newer Logitech C920's send a H264 stream that nvv4l2decoder has trouble decoding, so change it to MJPEG
		if( strcmp(deviceName, "HD Pro Webcam C920") == 0 && mOptions.codecType == videoOptions::CODEC_V4L2 && mOptions.codec == videoOptions::CODEC_UNKNOWN )
			mOptions.codec = videoOptions::CODEC_MJPEG;
	#endif
	
		GstStructure* properties = gst_device_get_properties(d);
		
		if( properties != NULL )
		{
			LogVerbose(LOG_GSTREAMER "%s\n", gst_structure_to_string(properties));
			
			const char* devicePath = gst_structure_get_string(properties, "device.path");
			
			if( devicePath != NULL && strcasecmp(devicePath, mOptions.resource.location.c_str()) == 0 )
			{
				device = d;
				break;
			}
		}
	}
	
	if( !device )
	{
		LogError(LOG_GSTREAMER "gstCamera -- could not find v4l2 device %s\n", mOptions.resource.location.c_str());
		return false;
	}
	
	// get the caps of the device
	GstCaps* device_caps = gst_device_get_caps(device);
	
	if( !device_caps )
	{
		LogError(LOG_GSTREAMER "gstCamera -- failed to retrieve caps for v4l2 device %s\n", mOptions.resource.location.c_str());
		return false;
	}
	
	printCaps(device_caps);
	
	// pick the best caps
	if( !matchCaps(device_caps) )
		return false;
	
	LogVerbose(LOG_GSTREAMER "gstCamera -- selected device profile:  codec=%s format=%s width=%u height=%u framerate=%u\n", videoOptions::CodecToStr(mOptions.codec), imageFormatToStr(mFormatYUV), GetWidth(), GetHeight(), GetFrameRate());
	
	return true;
}


// init
bool gstCamera::init()
{
	GError* err = NULL;
	LogInfo(LOG_GSTREAMER "gstCamera -- attempting to create device %s\n", GetResource().c_str());

	// discover device stats
	if( !discover() )
	{
		if( mOptions.resource.protocol == "v4l2" && fileExists(mOptions.resource.location) )
		{
			LogWarning(LOG_GSTREAMER "gstCamera -- device discovery failed, but %s exists\n", mOptions.resource.location.c_str());
			LogWarning(LOG_GSTREAMER "             support for compressed formats is disabled\n");
		}
		else
		{
			LogError(LOG_GSTREAMER "gstCamera -- device discovery and auto-negotiation failed\n");
			return false;
		}
	}
	
	// build pipeline string
	if( !buildLaunchStr() )
	{
		LogError(LOG_GSTREAMER "gstCamera failed to build pipeline string\n");
		return false;
	}

	// launch pipeline
	mPipeline = gst_parse_launch(mLaunchStr.c_str(), &err);

	if( err != NULL )
	{
		LogError(LOG_GSTREAMER "gstCamera failed to create pipeline\n");
		LogError(LOG_GSTREAMER "   (%s)\n", err->message);
		g_error_free(err);
		return false;
	}

	GstPipeline* pipeline = GST_PIPELINE(mPipeline);

	if( !pipeline )
	{
		LogError(LOG_GSTREAMER "gstCamera failed to cast GstElement into GstPipeline\n");
		return false;
	}	

	// retrieve pipeline bus
	/*GstBus**/ mBus = gst_pipeline_get_bus(pipeline);

	if( !mBus )
	{
		LogError(LOG_GSTREAMER "gstCamera failed to retrieve GstBus from pipeline\n");
		return false;
	}

	// add watch for messages (disabled when we poll the bus ourselves, instead of gmainloop)
	//gst_bus_add_watch(mBus, (GstBusFunc)gst_message_print, NULL);

	// get the appsrc
	GstElement* appsinkElement = gst_bin_get_by_name(GST_BIN(pipeline), "mysink");
	GstAppSink* appsink = GST_APP_SINK(appsinkElement);

	if( !appsinkElement || !appsink)
	{
		LogError(LOG_GSTREAMER "gstCamera failed to retrieve AppSink element from pipeline\n");
		return false;
	}
	
	mAppSink = appsink;
	
	// setup callbacks
	GstAppSinkCallbacks cb;
	memset(&cb, 0, sizeof(GstAppSinkCallbacks));
	
	cb.eos         = onEOS;
	cb.new_preroll = onPreroll;
	cb.new_sample  = onBuffer;
	
	gst_app_sink_set_callbacks(mAppSink, &cb, (void*)this, NULL);
	
	// disable looping for cameras
	mOptions.loop = 0;	

	return true;
}


// onEOS
void gstCamera::onEOS(_GstAppSink* sink, void* user_data)
{
	LogWarning(LOG_GSTREAMER "gstCamera -- end of stream (EOS)\n");
}

// onPreroll
GstFlowReturn gstCamera::onPreroll(_GstAppSink* sink, void* user_data)
{
	LogVerbose(LOG_GSTREAMER "gstCamera -- onPreroll\n");
	return GST_FLOW_OK;
}

// onBuffer
GstFlowReturn gstCamera::onBuffer(_GstAppSink* sink, void* user_data)
{
	//printf(LOG_GSTREAMER "gstCamera onBuffer\n");
	
	if( !user_data )
		return GST_FLOW_OK;
		
	gstCamera* dec = (gstCamera*)user_data;
	
	dec->checkBuffer();
	dec->checkMsgBus();
	
	return GST_FLOW_OK;
}
	

#define release_return { gst_sample_unref(gstSample); return; }

// checkBuffer
void gstCamera::checkBuffer()
{
	if( !mAppSink )
		return;

	// block waiting for the buffer
	GstSample* gstSample = gst_app_sink_pull_sample(mAppSink);
	
	if( !gstSample )
	{
		LogError(LOG_GSTREAMER "gstCamera -- app_sink_pull_sample() returned NULL...\n");
		return;
	}
	
	// retrieve sample caps
	GstCaps* gstCaps = gst_sample_get_caps(gstSample);
	
	if( !gstCaps )
	{
		LogError(LOG_GSTREAMER "gstCamera -- gst_sample had NULL caps...\n");
		release_return;
	}
	
	// retrieve the buffer from the sample
	GstBuffer* gstBuffer = gst_sample_get_buffer(gstSample);
	
	if( !gstBuffer )
	{
		LogError(LOG_GSTREAMER "gstCamera -- app_sink_pull_sample() returned NULL...\n");
		release_return;
	}

	// enqueue the buffer for color conversion
	if( !mBufferManager->Enqueue(gstBuffer, gstCaps) )
	{
		LogError(LOG_GSTREAMER "gstCamera -- failed to handle incoming buffer\n");
		release_return;
	}
	
	mOptions.frameCount++;
	release_return;
}


#define RETURN_STATUS(code)  { if( status != NULL ) { *status=(code); } return ((code) == videoSource::OK ? true : false); }


// Capture
bool gstCamera::Capture( void** output, imageFormat format, uint64_t timeout, int* status, cudaStream_t stream )
{
	// verify the output pointer exists
	if( !output )
		RETURN_STATUS(ERROR);

	// confirm the camera is streaming
	if( !mStreaming )
	{
		if( !Open() )
			RETURN_STATUS(ERROR);
	}

	// wait until a new frame is recieved
	const int result = mBufferManager->Dequeue(output, format, timeout, stream);
	
	if( result < 0 )
	{
		LogError(LOG_GSTREAMER "gstCamera::Capture() -- an error occurred retrieving the next image buffer\n");
		RETURN_STATUS(ERROR);
	}
	else if( result == 0 )
	{
		LogWarning(LOG_GSTREAMER "gstCamera::Capture() -- a timeout occurred waiting for the next image buffer\n");
		RETURN_STATUS(TIMEOUT);
	}

	mLastTimestamp = mBufferManager->GetLastTimestamp();
	mRawFormat = mBufferManager->GetRawFormat();

	RETURN_STATUS(OK);
}

// CaptureRGBA
bool gstCamera::CaptureRGBA( float** output, unsigned long timeout, bool zeroCopy, cudaStream_t stream )
{
	mOptions.zeroCopy = zeroCopy;
	return Capture((void**)output, IMAGE_RGBA32F, timeout, NULL, stream);
}


// Open
bool gstCamera::Open()
{
	if( mStreaming )
		return true;

	// transition pipline to STATE_PLAYING
	LogInfo(LOG_GSTREAMER "opening gstCamera for streaming, transitioning pipeline to GST_STATE_PLAYING\n");
	
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
			printf(LOG_GSTREAMER "gstCamera NULL message after transitioning pipeline to PLAYING...\n");
#endif
	}
	else if( result != GST_STATE_CHANGE_SUCCESS )
	{
		LogError(LOG_GSTREAMER "gstCamera failed to set pipeline state to PLAYING (error %u)\n", result);
		return false;
	}

	checkMsgBus();
	usleep(100*1000);
	checkMsgBus();

	mStreaming = true;
	return true;
}
	
// Close
void gstCamera::Close()
{
	if( !mStreaming )
		return;

	// stop pipeline
	LogInfo(LOG_GSTREAMER "gstCamera -- stopping pipeline, transitioning to GST_STATE_NULL\n");

	const GstStateChangeReturn result = gst_element_set_state(mPipeline, GST_STATE_NULL);

	if( result != GST_STATE_CHANGE_SUCCESS )
		LogError(LOG_GSTREAMER "gstCamera failed to set pipeline state to PLAYING (error %u)\n", result);

	usleep(250*1000);	
	checkMsgBus();
	mStreaming = false;
	LogInfo(LOG_GSTREAMER "gstCamera -- pipeline stopped\n");
}

// checkMsgBus
void gstCamera::checkMsgBus()
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

