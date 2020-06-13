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
#include "gstUtility.h"

#include <gst/gst.h>
#include <gst/app/gstappsink.h>

#include <sstream> 
#include <unistd.h>
#include <string.h>

#include "cudaColorspace.h"
#include "logging.h"

#include "NvInfer.h"


// constructor
gstCamera::gstCamera( const videoOptions& options ) : videoSource(options)
{	
	mAppSink    = NULL;
	mBus        = NULL;
	mPipeline   = NULL;	
	mFrameCount = 0;
	mFormatYUV  = IMAGE_UNKNOWN;
	
	mBufferRGB.SetThreaded(false);
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
	

// Capture
bool gstCamera::Capture( void** output, imageFormat format, uint64_t timeout )
{
	//if( format == IMAGE_RGBA32F )
	//	return CaptureRGBA((float**)image, timeout, mOptions.zeroCopy);

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
	void* latestYUV = mBufferYUV.Next(RingBuffer::ReadLatestOnce);

	if( !latestYUV )
		return false;

	// allocate ringbuffer for colorspace conversion
	const size_t rgbBufferSize = imageFormatSize(format, GetWidth(), GetHeight());

	if( !mBufferRGB.Alloc(mOptions.numBuffers, rgbBufferSize, mOptions.zeroCopy ? RingBuffer::ZeroCopy : 0) )
	{
		LogError(LOG_GSTREAMER "gstCamera -- failed to allocate %u buffers (%zu bytes each)\n", mOptions.numBuffers, rgbBufferSize);
		return false;
	}

	// perform colorspace conversion
	void* nextRGB = mBufferRGB.Next(RingBuffer::Write);

	if( CUDA_FAILED(cudaConvertColor(latestYUV, mFormatYUV, nextRGB, format, GetWidth(), GetHeight())) )
	{
		LogError(LOG_GSTREAMER "gstCamera::Capture() -- unsupported image format (%s)\n", imageFormatToStr(format));
		LogError(LOG_GSTREAMER "                        supported formats are:\n");
		LogError(LOG_GSTREAMER "                            * rgb8\n");		
		LogError(LOG_GSTREAMER "                            * rgba8\n");		
		LogError(LOG_GSTREAMER "                            * rgb32f\n");		
		LogError(LOG_GSTREAMER "                            * rgba32f\n");

		return false;
	}

	*output = nextRGB;
	return true;
}


// CaptureRGBA
bool gstCamera::CaptureRGBA( float** output, unsigned long timeout, bool zeroCopy )
{
	mOptions.zeroCopy = zeroCopy;
	return Capture((void**)output, IMAGE_RGBA32F, timeout);
}
	

// gst_convert_format
imageFormat gst_convert_format( const char* format )
{
	if( !format )
		return IMAGE_UNKNOWN;
	
	if( strcasecmp(format, "rgb") == 0 )
		return IMAGE_RGB8;
	else if( strcasecmp(format, "yuy2") == 0 )
		return IMAGE_YUY2;
	else if( strcasecmp(format, "i420") == 0 )
		return IMAGE_I420;
	else if( strcasecmp(format, "nv12") == 0 )
		return IMAGE_NV12;
	else if( strcasecmp(format, "yv12") == 0 )
		return IMAGE_YV12;
	else if( strcasecmp(format, "yuyv") == 0 )
		return IMAGE_YUYV;
	else if( strcasecmp(format, "uyvy") == 0 )
		return IMAGE_UYVY;
	
	return IMAGE_UNKNOWN;
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
		LogError(LOG_GSTREAMER "gstCamera -- gst_app_sink_pull_sample() returned NULL...\n");
		return;
	}
	
	GstBuffer* gstBuffer = gst_sample_get_buffer(gstSample);
	
	if( !gstBuffer )
	{
		LogError(LOG_GSTREAMER "gstCamera -- gst_sample_get_buffer() returned NULL...\n");
		return;
	}
	
	// retrieve data
	GstMapInfo map; 

	if( !gst_buffer_map(gstBuffer, &map, GST_MAP_READ) ) 
	{
		LogError(LOG_GSTREAMER "gstCamera -- gst_buffer_map() failed...\n");
		return;
	}
	
	//gst_util_dump_mem(map.data, map.size); 

	void* gstData = map.data; //GST_BUFFER_DATA(gstBuffer);
	const uint32_t gstSize = map.size; //GST_BUFFER_SIZE(gstBuffer);
	
	if( !gstData )
	{
		LogError(LOG_GSTREAMER "gstCamera -- gst_buffer had NULL data pointer...\n");
		release_return;
	}
	
	// retrieve caps
	GstCaps* gstCaps = gst_sample_get_caps(gstSample);
	
	if( !gstCaps )
	{
		LogError(LOG_GSTREAMER "gstCamera -- gst_buffer had NULL caps...\n");
		release_return;
	}
	
	GstStructure* gstCapsStruct = gst_caps_get_structure(gstCaps, 0);
	
	if( !gstCapsStruct )
	{
		LogError(LOG_GSTREAMER "gstCamera -- caps had NULL structure...\n");
		release_return;
	}
	
	// on the first frame, print out the recieve caps
	if( mFrameCount == 0 )
		LogVerbose(LOG_GSTREAMER "gstCamera recieve caps:  %s\n", gst_caps_to_string(gstCaps));

	// get width & height of the buffer
	int width  = 0;
	int height = 0;
	
	if( !gst_structure_get_int(gstCapsStruct, "width", &width) ||
		!gst_structure_get_int(gstCapsStruct, "height", &height) )
	{
		LogError(LOG_GSTREAMER "gstCamera -- recieve caps missing width/height...\n");
		release_return;
	}
	
	if( width < 1 || height < 1 )
		release_return;
	
	mOptions.width  = width;
	mOptions.height = height;

	// verify format 
	if( mFormatYUV == IMAGE_UNKNOWN )
	{
		const char* formatStr = gst_structure_get_string(gstCapsStruct, "format");
		
		if( !formatStr )
		{
			LogError(LOG_GSTREAMER "gstCamera -- recieve caps missing format\n");
			release_return;
		}
		
		mFormatYUV = gst_convert_format(formatStr);
		
		if( mFormatYUV == IMAGE_UNKNOWN )
		{
			LogError(LOG_GSTREAMER "gstCamera -- v4l2 device %s does not have a compatible format\n", mOptions.resource.path.c_str());
			release_return;
		}
	}
	
	LogDebug(LOG_GSTREAMER "gstCamera recieved %ix%i %s frame (%u bytes)\n", width, height, imageFormatToStr(mFormatYUV), gstSize);
	
	// make sure ringbuffer is allocated
	if( !mBufferYUV.Alloc(mOptions.numBuffers, gstSize, RingBuffer::ZeroCopy) )
	{
		LogError(LOG_GSTREAMER "gstCamera -- failed to allocate %u buffers (%u bytes each)\n", mOptions.numBuffers, gstSize);
		release_return;
	}

	// copy to next ringbuffer
	void* nextBuffer = mBufferYUV.Peek(RingBuffer::Write);

	if( !nextBuffer )
	{
		LogError(LOG_GSTREAMER "gstCamera -- failed to retrieve next ringbuffer for writing\n");
		release_return;
	}

	memcpy(nextBuffer, gstData, gstSize);
	mBufferYUV.Next(RingBuffer::Write);
	mWaitEvent.Wake();
	mFrameCount++;
	
#if GST_CHECK_VERSION(1,0,0)
	gst_buffer_unmap(gstBuffer, &map);
#endif	

	release_return;
}


// buildLaunchStr
bool gstCamera::buildLaunchStr()
{
	std::ostringstream ss;

	if( mOptions.resource.protocol == "csi" )
	{
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
		
		ss << "video/x-raw ! appsink name=mysink";
	}
	else
	{
		ss << "v4l2src device=" << mOptions.resource.path << " ! ";
		ss << "video/x-raw, width=(int)" << GetWidth() << ", height=(int)" << GetHeight(); 

#if 0
	#if NV_TENSORRT_MAJOR >= 5
		ss << ", format=YUY2 ! videoconvert ! video/x-raw, format=RGB ! videoconvert !";
	#else
		ss << ", format=RGB ! videoconvert ! video/x-raw, format=RGB ! videoconvert !";
	#endif
#else
		ss << " ! ";
#endif

		ss << "appsink name=mysink";
	}
	
	mLaunchStr = ss.str();

	LogInfo(LOG_GSTREAMER "gstCamera pipeline string:\n");
	LogInfo(LOG_GSTREAMER "%s\n", mLaunchStr.c_str());

	return true;
}


// parseCameraStr
#if 0
bool gstCamera::parseCameraStr( const char* camera )
{
	if( !camera || strlen(camera) == 0 )
	{
		mSensorCSI = 0;
		mCameraStr = "0";
		return true;
	}

	mCameraStr = camera;

	// check if the string is a V4L2 device
	const char* prefixV4L2 = "/dev/video";

	const size_t prefixLength = strlen(prefixV4L2);
	const size_t cameraLength = strlen(camera);

	if( cameraLength < prefixLength )
	{
		const int result = sscanf(camera, "%i", &mSensorCSI);

		if( result == 1 && mSensorCSI >= 0 )
			return true;
	}
	else if( strncmp(camera, prefixV4L2, prefixLength) == 0 )
	{
		return true;
	}

	LogError(LOG_GSTREAMER "gstCamera::Create('%s') -- invalid camera device requested\n", camera);
	return false;
}
#endif

// Create
gstCamera* gstCamera::Create( uint32_t width, uint32_t height, const char* camera )
{
	videoOptions opt;

	if( !camera )
		camera = "csi://0";

	opt.resource = camera;
	opt.width    = width;
	opt.height   = height;
	opt.ioType   = videoOptions::INPUT;

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
	
	//if( !cam->parseCameraStr(options.resource.path.c_str()) )
	//	return NULL;

	// check desired frame sizes
	if( cam->mOptions.width == 0 )
		cam->mOptions.width = DefaultWidth;

	if( cam->mOptions.height == 0 )
		cam->mOptions.height = DefaultHeight;

	//cam->mDepth = cam->csiCamera() ? 12 : 24;	// NV12 or RGB
	//cam->mSize  = (cam->GetWidth() * cam->GetHeight() * cam->mDepth) / 8;

	cam->mOptions.loop = 0;	// disable looping for cameras

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

	
// discover
bool gstCamera::discover()
{
	if( mOptions.resource.protocol != "v4l2" )
		return true;
	
	GstDeviceProvider* deviceProvider = gst_device_provider_factory_get_by_name("v4l2deviceprovider");
	
	if( !deviceProvider )
	{
		LogError(LOG_GSTREAMER "gstCamera -- failed to create v4l2 device provider during discovery\n");
		return false;
	}
	
	GList* deviceList = gst_device_provider_get_devices(deviceProvider);

	if( !deviceList )
	{
		LogError(LOG_GSTREAMER "gstCamera -- didn't discover any v4l2 devices\n");
		return false;
	}

	GstDevice* device = NULL;
	
	for( GList* n=deviceList; n; n = n->next )
	{
		GstDevice* d = GST_DEVICE(n->data);
		LogVerbose(LOG_GSTREAMER "gstCamera -- found v4l2 device: %s\n", gst_device_get_display_name(d));
		
		GstStructure* properties = gst_device_get_properties(d);
		
		if( properties != NULL )
		{
			LogVerbose(LOG_GSTREAMER "%s\n", gst_structure_to_string(properties));
			
			const char* devicePath = gst_structure_get_string(properties, "device.path");
			
			if( devicePath != NULL && strcasecmp(devicePath, mOptions.resource.path.c_str()) == 0 )
			{
				device = d;
				break;
			}
		}
	}
	
	if( !device )
	{
		LogError(LOG_GSTREAMER "gstCamera -- could not find v4l2 device %s\n", mOptions.resource.path.c_str());
		return false;
	}
	
	GstCaps* device_caps = gst_device_get_caps(device);
	
	if( !device_caps )
	{
		LogError(LOG_GSTREAMER "gstCamera -- failed to retrieve caps for v4l2 device %s\n", mOptions.resource.path.c_str());
		return false;
	}
	
	const uint32_t numCaps = gst_caps_get_size(device_caps);
	LogVerbose(LOG_GSTREAMER "gstCamera -- found %u caps for v4l2 device %s\n", numCaps, mOptions.resource.path.c_str());
	
	for( uint32_t n=0; n < numCaps; n++ )
	{
		GstStructure* caps = gst_caps_get_structure(device_caps, n);
		
		if( !caps )
			continue;
		
		LogVerbose(LOG_GSTREAMER "[%u] %s\n", n, gst_structure_to_string(caps));
	}
	
	// TODO select the "best" caps from above
	GstStructure* caps = gst_caps_get_structure(device_caps, 0);
	
	if( !caps )
	{
		LogError(LOG_GSTREAMER "gstCamera -- failed to retrieve caps structure for v4l2 device %s\n", mOptions.resource.path.c_str());
		return false;
	}
	
	const char* capsName = gst_structure_get_name(caps);
	const char* format = gst_structure_get_string(caps, "format");
	
	mFormatYUV = gst_convert_format(format);
	
	LogVerbose(LOG_GSTREAMER "gstCamera -- v4l2 device format (%s, %s -> %s)\n", capsName, (format != NULL) ? format : "null", imageFormatToStr(mFormatYUV));
	
	if( mFormatYUV == IMAGE_UNKNOWN )
	{
		LogError(LOG_GSTREAMER "gstCamera -- could not find a compatible format for v4l2 device %s\n", mOptions.resource.path.c_str());
		return false;
	}
	
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
		LogError(LOG_GSTREAMER "gstCamera -- resource discovery and auto-negotiation failed\n");
		return false;
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
	
	// set device flags
	if( mOptions.resource.protocol == "csi" )
		mOptions.deviceType = videoOptions::DEVICE_CSI;
	else if( mOptions.resource.protocol == "v4l2" )
		mOptions.deviceType = videoOptions::DEVICE_V4L2;

	return true;
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

