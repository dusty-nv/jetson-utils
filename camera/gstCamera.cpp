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

#include "cudaMappedMemory.h"
#include "cudaYUV.h"
#include "cudaRGB.h"

#include "NvInfer.h"


// gstCameraSrcToString
const char* gstCameraSrcToString( gstCameraSrc src )
{
	if( src == GST_SOURCE_NVCAMERA )		return "GST_SOURCE_NVCAMERA";
	else if( src == GST_SOURCE_NVARGUS )	return "GST_SOURCE_NVARGUS";
	else if( src == GST_SOURCE_V4L2 )		return "GST_SOURCE_V4L2";
	else if( src == GST_SOURCE_RTSP )		return "GST_SOURCE_RTSP";

	return "UNKNOWN";
}


// constructor
gstCamera::gstCamera()
{	
	mAppSink    = NULL;
	mBus        = NULL;
	mPipeline   = NULL;	
	mSensorCSI  = -1;
	mStreaming  = false;

	mWidth  = 0;
	mHeight = 0;
	mDepth  = 0;
	mSize   = 0;
	mSource = GST_SOURCE_NVCAMERA;

	mLatestRGBA       = 0;
	mLatestRingbuffer = 0;
	mLatestRetrieved  = false;
	
	for( uint32_t n=0; n < NUM_RINGBUFFERS; n++ )
	{
		mRingbufferCPU[n] = NULL;
		mRingbufferGPU[n] = NULL;
		mRGBA[n]          = NULL;
	}

	mRGBAZeroCopy = false;
}


// destructor	
gstCamera::~gstCamera()
{
	Close();

	for( uint32_t n=0; n < NUM_RINGBUFFERS; n++ )
	{
		// free capture buffer
		if( mRingbufferCPU[n] != NULL )
		{
			CUDA(cudaFreeHost(mRingbufferCPU[n]));

			mRingbufferCPU[n] = NULL;
			mRingbufferGPU[n] = NULL;
		}

		// free convert buffer
		if( mRGBA[n] != NULL )
		{
			if( mRGBAZeroCopy )
				CUDA(cudaFreeHost(mRGBA[n]));
			else
				CUDA(cudaFree(mRGBA[n]));

			mRGBA[n] = NULL; 
		}
	}
}


// onEOS
void gstCamera::onEOS(_GstAppSink* sink, void* user_data)
{
	printf(LOG_GSTREAMER "gstCamera onEOS\n");
}


// onPreroll
GstFlowReturn gstCamera::onPreroll(_GstAppSink* sink, void* user_data)
{
	printf(LOG_GSTREAMER "gstCamera onPreroll\n");
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
bool gstCamera::Capture( void** cpu, void** cuda, uint64_t timeout )
{
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
	mRingMutex.Lock();
	const uint32_t latest = mLatestRingbuffer;
	const bool retrieved = mLatestRetrieved;
	mLatestRetrieved = true;
	mRingMutex.Unlock();
	
	// skip if it was already retrieved
	if( retrieved )
		return false;
	
	// set output pointers
	if( cpu != NULL )
		*cpu = mRingbufferCPU[latest];
	
	if( cuda != NULL )
		*cuda = mRingbufferGPU[latest];
	
	return true;
}


// CaptureRGBA
bool gstCamera::CaptureRGBA( float** output, unsigned long timeout, bool zeroCopy )
{
	void* cpu = NULL;
	void* gpu = NULL;

	if( !Capture(&cpu, &gpu, timeout) )
	{
		printf(LOG_GSTREAMER "gstCamera failed to capture frame\n");
		return false;
	}

	if( !ConvertRGBA(gpu, output, zeroCopy) )
	{
		printf(LOG_GSTREAMER "gstCamera failed to convert frame to RGBA\n");
		return false;
	}

	return true;
}
	

// ConvertRGBA
bool gstCamera::ConvertRGBA( void* input, float** output, bool zeroCopy )
{
	if( !input || !output )
		return false;
	
	// check if the buffers were previously allocated with a different zeroCopy option
	// if necessary, free them so they can be re-allocated with the correct option
	if( mRGBA[0] != NULL && zeroCopy != mRGBAZeroCopy )
	{
		for( uint32_t n=0; n < NUM_RINGBUFFERS; n++ )
		{
			if( mRGBA[n] != NULL )
			{
				if( mRGBAZeroCopy )
					CUDA(cudaFreeHost(mRGBA[n]));
				else
					CUDA(cudaFree(mRGBA[n]));

				mRGBA[n] = NULL; 
			}
		}

		mRGBAZeroCopy = false;	// reset for sanity
	}

	// check if the buffers need allocated
	if( !mRGBA[0] )
	{
		const size_t size = mWidth * mHeight * sizeof(float4);

		for( uint32_t n=0; n < NUM_RINGBUFFERS; n++ )
		{
			if( zeroCopy )
			{
				void* cpuPtr = NULL;
				void* gpuPtr = NULL;

				if( !cudaAllocMapped(&cpuPtr, &gpuPtr, size) )
				{
					printf(LOG_GSTREAMER "gstCamera -- failed to allocate zeroCopy memory for %ux%xu RGBA texture\n", mWidth, mHeight);
					return false;
				}

				if( cpuPtr != gpuPtr )
				{
					printf(LOG_GSTREAMER "gstCamera -- zeroCopy memory has different pointers, please use a UVA-compatible GPU\n");
					return false;
				}

				mRGBA[n] = gpuPtr;
			}
			else
			{
				if( CUDA_FAILED(cudaMalloc(&mRGBA[n], size)) )
				{
					printf(LOG_GSTREAMER "gstCamera -- failed to allocate memory for %ux%u RGBA texture\n", mWidth, mHeight);
					return false;
				}
			}
		}
		
		printf(LOG_GSTREAMER "gstCamera -- allocated %u RGBA ringbuffers\n", NUM_RINGBUFFERS);
		mRGBAZeroCopy = zeroCopy;
	}
	
	if( csiCamera() )
	{
		// MIPI CSI camera is NV12
		if( CUDA_FAILED(cudaNV12ToRGBA32((uint8_t*)input, (float4*)mRGBA[mLatestRGBA], mWidth, mHeight)) )
			return false;
	}
	else
	{
		// V4L2 webcam is RGB
		if( CUDA_FAILED(cudaRGB8ToRGBA32((uchar3*)input, (float4*)mRGBA[mLatestRGBA], mWidth, mHeight)) )
			return false;
	}
	
	*output     = (float*)mRGBA[mLatestRGBA];
	mLatestRGBA = (mLatestRGBA + 1) % NUM_RINGBUFFERS;
	return true;
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
		printf(LOG_GSTREAMER "gstCamera -- gst_app_sink_pull_sample() returned NULL...\n");
		return;
	}
	
	GstBuffer* gstBuffer = gst_sample_get_buffer(gstSample);
	
	if( !gstBuffer )
	{
		printf(LOG_GSTREAMER "gstCamera -- gst_sample_get_buffer() returned NULL...\n");
		return;
	}
	
	// retrieve
	GstMapInfo map; 

	if(	!gst_buffer_map(gstBuffer, &map, GST_MAP_READ) ) 
	{
		printf(LOG_GSTREAMER "gstCamera -- gst_buffer_map() failed...\n");
		return;
	}
	
	//gst_util_dump_mem(map.data, map.size); 

	void* gstData = map.data; //GST_BUFFER_DATA(gstBuffer);
	const uint32_t gstSize = map.size; //GST_BUFFER_SIZE(gstBuffer);
	
	if( !gstData )
	{
		printf(LOG_GSTREAMER "gstCamera -- gst_buffer had NULL data pointer...\n");
		release_return;
	}
	
	// retrieve caps
	GstCaps* gstCaps = gst_sample_get_caps(gstSample);
	
	if( !gstCaps )
	{
		printf(LOG_GSTREAMER "gstCamera -- gst_buffer had NULL caps...\n");
		release_return;
	}
	
	GstStructure* gstCapsStruct = gst_caps_get_structure(gstCaps, 0);
	
	if( !gstCapsStruct )
	{
		printf(LOG_GSTREAMER "gstCamera -- gst_caps had NULL structure...\n");
		release_return;
	}
	
	// get width & height of the buffer
	int width  = 0;
	int height = 0;
	
	if( !gst_structure_get_int(gstCapsStruct, "width", &width) ||
		!gst_structure_get_int(gstCapsStruct, "height", &height) )
	{
		printf(LOG_GSTREAMER "gstCamera -- gst_caps missing width/height...\n");
		release_return;
	}
	
	if( width < 1 || height < 1 )
		release_return;
	
	mWidth  = width;
	mHeight = height;
	mDepth  = (gstSize * 8) / (width * height);
	mSize   = gstSize;
	
	//printf(LOG_GSTREAMER "gstCamera recieved %ix%i frame (%u bytes, %u bpp)\n", width, height, gstSize, mDepth);
	
	// make sure ringbuffer is allocated
	if( !mRingbufferCPU[0] )
	{
		for( uint32_t n=0; n < NUM_RINGBUFFERS; n++ )
		{
			if( !cudaAllocMapped(&mRingbufferCPU[n], &mRingbufferGPU[n], gstSize) )
				printf(LOG_GSTREAMER "gstCamera -- failed to allocate ringbuffer %u  (size=%u)\n", n, gstSize);
		}
		
		printf(LOG_GSTREAMER "gstCamera -- allocated %u ringbuffers, %u bytes each\n", NUM_RINGBUFFERS, gstSize);
	}
	
	// copy to next ringbuffer
	const uint32_t nextRingbuffer = (mLatestRingbuffer + 1) % NUM_RINGBUFFERS;		
	
	//printf(LOG_GSTREAMER "gstCamera -- using ringbuffer #%u for next frame\n", nextRingbuffer);
	memcpy(mRingbufferCPU[nextRingbuffer], gstData, gstSize);
	gst_buffer_unmap(gstBuffer, &map); 
	//gst_buffer_unref(gstBuffer);
	gst_sample_unref(gstSample);
	
	
	// update and signal sleeping threads
	mRingMutex.Lock();
	mLatestRingbuffer = nextRingbuffer;
	mLatestRetrieved  = false;
	mRingMutex.Unlock();
	mWaitEvent.Wake();
}


// buildLaunchStr
bool gstCamera::buildLaunchStr( gstCameraSrc src )
{
	// gst-launch-1.0 nvcamerasrc fpsRange="30.0 30.0" ! 'video/x-raw(memory:NVMM), width=(int)1920, height=(int)1080, format=(string)I420, framerate=(fraction)30/1' ! \
	// nvvidconv flip-method=2 ! 'video/x-raw(memory:NVMM), format=(string)I420' ! fakesink silent=false -v
	// #define CAPS_STR "video/x-raw(memory:NVMM), width=(int)2592, height=(int)1944, format=(string)I420, framerate=(fraction)30/1"
	// #define CAPS_STR "video/x-raw(memory:NVMM), width=(int)1920, height=(int)1080, format=(string)I420, framerate=(fraction)30/1"

	//rtspsrc location=rtsp://10.0.1.103/Streaming/Channels/103 ! queue ! rtph264depay ! h264parse ! queue ! omxh264dec ! appsink name=mysink

	std::ostringstream ss;

	if( csiCamera() && (src == GST_SOURCE_NVARGUS || src == GST_SOURCE_NVCAMERA) )
	{
	#if NV_TENSORRT_MAJOR > 1 && NV_TENSORRT_MAJOR < 5	// if JetPack 3.1-3.3 (different flip-method)
		const int flipMethod = 0;					// Xavier (w/TRT5) camera is mounted inverted
	#else
		const int flipMethod = 2;
	#endif	

		if( src == GST_SOURCE_NVCAMERA )
			ss << "nvcamerasrc fpsRange=\"30.0 30.0\" ! video/x-raw(memory:NVMM), width=(int)" << mWidth << ", height=(int)" << mHeight << ", format=(string)NV12 ! nvvidconv flip-method=" << flipMethod << " ! "; //'video/x-raw(memory:NVMM), width=(int)1920, height=(int)1080, format=(string)I420, framerate=(fraction)30/1' ! ";
		else if( src == GST_SOURCE_NVARGUS )
			ss << "nvarguscamerasrc sensor-id=" << mSensorCSI << " ! video/x-raw(memory:NVMM), width=(int)" << mWidth << ", height=(int)" << mHeight << ", framerate=30/1, format=(string)NV12 ! nvvidconv flip-method=" << flipMethod << " ! ";
		
		ss << "video/x-raw ! appsink name=mysink";
	}
	else if ( src == GST_SOURCE_V4L2 )
	{
		ss << "v4l2src device=" << mCameraStr << " ! ";
		ss << "video/x-raw, width=(int)" << mWidth << ", height=(int)" << mHeight << ", "; 
		
	#if NV_TENSORRT_MAJOR >= 5
		ss << "format=YUY2 ! videoconvert ! video/x-raw, format=RGB ! videoconvert !";
	#else
		ss << "format=RGB ! videoconvert ! video/x-raw, format=RGB ! videoconvert !";
	#endif

		ss << "appsink name=mysink";
	}
	else if ( src == GST_SOURCE_RTSP )
	{
		ss << "rtspsrc location=" << mCameraStr << " ! ";
		ss << "queue ! rtph264depay ! h264parse ! queue ! omxh264dec ! ";
		ss << "videoconvert ! video/x-raw, format=RGB ! ";
		//ss << "videoconvert ! videoscale ! video/x-raw, format=RGB, width=" << mWidth << ", height=" << mHeight << " ! ";
		
		ss << "appsink name=mysink";
	}
	
	mLaunchStr = ss.str();

	printf(LOG_GSTREAMER "gstCamera pipeline string:\n");
	printf("%s\n", mLaunchStr.c_str());
	return true;
}


// parseCameraStr
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

	// check if the string is a rtsp streaming
	const char* prefixRTSP = "rtsp://";

	const size_t cameraLength = strlen(camera);

	if( cameraLength <= 2 )
	{ 
		const int result = sscanf(camera, "%i", &mSensorCSI);

		if( result == 1 && mSensorCSI >= 0 )
			return true;
	}
	else if( strncmp(camera, prefixV4L2, strlen(prefixV4L2)) == 0 )
	{
		mSource = GST_SOURCE_V4L2;
		return true;
	} 
	else if( strncmp(camera, prefixRTSP, strlen(prefixRTSP)) == 0 )
	{
		mSource = GST_SOURCE_RTSP;
		return true;
	}

	printf(LOG_GSTREAMER "gstCamera::Create('%s') -- invalid camera device requested...  \n", camera);
	return false;
}


// Create
gstCamera* gstCamera::Create( uint32_t width, uint32_t height, const char* camera )
{
	if( !gstreamerInit() )
	{
		printf(LOG_GSTREAMER "failed to initialize gstreamer API\n");
		return NULL;
	}
	
	gstCamera* cam = new gstCamera();
	
	if( !cam )
		return NULL;
	
	if( !cam->parseCameraStr(camera) )
		return NULL;

	cam->mWidth      = width;
	cam->mHeight     = height;
	cam->mDepth      = cam->csiCamera() ? 12 : 24;	// NV12 or RGB
	cam->mSize       = (width * height * cam->mDepth) / 8;

	if(cam->mSource == GST_SOURCE_V4L2 || cam->mSource == GST_SOURCE_RTSP)
	{	
		if( cam->mSensorCSI >= 0 )
 			cam->mSensorCSI = -1;
		if( !cam->init(cam->mSource) )
		{
			printf(LOG_GSTREAMER "failed to init gstCamera (%s, camera %s)\n", gstCameraSrcToString(cam->mSource), cam->mCameraStr.c_str());
			return NULL;
		}
	}
	else
	{
		cam->mSource = GST_SOURCE_NVARGUS;
		if( !cam->init(cam->mSource) )
		{
			cam->mSource = GST_SOURCE_NVCAMERA;
			if( !cam->init(cam->mSource) )
			{
				printf(LOG_GSTREAMER "failed to init gstCamera (%s, camera %s)\n", gstCameraSrcToString(cam->mSource), cam->mCameraStr.c_str());
				return NULL;
			}
		}
	}
	
	printf(LOG_GSTREAMER "gstCamera successfully initialized with %s, camera %s\n", gstCameraSrcToString(cam->mSource), cam->mCameraStr.c_str()); 
	return cam;
}


// Create
gstCamera* gstCamera::Create( const char* camera )
{
	return Create( DefaultWidth, DefaultHeight, camera );
}


// init
bool gstCamera::init( gstCameraSrc src )
{
	GError* err = NULL;
	printf(LOG_GSTREAMER "gstCamera attempting to initialize with %s, camera %s\n", gstCameraSrcToString(src), mCameraStr.c_str());

	// build pipeline string
	if( !buildLaunchStr(src) )
	{
		printf(LOG_GSTREAMER "gstCamera failed to build pipeline string\n");
		return false;
	}

	// launch pipeline
	mPipeline = gst_parse_launch(mLaunchStr.c_str(), &err);

	if( err != NULL )
	{
		printf(LOG_GSTREAMER "gstCamera failed to create pipeline\n");
		printf(LOG_GSTREAMER "   (%s)\n", err->message);
		g_error_free(err);
		return false;
	}

	GstPipeline* pipeline = GST_PIPELINE(mPipeline);

	if( !pipeline )
	{
		printf(LOG_GSTREAMER "gstCamera failed to cast GstElement into GstPipeline\n");
		return false;
	}	

	// retrieve pipeline bus
	/*GstBus**/ mBus = gst_pipeline_get_bus(pipeline);

	if( !mBus )
	{
		printf(LOG_GSTREAMER "gstCamera failed to retrieve GstBus from pipeline\n");
		return false;
	}

	// add watch for messages (disabled when we poll the bus ourselves, instead of gmainloop)
	//gst_bus_add_watch(mBus, (GstBusFunc)gst_message_print, NULL);

	// get the appsrc
	GstElement* appsinkElement = gst_bin_get_by_name(GST_BIN(pipeline), "mysink");
	GstAppSink* appsink = GST_APP_SINK(appsinkElement);

	if( !appsinkElement || !appsink)
	{
		printf(LOG_GSTREAMER "gstCamera failed to retrieve AppSink element from pipeline\n");
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
	
	return true;
}


// Open
bool gstCamera::Open()
{
	if( mStreaming )
		return true;

	// transition pipline to STATE_PLAYING
	printf(LOG_GSTREAMER "opening gstCamera for streaming, transitioning pipeline to GST_STATE_PLAYING\n");
	
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
		printf(LOG_GSTREAMER "gstCamera failed to set pipeline state to PLAYING (error %u)\n", result);
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
	printf(LOG_GSTREAMER "closing gstCamera for streaming, transitioning pipeline to GST_STATE_NULL\n");

	const GstStateChangeReturn result = gst_element_set_state(mPipeline, GST_STATE_NULL);

	if( result != GST_STATE_CHANGE_SUCCESS )
		printf(LOG_GSTREAMER "gstCamera failed to set pipeline state to PLAYING (error %u)\n", result);

	usleep(250*1000);
	mStreaming = false;
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

