/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

#include "gstBufferManager.h"
#include "cudaColorspace.h"
#include "timespec.h"
#include "logging.h"


#ifdef ENABLE_NVMM
#include <nvbuf_utils.h>
#include <cuda_egl_interop.h>
#include <NvInfer.h>

#if NV_TENSORRT_MAJOR > 8 || (NV_TENSORRT_MAJOR == 8 && NV_TENSORRT_MINOR >= 4)
#include <nvbufsurface.h>   // JetPack 5
#endif
#endif


// constructor
gstBufferManager::gstBufferManager( videoOptions* options )
{	
	mOptions    = options;
	mFormatYUV  = IMAGE_UNKNOWN;
	mFrameCount = 0;
	mLastTimestamp = 0;
	mNvmmUsed   = false;
	
#ifdef ENABLE_NVMM
	mNvmmFD        = -1;
	mNvmmEGL       = NULL;
	mNvmmCUDA      = NULL;
	mNvmmSize      = 0;
	mNvmmReleaseFD = false;
#endif
	
	mBufferRGB.SetThreaded(false);
}


// destructor
gstBufferManager::~gstBufferManager()
{
	
}


// Enqueue
bool gstBufferManager::Enqueue( GstBuffer* gstBuffer, GstCaps* gstCaps )
{
	if( !gstBuffer || !gstCaps )
		return false;

	uint64_t timestamp = apptime_nano();

#if GST_CHECK_VERSION(1,0,0)	
	// map the buffer memory for read access
	GstMapInfo map; 
	
	if( !gst_buffer_map(gstBuffer, &map, GST_MAP_READ) ) 
	{ 
		LogError(LOG_GSTREAMER "gstBufferManager -- failed to map gstreamer buffer memory\n");
		return false;
	}
	
	const void* gstData = map.data;
	const gsize gstSize = map.maxsize; //map.size;

	if( !gstData )
	{
		LogError(LOG_GSTREAMER "gstBufferManager -- gst_buffer_map had NULL data pointer...\n");
		return false;
	}

	if( map.maxsize > map.size && mFrameCount == 0 ) 
	{
		LogWarning(LOG_GSTREAMER "gstBufferManager -- map buffer size was less than max size (%zu vs %zu)\n", map.size, map.maxsize);
	}
#else
	// retrieve data pointer
	void* gstData = GST_BUFFER_DATA(gstBuffer);
	const guint gstSize = GST_BUFFER_SIZE(gstBuffer);
	
	if( !gstData )
	{
		LogError(LOG_GSTREAMER "gstBufferManager -- gst_buffer had NULL data pointer...\n");
		return false;
	}
#endif
	// on the first frame, print out the recieve caps
	if( mFrameCount == 0 )
		LogVerbose(LOG_GSTREAMER "gstBufferManager recieve caps:  %s\n", gst_caps_to_string(gstCaps));

	// retrieve caps structure
	GstStructure* gstCapsStruct = gst_caps_get_structure(gstCaps, 0);
	
	if( !gstCapsStruct )
	{
		LogError(LOG_GSTREAMER "gstBufferManager -- gst_caps had NULL structure...\n");
		return false;
	}
	
	// retrieve the width and height of the buffer
	int width  = 0;
	int height = 0;
	
	if( !gst_structure_get_int(gstCapsStruct, "width", &width) ||
		!gst_structure_get_int(gstCapsStruct, "height", &height) )
	{
		LogError(LOG_GSTREAMER "gstBufferManager -- gst_caps missing width/height...\n");
		return false;
	}
	
	if( width < 1 || height < 1 )
		return false;
	
	mOptions->width = width;
	mOptions->height = height;

	// verify format 
	if( mFrameCount == 0 )
	{
		mFormatYUV = gst_parse_format(gstCapsStruct);
		
		if( mFormatYUV == IMAGE_UNKNOWN )
		{
			LogError(LOG_GSTREAMER "gstBufferManager -- stream %s does not have a compatible decoded format\n", mOptions->resource.c_str());
			return false;
		}
		
		LogVerbose(LOG_GSTREAMER "gstBufferManager -- recieved first frame, codec=%s format=%s width=%u height=%u size=%zu\n", videoOptions::CodecToStr(mOptions->codec), imageFormatToStr(mFormatYUV), mOptions->width, mOptions->height, gstSize);
	}

	//LogDebug(LOG_GSTREAMER "gstBufferManager -- recieved %ix%i frame (%zu bytes)\n", width, height, gstSize);
		
#ifdef ENABLE_NVMM
	// check for NVMM buffer	
	GstCapsFeatures* gstCapsFeatures = gst_caps_get_features(gstCaps, 0);
	
	if( gst_caps_features_contains(gstCapsFeatures, GST_CAPS_FEATURE_MEMORY_NVMM))
	{
		mNvmmUsed = true;
		int nvmmFD = -1;
		
		if( mFrameCount == 0 )
			LogVerbose(LOG_GSTREAMER "gstBufferManager -- recieved NVMM memory\n");
	
	#if NV_TENSORRT_MAJOR > 8 || (NV_TENSORRT_MAJOR == 8 && NV_TENSORRT_MINOR >= 4)
		NvBufSurface* surf = (NvBufSurface*)map.data;
		nvmmFD = surf->surfaceList[0].bufferDesc;
	#else
		if( ExtractFdFromNvBuffer(map.data, &nvmmFD) != 0 )
		{
			LogError(LOG_GSTREAMER "gstBufferManager -- failed to get FD from NVMM memory\n");
			return false;
		}
	#endif
	
		NvBufferParams nvmmParams;
	
		if( NvBufferGetParams(nvmmFD, &nvmmParams) != 0 )
		{
			LogError(LOG_GSTREAMER "gstBufferManager -- failed to get NVMM buffer params\n");
			return false;
		}
	
	#ifdef DEBUG
		LogVerbose(LOG_GSTREAMER "gstBufferManager -- NVMM buffer payload type:  %s\n", nvmmParams.payloadType == NvBufferPayload_MemHandle ? "MemHandle" : "SurfArray");
		LogVerbose(LOG_GSTREAMER "gstBufferManager -- NVMM buffer planes:  %u   format=%u\n", nvmmParams.num_planes, (uint32_t)nvmmParams.pixel_format);
		
		for( uint32_t n=0; n < nvmmParams.num_planes; n++ )
			LogVerbose(LOG_GSTREAMER "gstBufferManager -- NVMM buffer plane %u:  %ux%u\n", n, nvmmParams.width[n], nvmmParams.height[n]);
	#endif

		EGLImageKHR eglImage = NvEGLImageFromFd(NULL, nvmmFD);
		
		if( !eglImage )
		{
			LogError(LOG_GSTREAMER "gstBufferManager -- failed to map EGLImage from NVMM buffer\n");
			return false;
		}
		
		// nvfilter memory comes from nvvidconv, which handles NvReleaseFd() internally
		GstMemory* gstMemory = gst_buffer_peek_memory(gstBuffer, 0);
		
		if( !gstMemory )
		{
			LogError(LOG_GSTREAMER "gstBufferManager -- failed to retrieve GstMemory object from GstBuffer\n");
			return false;
		}
		
		const bool nvmmReleaseFD = (g_strcmp0(gstMemory->allocator->mem_type, "nvfilter") != 0);	
		
		// update latest frame so capture thread can grab it
		mNvmmMutex.Lock();
		
		if( mNvmmEGL != NULL )
		{
			NvDestroyEGLImage(NULL, mNvmmEGL);
			
			if( mNvmmReleaseFD )
				NvReleaseFd(mNvmmFD);
		}
		
		mNvmmFD = nvmmFD;
		mNvmmEGL = eglImage;
		mNvmmReleaseFD = nvmmReleaseFD;
		
		mNvmmMutex.Unlock();
	}
	else
	{
		mNvmmUsed = false;
	}
#endif

	// handle CPU path (non-NVMM)
	if( !mNvmmUsed )
	{
		// allocate image ringbuffer
		if( !mBufferYUV.Alloc(mOptions->numBuffers, gstSize, RingBuffer::ZeroCopy) )
		{
			LogError(LOG_GSTREAMER "gstBufferManager -- failed to allocate %u image buffers (%zu bytes each)\n", mOptions->numBuffers, gstSize);
			return false;
		}

		// copy to next image ringbuffer
		void* nextBuffer = mBufferYUV.Peek(RingBuffer::Write);

		if( !nextBuffer )
		{
			LogError(LOG_GSTREAMER "gstBufferManager -- failed to retrieve next image ringbuffer for writing\n");
			return false;
		}

		memcpy(nextBuffer, gstData, gstSize);
		mBufferYUV.Next(RingBuffer::Write);
	}

	// handle timestamps in either case (CPU or NVMM path)
	size_t timestamp_size = sizeof(uint64_t);

	// allocate timestamp ringbuffer (GPU only if not ZeroCopy)
	if( !mTimestamps.Alloc(mOptions->numBuffers, timestamp_size, RingBuffer::ZeroCopy) )
	{
		LogError(LOG_GSTREAMER "gstBufferManager -- failed to allocate %u timestamp buffers (%zu bytes each)\n", mOptions->numBuffers, timestamp_size);
		return false;
	}

	// copy to next timestamp ringbuffer
	void* nextTimestamp = mTimestamps.Peek(RingBuffer::Write);

	if( !nextTimestamp )
	{
		LogError(LOG_GSTREAMER "gstBufferManager -- failed to retrieve next timestamp ringbuffer for writing\n");
		return false;
	}

	if( GST_BUFFER_DTS_IS_VALID(gstBuffer) || GST_BUFFER_PTS_IS_VALID(gstBuffer) )
	{
		timestamp = GST_BUFFER_DTS_OR_PTS(gstBuffer);
	}

	memcpy(nextTimestamp, (void*)&timestamp, timestamp_size);
	mTimestamps.Next(RingBuffer::Write);

	mWaitEvent.Wake();
	mFrameCount++;
	
#if GST_CHECK_VERSION(1,0,0)
	gst_buffer_unmap(gstBuffer, &map);
#endif
	
	return true;
}


// Dequeue
int gstBufferManager::Dequeue( void** output, imageFormat format, uint64_t timeout, cudaStream_t stream )
{
	// wait until a new frame is recieved
	if( !mWaitEvent.Wait(timeout) )
		return 0;

	void* latestYUV = NULL;
	
#ifdef ENABLE_NVMM
	if( mNvmmUsed )
	{
		mNvmmMutex.Lock();
		
		const int nvmmFD = mNvmmFD;
		const bool nvmmReleaseFD = mNvmmReleaseFD;
		EGLImageKHR eglImage = (EGLImageKHR)mNvmmEGL;
		
		mNvmmFD = -1;
		mNvmmEGL = NULL;
		mNvmmReleaseFD = false;
		
		mNvmmMutex.Unlock();
		
		if( !eglImage )
			return -1;
		
		// map EGLImage into CUDA array
		cudaGraphicsResource* eglResource = NULL;
		cudaEglFrame eglFrame;
		
		if( CUDA_FAILED(cudaGraphicsEGLRegisterImage(&eglResource, eglImage, cudaGraphicsRegisterFlagsReadOnly)) )
			return -1;
		
		if( CUDA_FAILED(cudaGraphicsResourceGetMappedEglFrame(&eglFrame, eglResource, 0, 0)) )
			return -1;

		if( eglFrame.planeCount != 2 )
			LogWarning(LOG_GSTREAMER "gstBufferManager -- unexpected number of planes in NVMM buffer (%u vs 2 expected)\n", eglFrame.planeCount);

		if( eglFrame.planeDesc[0].width != mOptions->width || eglFrame.planeDesc[0].height != mOptions->height )
		{
			LogError(LOG_GSTREAMER "gstBufferManager -- NVMM EGLImage dimensions mismatch (%ux%u when expected %ux%u)", eglFrame.planeDesc[0].width, eglFrame.planeDesc[0].height, mOptions->width, mOptions->height);
			return -1;
		}
		
		if( eglFrame.frameType != cudaEglFrameTypeArray )  // cudaEglFrameTypePitch
		{
			LogError(LOG_GSTREAMER "gstBufferManager -- NVMM had unexpected frame type (was pitched pointer, expected CUDA array)\n");
			return -1;
		}
		
		// NV12 buffers have multiple planes (Y @ full res and UV @ half res)
		const size_t maxPlanes = 16;
		size_t planePitch[maxPlanes];
		size_t planeSize[maxPlanes];
		size_t sizeYUV = 0;
		
		for( uint32_t n=0; n < eglFrame.planeCount && n < maxPlanes; n++ )
		{
			cudaChannelFormatDesc arrayDesc;
			cudaExtent arrayExtent;
			
			CUDA(cudaArrayGetInfo(&arrayDesc, &arrayExtent, NULL, eglFrame.frame.pArray[n]));
			
			const size_t bpp = arrayDesc.x + arrayDesc.y + arrayDesc.z;
			
			planePitch[n] = (bpp * arrayExtent.width) / 8;
			planeSize[n] = planePitch[n] * arrayExtent.height;
			
			sizeYUV += planeSize[n];
			
		#ifdef DEBUG
			LogDebug(LOG_GSTREAMER "gstBufferManager -- plane=%u x=%i y=%i z=%i  w=%zu h=%zu d=%zu  pitch=%zu size=%zu\n", n, arrayDesc.x, arrayDesc.y, arrayDesc.z, arrayExtent.width, arrayExtent.height, arrayExtent.depth, planePitch[n], planeSize[n]);
		#endif
		}

		// allocate CUDA memory for the image
		if( !mNvmmCUDA || mNvmmSize != sizeYUV )
		{
			CUDA_FREE(mNvmmCUDA);
			
			if( CUDA_FAILED(cudaMalloc(&mNvmmCUDA, sizeYUV)) )
				return -1;
		}
		
		// copy arrays into linear memory (so our CUDA kernels can use it)
		size_t planeOffset = 0;
		
		for( uint32_t n=0; n < eglFrame.planeCount && n < maxPlanes; n++ )
		{
			if( CUDA_FAILED(cudaMemcpy2DFromArrayAsync(((uint8_t*)mNvmmCUDA) + planeOffset, planePitch[n], eglFrame.frame.pArray[n], 0, 0, planePitch[n], eglFrame.planeDesc[n].height, cudaMemcpyDeviceToDevice)) )
				return -1;
		
			planeOffset += planeSize[n];
		}

		latestYUV = mNvmmCUDA;
		
		CUDA(cudaGraphicsUnregisterResource(eglResource));
		NvDestroyEGLImage(NULL, eglImage);
		
		if( nvmmReleaseFD )
			NvReleaseFd(nvmmFD);
	}
#endif

	// handle the CPU path (non-NVMM)
	if( !mNvmmUsed )
		latestYUV = mBufferYUV.Next(RingBuffer::ReadLatestOnce);

	if( !latestYUV )
		return -1;

	// handle timestamp (both paths)
	void* pLastTimestamp = NULL;
	pLastTimestamp = mTimestamps.Next(RingBuffer::ReadLatestOnce);

	if( !pLastTimestamp )
	{
		LogWarning(LOG_GSTREAMER "gstBufferManager -- failed to retrieve timestamp buffer (default to 0)\n");
		mLastTimestamp = 0;
	}
	else
	{
		mLastTimestamp = *((uint64_t*)pLastTimestamp);
	}

	// output raw image if conversion format is unknown
	if ( format == IMAGE_UNKNOWN )
	{
		*output = latestYUV;
		return 1;
	}

	// allocate ringbuffer for colorspace conversion
	const size_t rgbBufferSize = imageFormatSize(format, mOptions->width, mOptions->height);

	if( !mBufferRGB.Alloc(mOptions->numBuffers, rgbBufferSize, mOptions->zeroCopy ? RingBuffer::ZeroCopy : 0) )
	{
		LogError(LOG_GSTREAMER "gstBufferManager -- failed to allocate %u buffers (%zu bytes each)\n", mOptions->numBuffers, rgbBufferSize);
		return -1;
	}

	// perform colorspace conversion
	void* nextRGB = mBufferRGB.Next(RingBuffer::Write);

	if( CUDA_FAILED(cudaConvertColor(latestYUV, mFormatYUV, nextRGB, format, mOptions->width, mOptions->height, stream)) )
	{
		LogError(LOG_GSTREAMER "gstBufferManager -- unsupported image format (%s)\n", imageFormatToStr(format));
		LogError(LOG_GSTREAMER "                    supported formats are:\n");
		LogError(LOG_GSTREAMER "                       * rgb8\n");		
		LogError(LOG_GSTREAMER "                       * rgba8\n");		
		LogError(LOG_GSTREAMER "                       * rgb32f\n");		
		LogError(LOG_GSTREAMER "                       * rgba32f\n");

		return -1;
	}

	*output = nextRGB;
	return 1;
}

