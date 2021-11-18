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

#ifndef __GSTREAMER_BUFFER_MANAGER_H__
#define __GSTREAMER_BUFFER_MANAGER_H__

#include "gstUtility.h"
#include "imageFormat.h"
#include "videoOptions.h"

#include "Event.h"
#include "Mutex.h"
#include "RingBuffer.h"


#if !GST_CHECK_VERSION(1,0,0) && defined(ENABLE_NVMM)
#undef ENABLE_NVMM	// NVMM is only enabled for GStreamer 1.0 and newer
#endif

#ifdef ENABLE_NVMM
#define GST_CAPS_FEATURE_MEMORY_NVMM "memory:NVMM"
#endif


/**
 * gstBufferManager recieves GStreamer buffers from appsink elements and unpacks/maps 
 * them into CUDA address space, and handles colorspace conversion into RGB format.
 *
 * It can handle both normal CPU-based GStreamer buffers and NVMM memory which can
 * be mapped directly to the GPU without requiring memory copies using the CPU.
 *
 * To disable the use of NVMM memory, set -DENABLE_NVMM=OFF when building with CMake:
 *
 *     cmake -DENABLE_NVMM=OFF ../
 *
 * @ingroup codec
 */
class gstBufferManager
{
public:
	/**
	 * Constructor
	 */
	gstBufferManager( videoOptions* options );
	
	/**
	 * Destructor
	 */
	~gstBufferManager();
	
	/**
	 * Enqueue a GstBuffer from GStreamer.
	 */
	bool Enqueue( GstBuffer* buffer, GstCaps* caps );
	
	/**
	 * Dequeue the next frame.
	 */
	bool Dequeue( void** output, imageFormat format, uint64_t timeout=UINT64_MAX );

	/**
	 * Get the total number of frames that have been recieved.
	 */
	inline uint64_t GetFrameCount() const	{ return mFrameCount; }
	
protected:

	imageFormat   mFormatYUV;  /**< The YUV colorspace format coming from appsink (typically NV12 or YUY2) */
	RingBuffer    mBufferYUV;  /**< Ringbuffer of CPU-based YUV frames (non-NVMM) that come from appsink */
	RingBuffer    mBufferRGB;  /**< Ringbuffer of frames that have been converted to RGB colorspace */
	Event	    mWaitEvent;  /**< Event that gets triggered when a new frame is recieved */
	
	videoOptions* mOptions;    /**< Options of the gstDecoder / gstCamera object */			
	uint64_t	    mFrameCount; /**< Total number of frames that have been recieved */
	bool 	    mNvmmUsed;	  /**< Is NVMM memory actually used by the stream? */
	
#ifdef ENABLE_NVMM
	Mutex  mNvmmMutex;
	int    mNvmmFD;
	void*  mNvmmEGL;
	void*  mNvmmCUDA;
	size_t mNvmmSize;
	bool   mNvmmReleaseFD;
#endif
};
  
#endif
