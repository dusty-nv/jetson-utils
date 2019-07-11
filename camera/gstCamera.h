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

#ifndef __GSTREAMER_CAMERA_H__
#define __GSTREAMER_CAMERA_H__

#include <gst/gst.h>
#include <string>

#include "Mutex.h"
#include "Event.h"


// Forward declarations
struct _GstAppSink;


/**
 * Enumeration of camera input source methods
 * @ingroup gstCamera
 */
enum gstCameraSrc
{
	GST_SOURCE_NVCAMERA,	/* use nvcamerasrc element */
	GST_SOURCE_NVARGUS,		/* use nvargussrc element */
	GST_SOURCE_V4L2,		/* use v4l2src element */
	GST_SOURCE_RTSP			/* use rtspsrc element */
};

/**
 * Stringize function to convert gstCameraSrc enum to text
 * @ingroup gstCamera
 */
const char* gstCameraSrcToString( gstCameraSrc src );	


/**
 * MIPI CSI and V4L2 camera capture using GStreamer and `nvarguscamerasrc` or `v4l2src` elements.
 * gstCamera supports both MIPI CSI cameras and V4L2-compliant devices like USB webcams.
 *
 * Examples of MIPI CSI cameras that work out of the box are the OV5693 module from the
 * Jetson TX1/TX2 devkits, and the IMX219 sensor from the Raspberry Pi Camera Module v2.
 *
 * For MIPI CSI cameras, the GStreamer element `nvarguscamerasrc` will be used for capture.
 * For V4L2 devices, the GStreamer element `v4l2src` will be used for camera capture.
 *
 * gstCamera uses CUDA underneath for any necessary colorspace conversion, and provides
 * the captured image frames in CUDA device memory, or zero-copy shared CPU/GPU memory.
 *
 * @ingroup gstCamera
 */
class gstCamera
{
public:
	/**
	 * Create a MIPI CSI or V4L2 camera device.
	 *
	 * gstCamera will use the `nvarguscamerasrc` GStreamer element for MIPI CSI cameras,
	 * and the `v4l2src` GStreamer element for capturing V4L2 cameras, like USB webcams.
	 *
	 * The camera will be created with a resolution indicated by gstCamera::DefaultWidth
	 * and gstCamera::DefaultHeight (1280x720 by default).
	 *
	 * @param camera Camera device to use.  If using MIPI CSI, this string can be `NULL`
	 *			  to default to CSI camera 0, otherwise the string should contain the
	 *			  device index of the CSI camera (e.g. `"0"` for CSI camera 0 or `"1"`
	 *               for CSI camera 1, ect).  If using V4L2, the string should contain
	 *               the `/dev/video` node to use (e.g. `"/dev/video0"` for V4L2 camera 0).
	 *               By default, `camera` parameter is NULL and MIPI CSI camera 0 is used.
	 *
	 * @returns A pointer to the created gstCamera device, or NULL if there was an error.
	 */
	static gstCamera* Create( const char* camera=NULL ); // use MIPI CSI camera by default

	/**
	 * Create a MIPI CSI or V4L2 camera device.
	 *
	 * gstCamera will use the `nvarguscamerasrc` GStreamer element for MIPI CSI cameras,
	 * and the `v4l2src` GStreamer element for capturing V4L2 cameras, like USB webcams.
	 *
	 * @param width desired width (in pixels) of the camera resolution.  
	 *              This should be from a format that the camera supports.
	 *
	 * @param height desired height (in pixels) of the camera resolution.  
	 *               This should be from a format that the camera supports.
	 *
	 * @param camera Camera device to use.  If using MIPI CSI, this string can be `NULL`
	 *			  to default to CSI camera 0, otherwise the string should contain the
	 *			  device index of the CSI camera (e.g. `"0"` for CSI camera 0 or `"1"`
	 *               for CSI camera 1, ect).  If using V4L2, the string should contain
	 *               the `/dev/video` node to use (e.g. `"/dev/video0"` for V4L2 camera 0).
	 *               By default, `camera` parameter is NULL and MIPI CSI camera 0 is used.
	 *
	 * @returns A pointer to the created gstCamera device, or NULL if there was an error.
	 */
	static gstCamera* Create( uint32_t width, uint32_t height, const char* camera=NULL );
	
	/**
	 * Release the camera interface and resources.
	 * Destroying the camera will also Close() the stream if it is still open.
	 */
	~gstCamera();

	/**
	 * Begin streaming the camera.
	 * After Open() is called, frames from the camera will begin to be captured.
	 *
	 * Open() is not stricly necessary to call, if you call one of the Capture()
	 * functions they will first check to make sure that the stream is opened,
	 * and if not they will open it automatically for you.
	 *
	 * @returns `true` on success, `false` if an error occurred opening the stream.
	 */
	bool Open();

	/**
	 * Stop streaming the camera.
	 * @note Close() is automatically called by the camera's destructor when
	 * it gets deleted, so you do not explicitly need to call Close() before
	 * exiting the program if you delete your camera object.
	 */
	void Close();
	
	/**
	 * Check if the camera is streaming or not.
	 * @returns `true` if the camera is streaming (open), or `false` if it's closed.
	 */
	inline bool IsStreaming() const	   { return mStreaming; }

	/**
	 * Capture the next image frame from the camera.
	 *
	 * For MIPI CSI cameras, Capture() will provide an image in YUV (NV12) format.
	 * For V4L2 devices, Capture() will provide an image in RGB (24-bit) format.
	 *
	 * The captured images reside in shared CPU/GPU memory, also known as CUDA
	 * mapped memory or zero-copy memory.  Hence it is unnessary to copy them to GPU.
	 * This memory is managed internally by gstCamera, so don't attempt to free it.
	 *
	 * @param[out] cpu Pointer that gets returned to the image in CPU address space.
	 * @param[out] cuda Pointer that gets returned to the image in GPU address space.
	 *
	 * @param[in] timeout The time in milliseconds for the calling thread to wait to
	 *                    return if a new camera frame isn't recieved by that time.
	 *                    If timeout is 0, the calling thread will return immediately
	 *                    if a new frame isn't already available.
	 *                    If timeout is UINT64_MAX, the calling thread will wait
	 *                    indefinetly for a new frame to arrive (this is the default behavior).
	 *
	 * @returns `true` if a frame was successfully captured, otherwise `false` if a timeout
	 *               or error occurred, or if timeout was 0 and a frame wasn't ready.
	 */
	bool Capture( void** cpu, void** cuda, uint64_t timeout=UINT64_MAX );

	/**
	 * Capture the next image frame from the camera and convert it to float4 RGBA format,
	 * with pixel intensities ranging between 0.0 and 255.0.
	 *
	 * Internally, CaptureRGBA() first calls Capture() and then ConvertRGBA().
	 * The ConvertRGBA() function uses CUDA, so if you want to capture from a different 
	 * thread than your CUDA device, use the Capture() and ConvertRGBA() functions.
	 *
	 * @param[out] image Pointer that gets returned to the image in GPU address space,
	 *                   or if the zeroCopy parameter is true, then the pointer is valid
	 *                   in both CPU and GPU address spaces.  Do not manually free the image memory, 
	 *                   it is managed internally.  The image is in float4 RGBA format.
	 *                   The size of the image is:  `GetWidth() * GetHeight() * sizeof(float) * 4`
	 *
	 * @param[in] timeout The time in milliseconds for the calling thread to wait to
	 *                    return if a new camera frame isn't recieved by that time.
	 *                    If timeout is 0, the calling thread will return immediately
	 *                    if a new frame isn't already available.
	 *                    If timeout is UINT64_MAX, the calling thread will wait
	 *                    indefinetly for a new frame to arrive (this is the default behavior).
	 *
	 * @param[in] zeroCopy If `true`, the image will reside in shared CPU/GPU memory.
	 *                     If `false`, the image will only be accessible from the GPU.
	 *                     You would need to set zeroCopy to `true` if you wanted to
	 *                     access the image pixels from the CPU.  Since this isn't
	 *                     generally the case, the default is `false` (GPU only).
	 *
	 * @returns `true` if a frame was successfully captured, otherwise `false` if a timeout
	 *               or error occurred, or if timeout was 0 and a frame wasn't ready.
	 */
	bool CaptureRGBA( float** image, uint64_t timeout=UINT64_MAX, bool zeroCopy=false );
	
	/**
	 * Convert an image to float4 RGBA that was previously aquired with Capture().
	 * This function uses CUDA to perform the colorspace conversion to float4 RGBA,
 	 * with pixel intensities ranging from 0.0 to 255.0.
	 *
	 * @param[in] input Pointer to the input image, typically the pointer from Capture().
	 *                  If this is a MIPI CSI camera, it's expected to be in YUV (NV12) format.
	 *                  If this is a V4L2 device, it's expected to be in RGB (24-bit) format.
	 *                  In both cases, these are the formats that Capture() provides the image in.
	 *
	 * @param[out] output Pointer that gets returned to the image in GPU address space,
	 *                    or if the zeroCopy parameter is true, then the pointer is valid
	 *                    in both CPU and GPU address spaces.  Do not manually free the image memory, 
	 *                   it is managed internally.  The image is in float4 RGBA format.
	 *                   The size of the image is:  `GetWidth() * GetHeight() * sizeof(float) * 4`
	 *
	 * @param[in] zeroCopy If `true`, the image will reside in shared CPU/GPU memory.
	 *                     If `false`, the image will only be accessible from the GPU.
	 *                     You would need to set zeroCopy to `true` if you wanted to
	 *                     access the image pixels from the CPU.  Since this isn't
	 *                     generally the case, the default is `false` (GPU only).
	 * 
	 * @returns `true` on success, `false` if an error occurred.
	 */
	bool ConvertRGBA( void* input, float** output, bool zeroCopy=false );
	
	/**
	 * Return the width of the camera.
	 */
	inline uint32_t GetWidth() const	   { return mWidth; }

	/**
	 * Return the height of the camera.
	 */
	inline uint32_t GetHeight() const	   { return mHeight; }

	/**
	 * Return the pixel bit depth of the camera (measured in bits).
	 * This will be 12 for MIPI CSI cameras (YUV NV12 format)
	 * or 24 for VL42 cameras (RGB 24-bit).
	 */
	inline uint32_t GetPixelDepth() const { return mDepth; }

	/**
	 * Return the size (in bytes) of a camera frame from Capture().
	 *
	 * @note this is not the size of the converted float4 RGBA image
	 *       from Convert(), but rather the YUV (NV12) or RGB (24-bit)
	 *       image that gets aquired by the Capture() function.
	 *       To calculate the size of the converted float4 RGBA image,
	 *       take:  `GetWidth() * GetHeight() * sizeof(float) * 4`
	 */
	inline uint32_t GetSize() const	   { return mSize; }
	
	/**
	 * Default camera width, unless otherwise specified during Create()
 	 */
	static const uint32_t DefaultWidth  = 1280;

	/**
	 * Default camera height, unless otherwise specified during Create()
 	 */
	static const uint32_t DefaultHeight = 720;
	
private:
	static void onEOS(_GstAppSink* sink, void* user_data);
	static GstFlowReturn onPreroll(_GstAppSink* sink, void* user_data);
	static GstFlowReturn onBuffer(_GstAppSink* sink, void* user_data);

	gstCamera();

	bool init( gstCameraSrc src );
	bool buildLaunchStr( gstCameraSrc src );
	bool parseCameraStr( const char* camera );

	void checkMsgBus();
	void checkBuffer();
	
	_GstBus*     mBus;
	_GstAppSink* mAppSink;
	_GstElement* mPipeline;
	gstCameraSrc mSource;

	std::string  mLaunchStr;
	std::string  mCameraStr;

	uint32_t mWidth;
	uint32_t mHeight;
	uint32_t mDepth;
	uint32_t mSize;

	static const uint32_t NUM_RINGBUFFERS = 16;
	
	void* mRingbufferCPU[NUM_RINGBUFFERS];
	void* mRingbufferGPU[NUM_RINGBUFFERS];
	
	Event mWaitEvent;
	Mutex mWaitMutex;
	Mutex mRingMutex;
	
	uint32_t mLatestRGBA;
	uint32_t mLatestRingbuffer;
	bool     mLatestRetrieved;
	
	void*  mRGBA[NUM_RINGBUFFERS];
	bool   mRGBAZeroCopy; // were the RGBA buffers allocated with zeroCopy?
	bool   mStreaming;	  // true if the device is currently open
	int    mSensorCSI;	  // -1 for V4L2, >=0 for MIPI CSI

	inline bool csiCamera() const		{ return (mSensorCSI >= 0); }
};

#endif
