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

#include "Event.h"
#include "RingBuffer.h"

#include "videoSource.h"


// Forward declarations
struct _GstAppSink;


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
 * @note gstCamera now implements the videoSource interface and is intended to be used through 
 * that as opposed to directly. videoSource implements additional command-line parsing of 
 * videoOptions to construct instances. Some legacy APIs of gstCamera are now marked deprecated.
 *
 * @see videoSource
 * @ingroup camera
 */
class gstCamera : public videoSource
{
public:
	/**
	 * Create a MIPI CSI or V4L2 camera device.
	 */
	static gstCamera* Create( const videoOptions& options );

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
	virtual bool Open();

	/**
	 * Stop streaming the camera.
	 * @note Close() is automatically called by the camera's destructor when
	 * it gets deleted, so you do not explicitly need to call Close() before
	 * exiting the program if you delete your camera object.
	 */
	virtual void Close();

	/**
	 * Capture the next image frame from the camera.
	 * @see videoSource::Capture
	 */
	template<typename T> bool Capture( T** image, uint64_t timeout=UINT64_MAX )		{ return Capture((void**)image, imageFormatFromType<T>(), timeout); }
	
	/**
	 * Capture the next image frame from the camera.
	 * @see videoSource::Capture
	 */
	virtual bool Capture( void** image, imageFormat format, uint64_t timeout=UINT64_MAX );

	/**
	 * Capture the next image frame from the camera and convert it to float4 RGBA format,
	 * with pixel intensities ranging between 0.0 and 255.0.
	 *
	 * @deprecated CaptureRGBA() has been deprecated and is only provided for legacy 
	 *             compatibility. Please use the updated Capture() function instead.
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
	 * Return the interface type (gstCamera::Type)
	 */
	virtual inline uint32_t GetType() const		{ return Type; }

	/**
	 * Unique type identifier of gstCamera class.
	 */
	static const uint32_t Type = (1 << 0);

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

	gstCamera( const videoOptions& options );

	bool init();
	bool discover();
	bool buildLaunchStr();

	void checkMsgBus();
	void checkBuffer();
	
	bool matchCaps( GstCaps* caps );
	bool printCaps( GstCaps* caps );
	bool parseCaps( GstStructure* caps, videoOptions::Codec* codec, imageFormat* format, uint32_t* width, uint32_t* height, float* frameRate );
	
	_GstBus*     mBus;
	_GstAppSink* mAppSink;
	_GstElement* mPipeline;

	std::string  mLaunchStr;

	imageFormat mFormatYUV;
	size_t      mFrameCount;
	
	RingBuffer mBufferYUV;
	RingBuffer mBufferRGB;

	Event mWaitEvent;
};

#endif
