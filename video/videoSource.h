/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
 
#ifndef __VIDEO_SOURCE_H_
#define __VIDEO_SOURCE_H_


#include "videoOptions.h"
#include "imageFormat.h"		
#include "commandLine.h"


/**
 * The videoSource API is for capturing frames from video input devices
 * such as MIPI CSI cameras, V4L2 cameras, video/images files from disk,
 * directories containing a sequence of images, and from RTP/RTSP network
 * video streams over UDP/IP.
 *
 * videoSource supports the following protocols and resource URI's:
 *
 *     -  `csi://0` for MIPI CSI cameras, where `0` can be replaced with the camera port.
 *        It is also assumed that if a number with no protocol is specified (e.g. `"0"`),
 *        this means the MIPI CSI camera of that number (`"0"` ~= `csi://0`)
 *
 *     - `v4l2:///dev/video0` for V4L2 cameras, where `/dev/video0` can be replaced with
 *        a different video device (e.g. `v4l2:///dev/video1` for V4L2 video device `1`).
 *        If no protocol is specified but the string begins with `/dev/video`, then it is
 *        assumed that the protocol is V4L2 (`/dev/video0` ~= `v4l2:///dev/video0`)
 *
 *     - `file:///home/user/my_video.mp4` for disk-based videos, images, and directories of images.
 *        You can leave off the `file://` protocol identifier and it will be deduced from the path.
 *        It can be a relative or absolute path.  If a directory is specified that contains images,
 *        those images will be loaded in sequence (sorted alphanumerically).  The path can also
 *        contain wildcard characters, for example `"images/*.jpg"` - however when using wildcards
 *        from the command line, enclose the string in quotes otherwise the OS will pre-expand them.
 *
 *     - `rtp://@:1234` to recieve an RTP network stream, where `1234` is the port and `@` is shorthand
 *        for localhost.  `@` can also be substituted for the IP address of a multicast group.
 *        Note that it is important to manually specify the codec and width/height when using RTP,
 *        as these values cannot be discovered from the RTP stream itself and need to be provided.
 *        @see videoOptions for more info about `--input-codec`, `--input-width`, and `--input-height`.
 *
 *     - `rtsp://192.168.1.1:1234` to subscribe to an RTSP network stream, where `192.168.1.1` should
 *        be substituted for the remote host's IP address or hostname, and `1234` is the port.
 *  
 * @see URI for more info about resource URI formats.
 * @ingroup video
 */
class videoSource
{
public:
	/**
	 * Create videoSource interface from a videoOptions struct that's already been filled out.
	 * It's expected that the supplied videoOptions already contain a valid resource URI.
	 */
	static videoSource* Create( const videoOptions& options );

	/**
	 * Create videoSource interface from a resource URI string and optional videoOptions.
	 * @see the documentation above and the URI struct for more info about resource URI's.
	 */
	static videoSource* Create( const char* URI, const videoOptions& options=videoOptions() );

	/**
	 * Create videoSource interface from a resource URI string and parsing command line arguments.
	 * @see videoOptions for valid command-line arguments to be parsed.
	 * @see the documentation above and the URI struct for more info about resource URI's.
	 */
	static videoSource* Create( const char* URI, const commandLine& cmdLine );
	
	/**
	 * Create videoSource interface from a resource URI string and parsing command line arguments.
	 * @see videoOptions for valid command-line arguments to be parsed.
	 * @see the documentation above and the URI struct for more info about resource URI's.
	 */
	static videoSource* Create( const char* URI, const int argc, char** argv );

	/**
	 * Create videoSource interface by parsing command line arguments, including the resource URI.
	 * @param positionArg indicates the positional argument number in the command line of
	 *                    the resource URI (or `-1` if a positional argument isn't used,
	 *                    and should instead be parsed from the `--input=` option). 
	 * @see videoOptions for valid command-line arguments to be parsed.
	 * @see the documentation above and the URI struct for more info about resource URI's.
	 */
	static videoSource* Create( const int argc, char** argv, int positionArg=-1 );

	/**
	 * Create videoSource interface by parsing command line arguments, including the resource URI.
	 * @param positionArg indicates the positional argument number in the command line of
	 *                    the resource URI (or `-1` if a positional argument isn't used,
	 *                    and should instead be parsed from the `--input=` option). 
	 * @see videoOptions for valid command-line arguments to be parsed.
	 * @see the documentation above and the URI struct for more info about resource URI's.
	 */
	static videoSource* Create( const commandLine& cmdLine, int positionArg=-1 );
	
	/**
	 * Destroy interface and release all resources.
	 */
	virtual ~videoSource();

	/**
	 * Capture the next image from the video stream.
	 *
	 * The image formats supported by this templated version of Capture() include the following:
	 *
	 *    - uchar3 (`IMAGE_RGB8`)
	 *    - uchar4 (`IMAGE_RGBA8`)
	 *    - float3 (`IMAGE_RGB32F`)
	 *    - float4 (`IMAGE_RGBA32F`)
	 *
	 * The image format will automatically be deduced from these types.  If other types are used
	 * with this overload, a static compile-time error will be asserted.
	 *
	 * @param[out] image output pointer that will be set to the memory containing the image.
 	 *                   If this interface has it's videoOptions::zeroCopy flag set to true,
	 *                   the memory was allocated in mapped CPU/GPU memory and is be accessible
	 *                   from both CPU and CUDA.  Otherwise, it's accessible only from CUDA.
	 *
	 * @param[in] timeout timeout in milliseconds to wait to capture the image before returning.
	 *                    A timeout value of `UINT64_MAX` (the default) will wait forever, and
	 *                    a timeout of 0 will return instantly if a frame wasn't immediately ready.
	 *
	 * @returns `true` if a frame was captured, `false` if there was an error or a timeout occurred.
	 */
	template<typename T> bool Capture( T** image, uint64_t timeout=UINT64_MAX )		{ return Capture((void**)image, imageFormatFromType<T>(), timeout); }
	
	/**
	 * Capture the next image from the video stream.
	 *
	 * The image formats supported by Capture() are `IMAGE_RGB8` (uchar3), `IMAGE_RGBA8` (uchar4), 
	 * `IMAGE_RGB32F` (float3), and `IMAGE_RGBA32F` (float4). @see imageFormats for more info.
	 *
	 * @param[out] image output pointer that will be set to the memory containing the image.
 	 *                   If this interface has it's videoOptions::zeroCopy flag set to true,
	 *                   the memory was allocated in mapped CPU/GPU memory and is be accessible
	 *                   from both CPU and CUDA.  Otherwise, it's accessible only from CUDA.
	 *
	 * @param[in] timeout timeout in milliseconds to wait to capture the image before returning.
	 *                    A timeout value of `UINT64_MAX` (the default) will wait forever, and
	 *                    a timeout of 0 will return instantly if a frame wasn't immediately ready.
	 *
	 * @returns `true` if a frame was captured, `false` if there was an error or a timeout occurred.
	 */
	virtual bool Capture( void** image, imageFormat format, uint64_t timeout=UINT64_MAX ) = 0;

	/**
	 * Begin streaming the device.
	 * After Open() is called, frames from the device will begin to be captured.
	 *
	 * Open() is not stricly necessary to call, if you call one of the Capture()
	 * functions they will first check to make sure that the stream is opened,
	 * and if not they will open it automatically for you.
	 *
	 * @returns `true` on success, `false` if an error occurred opening the stream.
	 */
	virtual bool Open();

	/**
	 * Stop streaming the device.
	 *
	 * @note Close() is automatically called by the videoSource destructor when
	 * it gets deleted, so you do not explicitly need to call Close() before
	 * exiting the program if you delete your videoSource object.
	 */
	virtual void Close();

	/**
	 * Check if the device is actively streaming or not.
	 *
	 * @returns `true` if the device is streaming (open), or `false` if it's closed
 	 *          or has reached EOS (End Of Stream).
	 */
	inline bool IsStreaming() const	   			{ return mStreaming; }

	/**
	 * Return the width of the stream, in pixels.
	 */
	inline uint32_t GetWidth() const				{ return mOptions.width; }

	/**
	 * Return the height of the stream, in pixels.
	 */
	inline uint32_t GetHeight() const				{ return mOptions.height; }
	
	/**
	 * Return the framerate, in Hz or FPS.
	 */
	inline uint32_t GetFrameRate() const			{ return mOptions.frameRate; }

	/**
	 * Return the resource URI of the stream.
	 */
	inline const URI& GetResource() const			{ return mOptions.resource; }

	/**
	 * Return the videoOptions of the stream.
	 */
	inline const videoOptions& GetOptions() const	{ return mOptions; }

	/**
	 * Return the interface type of the stream.
	 * This could be one of the following values:
	 *
	 *    - gstCamera::Type
	 *    - gstDecoder::Type
	 *    - imageLoader::Type
	 */
	virtual inline uint32_t GetType() const			{ return 0; }

	/**
	 * Check if this stream is of a particular type.
	 * @see GetType() for possible values.
	 */
	inline bool IsType( uint32_t type ) const		{ return (type == GetType()); }

	/**
	 *
	 */
	template<typename T> bool IsType() const		{ return IsType(T::Type); }

	/**
	 *
	 */
	inline const char* TypeToStr() const			{ return TypeToStr(GetType()); }

	/**
	 *
	 */
	static const char* TypeToStr( uint32_t type );

protected:
	//videoSource();
	videoSource( const videoOptions& options );

	bool         mStreaming;
	videoOptions mOptions;
};

#endif
