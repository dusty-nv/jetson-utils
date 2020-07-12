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
 * Standard command-line options able to be passed to videoSource::Create()
 * @ingroup video
 */
#define VIDEO_SOURCE_USAGE_STRING  "videoSource arguments: \n" 								\
		  "    input_URI            resource URI of the input stream, for example:\n"			\
		  "                             * /dev/video0              (V4L2 camera #0)\n"			\
		  "                             * csi://0                  (MIPI CSI camera #0)\n"		\
		  "                             * rtp://@:1234             (RTP stream)\n"				\
		  "                             * rtsp://user:pass@ip:1234 (RTSP stream)\n"			\
		  "                             * file://my_image.jpg      (image file)\n"				\
		  "                             * file://my_video.mp4      (video file)\n"				\
		  "                             * file://my_directory/     (directory of images)\n"		\
		  "  --input-width=WIDTH    explicitly request a width of the stream (optional)\n"   	\
		  "  --input-height=HEIGHT  explicitly request a height of the stream (optional)\n"  	\
		  "  --input-rate=RATE      explicitly request a framerate of the stream (optional)\n"	\
		  "  --input-codec=CODEC    RTP requires the codec to be set, one of these:\n"			\
		  "                             * h264, h265\n"									\
		  "                             * vp8, vp9\n"									\
		  "                             * mpeg2, mpeg4\n"									\
		  "                             * mjpeg\n"        								\
		  "  --input-flip=FLIP      flip method to apply to input (excludes V4L2):\n" 			\
		  "                             * none (default)\n" 								\
		  "                             * counterclockwise\n" 								\
		  "                             * rotate-180\n" 									\
		  "                             * clockwise\n" 									\
		  "                             * horizontal\n" 									\
		  "                             * vertical\n" 									\
		  "                             * upper-right-diagonal\n" 							\
		  "                             * upper-left-diagonal\n" 							\
		  "  --input-loop=LOOP      for file-based inputs, the number of loops to run:\n"		\
		  "                             * -1 = loop forever\n"								\
		  "                             *  0 = don't loop (default)\n"						\
		  "                             * >0 = set number of loops\n\n"


/**
 * The videoSource API is for capturing frames from video input devices such as MIPI CSI cameras, 
 * V4L2 cameras, video/images files from disk, directories containing a sequence of images, 
 * and from RTP/RTSP network video streams over UDP/IP.
 *
 * videoSource interfaces are implemented by gstCamera, gstDecoder, and imageLoader.
 * The specific implementation is selected at runtime based on the type of resource URI.
 *
 * videoSource supports the following protocols and resource URI's:
 *
 *     -  `csi://0` for MIPI CSI cameras, where `0` can be replaced with the camera port.
 *        It is also assumed that if a number with no protocol is specified (e.g. `"0"`),
 *        this means to use the MIPI CSI camera of that number (`"0"` -> `csi://0`).
 *
 *     - `v4l2:///dev/video0` for V4L2 cameras, where `/dev/video0` can be replaced with
 *        a different video device (e.g. `v4l2:///dev/video1` for V4L2 video device `1`).
 *        If no protocol is specified but the string begins with `/dev/video`, then it is
 *        assumed that the protocol is V4L2 (`/dev/video0` -> `v4l2:///dev/video0`)
 *
 *     - `rtp://@:1234` to recieve an RTP network stream, where `1234` is the port and `@` is shorthand
 *        for localhost.  `@` can also be substituted for the IP address of a multicast group.
 *        Note that it is important to manually specify the codec of the stream when using RTP,
 *        as the codec cannot be discovered from the RTP stream itself and need to be provided.
 *        @see videoOptions for more info about the `--input-codec` option.
 *
 *     - `rtsp://username:password@<remote-host>:1234` to subscribe to an RTSP network stream, where
 *        `<remote-host>` should be substituted for the remote host's IP address or hostname, and 
 *        `1234` is the port.  For example, `rtsp://192.168.1.2:5000`.  The `username` and `password` 
 *        are optional, and are only used for RTSP streams that require authentication.
 *
 *     - `file:///home/user/my_video.mp4` for disk-based videos, images, and directories of images.
 *        You can leave off the `file://` protocol identifier and it will be deduced from the path.
 *        It can be a relative or absolute path.  If a directory is specified that contains images,
 *        those images will be loaded in sequence (sorted alphanumerically).  The path can also
 *        contain wildcard characters, for example `"images/*.jpg"` - however when using wildcards
 *        from the command line, enclose the string in quotes otherwise the OS will pre-expand them.
 *        Supported video formats for loading include MKV, MP4, AVI, and FLV. Supported codecs for 
 *        decoding include H.264, H.265, VP8, VP9, MPEG-2, MPEG-4, and MJPEG. Supported image formats
 *        for loading include JPG, PNG, TGA, BMP, GIF, PSD, HDR, PIC, and PNM (PPM/PGM binary).
 *  
 * @see URI for info about resource URI formats.
 * @see videoOptions for additional options and command-line arguments.
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
	 * Usage string for command line arguments to Create()
	 */
	static inline const char* Usage() 		{ return VIDEO_SOURCE_USAGE_STRING; }
	
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
	 * `IMAGE_RGB32F` (float3), and `IMAGE_RGBA32F` (float4). @see imageFormat for more info.
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
	 * Check if a this stream is of a particular type.
	 * Can be used with gstCamera, gstDecoder, and imageLoader.  For example:  
	 *
	 *    if( stream->IsType<gstCamera>() )
	 *         gstCamera* camera = (gstCamera*)stream;	// safe to cast
	 */
	template<typename T> bool IsType() const		{ return IsType(T::Type); }

	/**
	 * Convert this stream's class type to string.
	 */
	inline const char* TypeToStr() const			{ return TypeToStr(GetType()); }

	/**
	 * Convert a class type to a string.
	 */
	static const char* TypeToStr( uint32_t type );

protected:
	//videoSource();
	videoSource( const videoOptions& options );

	bool         mStreaming;
	videoOptions mOptions;
};

#endif
