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
 
#ifndef __VIDEO_OUTPUT_H_
#define __VIDEO_OUTPUT_H_


#include "videoOptions.h"
#include "imageFormat.h"		
#include "commandLine.h"

#include <vector>


/**
 * Standard command-line options able to be passed to videoOutput::Create()
 * @ingroup video
 */
#define VIDEO_OUTPUT_USAGE_STRING  "videoOutput arguments: \n" 							\
		  "    output_URI           resource URI of the output stream, for example:\n"		\
		  "                             * file://my_image.jpg    (image file)\n"			\
		  "                             * file://my_video.mp4    (video file)\n"			\
		  "                             * file://my_directory/   (directory of images)\n"	\
		  "                             * rtp://<remote-ip>:1234 (RTP stream)\n"			\
		  "                             * display://0            (OpenGL window)\n" 		\
		  "  --output-codec=CODEC   desired codec for compressed output streams:\n"		\
		  "                            * h264 (default), h265\n"						\
		  "                            * vp8, vp9\n"									\
		  "                            * mpeg2, mpeg4\n"								\
		  "                            * mjpeg\n"        								\
		  "  --bitrate=BITRATE      desired target VBR bitrate for compressed streams,\n"    \
		  "                         in bits per second. The default is 4000000 (4 Mbps)\n"	\
		  "  --headless             don't create a default OpenGL GUI window\n\n"


/**
 * The videoOutput API is for rendering and transmitting frames to video input devices such as display windows, 
 * broadcasting RTP network streams to remote hosts over UDP/IP, and saving videos/images/directories to disk. 
 *
 * videoOutput interfaces are implemented by glDisplay, gstEncoder, and imageWriter.  
 * The specific implementation is selected at runtime based on the type of resource URI.
 * An instance can have multiple sub-streams, for example simultaneously outputting to 
 * a display and encoded video on disk or RTP stream.
 *
 * videoOutput supports the following protocols and resource URI's:
 *
 *     - `display://0` for rendering to display using OpenGL, where `0` corresponds to the display number.
 *        By default, an OpenGL window will be created, unless the `--headless` command line option is used.
 *
 *     - `rtp://<remote-host>:1234` to broadcast a compressed RTP stream to a remote host, where you should
 *        substitute `<remote-host>` with the remote host's IP address or hostname, and `1234` is the port.
 *
 *     - `file:///home/user/my_video.mp4` for saving videos, images, and directories of images to disk.
 *        You can leave off the `file://` protocol identifier and it will be deduced from the path.
 *        It can be a relative or absolute path.  You can output a sequence of images using a path of
 *        the form `my_dir/image_%i.jpg` (where `%i` can include printf-style modifiers like `%03i`).  
 *        The `%i` will be replaced with the image number in the sequence.  If just a directory is
 *        specified, then by default it will create a sequence of the form `%i.jpg` in that directory.
 *        Supported video formats for saving include MKV, MP4, AVI, and FLV. Supported codecs for 
 *        encoding include H.264, H.265, VP8, VP9, and MJPEG. Supported image formats for saving 
 *        include JPG, PNG, TGA, and BMP.
 *
 * @see URI for info about resource URI formats.
 * @see videoOptions for additional options and command-line arguments.
 * @ingroup video
 */
class videoOutput
{
public:
	/**
	 * Create videoOutput interface from a videoOptions struct that's already been filled out.
	 * It's expected that the supplied videoOptions already contain a valid resource URI.
	 */
	static videoOutput* Create( const videoOptions& options );

	/**
	 * Create videoOutput interface from a resource URI string and optional videoOptions.
	 * @see the documentation above and the URI struct for more info about resource URI's.
	 */
	static videoOutput* Create( const char* URI, const videoOptions& options=videoOptions() );

	/**
	 * Create videoOutput interface from a resource URI string and parsing command line arguments.
	 * @see videoOptions for valid command-line arguments to be parsed.
	 * @see the documentation above and the URI struct for more info about resource URI's.
	 */
	static videoOutput* Create( const char* URI, const commandLine& cmdLine );
	
	/**
	 * Create videoOutput interface from a resource URI string and parsing command line arguments.
	 * @see videoOptions for valid command-line arguments to be parsed.
	 * @see the documentation above and the URI struct for more info about resource URI's.
	 */
	static videoOutput* Create( const char* URI, const int argc, char** argv );

	/**
	 * Create videoOutput interface by parsing command line arguments, including the resource URI.
	 * @param positionArg indicates the positional argument number in the command line of
	 *                    the resource URI (or `-1` if a positional argument isn't used,
	 *                    and should instead be parsed from the `--input=` option). 
	 * @see videoOptions for valid command-line arguments to be parsed.
	 * @see the documentation above and the URI struct for more info about resource URI's.
	 */
	static videoOutput* Create( const int argc, char** argv, int positionArg=-1 );

	/**
	 * Create videoOutput interface by parsing command line arguments, including the resource URI.
	 * @param positionArg indicates the positional argument number in the command line of
	 *                    the resource URI (or `-1` if a positional argument isn't used,
	 *                    and should instead be parsed from the `--input=` option). 
	 * @see videoOptions for valid command-line arguments to be parsed.
	 * @see the documentation above and the URI struct for more info about resource URI's.
	 */
	static videoOutput* Create( const commandLine& cmdLine, int positionArg=-1 );
	
	/**
	 * Create videoOutput interface that acts as a NULL output and does nothing with incoming frames.
	 * CreateNullOutput() can be used when there are no other outputs created and programs expect one to run.
	 */
	static videoOutput* CreateNullOutput();
	
	/**
	 * Destroy interface and release all resources.
	 */
	virtual ~videoOutput();

	/**
	 * Usage string for command line arguments to Create()
	 */
	static inline const char* Usage() 		{ return VIDEO_OUTPUT_USAGE_STRING; }
	
	/**
	 * Render and output the next frame to the stream.
	 * The image formats supported by this templated version of Render() include the following:
	 *
	 *    - uchar3 (`IMAGE_RGB8`)
	 *    - uchar4 (`IMAGE_RGBA8`)
	 *    - float3 (`IMAGE_RGB32F`)
	 *    - float4 (`IMAGE_RGBA32F`)
	 *
	 * The image format will automatically be deduced from these types.  If other types are used
	 * with this overload, a static compile-time error will be asserted.
	 *
	 * @param image CUDA pointer containing the image to output.
	 * @param width width of the image, in pixels.
	 * @param height height of the image, in pixels.
	 * @returns `true` on success, `false` on error.
	 */
	template<typename T> bool Render( T* image, uint32_t width, uint32_t height )		{ return Render((void**)image, width, height, imageFormatFromType<T>()); }
	
	/**
	 * Render and output the next frame to the stream.
	 *
	 * The image formats supported by Render() are `IMAGE_RGB8` (uchar3), `IMAGE_RGBA8` (uchar4), 
	 * `IMAGE_RGB32F` (float3), and `IMAGE_RGBA32F` (float4). @see imageFormat for more info.
	 *
	 * @param image CUDA pointer containing the image to output.
	 * @param width width of the image, in pixels.
	 * @param height height of the image, in pixels.
	 * @param format format of the image (@see imageFormat)
	 * @returns `true` on success, `false` on error.
	 */
	virtual bool Render( void* image, uint32_t width, uint32_t height, imageFormat format );

	/**
	 * Begin streaming the device.
	 * After Open() is called, frames from the device can begin to be rendered.
	 *
	 * Open() is not stricly necessary to call, if you call one of the Render()
	 * functions they will first check to make sure that the stream is opened,
	 * and if not they will open it automatically for you.
	 *
	 * @returns `true` on success, `false` if an error occurred opening the stream.
	 */
	virtual bool Open();

	/**
	 * Stop streaming the device.
	 *
	 * @note Close() is automatically called by the videoOutput destructor when
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
	inline bool IsStreaming() const	   					{ return mStreaming; }

	/**
	 * Return the width of the stream, in pixels.
	 */
	inline uint32_t GetWidth() const						{ return mOptions.width; }

	/**
	 * Return the height of the stream, in pixels.
	 */
	inline uint32_t GetHeight() const						{ return mOptions.height; }
	
	/**
	 * Return the framerate, in Hz or FPS.
	 */
	inline float GetFrameRate() const						{ return mOptions.frameRate; }
	
	/**
	 * Return the resource URI of the stream.
	 */
	inline const URI& GetResource() const					{ return mOptions.resource; }

	/**
	 * Return the videoOptions of the stream.
	 */
	inline const videoOptions& GetOptions() const			{ return mOptions; }

	/**
	 * Add an output sub-stream.  When a frame is rendered
	 * to this stream, it will be rendered to each sub-stream.
	 */
	inline void AddOutput( videoOutput* output )				{ if(output != NULL) mOutputs.push_back(output); }

	/**
	 * Return the number of sub-streams.
	 */
	inline uint32_t GetNumOutputs( videoOutput* output ) const	{ mOutputs.size(); }

	/**
	 * Return a sub-stream.
	 */
	inline videoOutput* GetOutput( uint32_t index ) const		{ return mOutputs[index]; }

	/**
	 * Set a status string (i.e. status bar text on display window).
	 * Other types of interfaces may ignore the status text.
	 */
	virtual void SetStatus( const char* str );

	/**
	 * Return the interface type of the stream.
	 * This could be one of the following values:
	 *
	 *    - glDisplay::Type
	 *    - gstEncoder::Type
	 *    - imageWriter::Type
	 */
	virtual inline uint32_t GetType() const			{ return 0; }

	/**
	 * Check if this stream is of a particular type.
	 * @see GetType() for possible values.
	 */
	inline bool IsType( uint32_t type ) const		{ return (type == GetType()); }

	/**
	 * Check if a this stream is of a particular type.
	 * Can be used with glDisplay, gstEncoder, and imageWriter.  For example:  
	 *
	 *    if( stream->IsType<glDisplay>() )
	 *         glDisplay* display = (glDisplay*)stream;	// safe to cast
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
	videoOutput( const videoOptions& options );

	bool         mStreaming;
	videoOptions mOptions;

	std::vector<videoOutput*> mOutputs;
};

#endif
