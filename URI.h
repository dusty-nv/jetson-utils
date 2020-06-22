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
 
#ifndef __URI_RESOURCE_H_
#define __URI_RESOURCE_H_


#include <string> 


/**
 * Resource URI of a video device, IP stream, or file/directory.
 *
 * The URI object is used by videoSource, videoOutput, and videoOptions
 * to identify which resource is being streamed.  It will parse a string
 * provided by the user into protocol, path, and port/extension components.
 *
 * videoSource input streams support protocols and URI strings for 
 * MIPI CSI cameras, V4L2 cameras, disk-based videos/images/directories,
 * and RTP/RTSP networking streams over UDP/IP.
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
 * videoOutput output streams support protocols and URI strings
 * for rendering to displays, saving to videos/images/directories on disk,
 * and broadcasting RTP streams to a remote host over UDP/IP:
 *
 *     - `display://0` for rendering to display using OpenGL, where `0` corresponds to the display number.
 *        By default, an OpenGL display will be created, unless the `--headless` command line option is used.
 *
 *     - `file:///home/user/my_video.mp4` for saving videos, images, and directories of images to disk.
 *        You can leave off the `file://` protocol identifier and it will be deduced from the path.
 *        It can be a relative or absolute path.  You can output a sequence of images using a path of
 *        the form `my_dir/image_%i.jpg` (where `%i` can include printf-style modifiers like `%03i`).  
 *        The `%i` will be replaced with the image number in the sequence.  If just a directory is
 *        specified, then by default it will create a sequence of the form `%i.jpg` in that directory.
 *
 *     - `rtp://192.168.1.1:1234` to broadcast a compressed RTP stream to a remote host, where you should
 *        substitute `192.168.1.1` with the remote host's IP address or hostname, and `1234` is the port.
 *
 * The URI strings used should take one of the above forms for input/output streams to be parsed correctly.
 * @ingroup video
 */
struct URI
{
public:
	/**
	 * Default constructor.
	 */
	URI();

	/**
	 * Construct a new URI from the given resource string.
	 * @see the documentation above for valid string formats.
	 */
	URI( const char* uri );

	/**
	 * Parse the URI from the given resource string.
	 * @see the documentation above for valid string formats.
	 */
	bool Parse( const char* uri );

	/**
	 * Log the URI, with an optional prefix label.
	 */
	void Print( const char* prefix="" ) const;

	/**
	 * Cast to C-style string (`const char*`)
	 */
	inline const char* c_str() const				{ return string.c_str(); }

	/**
	 * Cast to C-style string (`const char*`)
	 */
	operator const char* () const					{ return string.c_str(); }

	/**
	 * Cast to `std::string`
	 */
	operator std::string () const					{ return string; }

	/**
	 * Assignment operator (parse URI string)
	 */
	inline void operator = (const char* uri ) 		{ Parse(uri); }

	/**
	 * Assignment operator (parse URI string)
	 */
	inline void operator = (const std::string& uri ) 	{ Parse(uri.c_str()); }

	/**
	 * Full resource URI (what was originally parsed)
	 */
	std::string string;

	/**
	 * Protocol string (e.g. `file`, `csi`, `v4l2`, `rtp`, ect)
	 */
	std::string protocol;

	/**
	 * Path, IP address, or device name
	 */
	std::string path;

	/**
	 * File extension (for files only, otherwise empty)
	 */
	std::string extension;

	/**
	 * IP port, camera port, ect.
	 */
	int port;
};

#endif

