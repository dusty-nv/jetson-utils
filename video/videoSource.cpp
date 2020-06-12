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
 
#include "videoSource.h"
#include "imageLoader.h"

#include "gstCamera.h"
#include "gstDecoder.h"

#include "logging.h"


// constructor
videoSource::videoSource( const videoOptions& options ) : mOptions(options)
{
	mStreaming = false;
}


// destructor
videoSource::~videoSource()
{

}

// Create
videoSource* videoSource::Create( const videoOptions& options )
{
	videoSource* src = NULL;
	const URI& uri = options.resource;

	if( uri.protocol == "file" )
	{
		if( gstDecoder::IsSupportedExtension(uri.extension.c_str()) )
			src = gstDecoder::Create(options);
		else
			src = imageLoader::Create(options);
	}
	else if( uri.protocol == "rtp" || uri.protocol == "rtsp" )
	{
		src = gstDecoder::Create(options);
	}
	else if( uri.protocol == "csi" || uri.protocol == "v4l2" )
	{
		src = gstCamera::Create(options);
	}
	else
	{
		LogError(LOG_VIDEO "videoSource -- unsupported protocol (%s)\n", uri.protocol.size() > 0 ? uri.protocol.c_str() : "null");
	}

	if( !src )
		return NULL;

	LogSuccess(LOG_VIDEO "created %s from %s\n", src->TypeToStr(), src->GetResource().string.c_str());
	src->GetOptions().Print(src->TypeToStr());
	return src;
}

// Create
videoSource* videoSource::Create( const char* resource, const videoOptions& options )
{
	videoOptions opt = options;
	opt.resource = resource;
	return Create(opt);
}

// Create
videoSource* videoSource::Create( const char* resource, const int argc, char** argv )
{
	commandLine cmdLine(argc, argv);
	return Create(resource, cmdLine);
}

// Create
videoSource* videoSource::Create( const char* resource, const commandLine& cmdLine )
{
	videoOptions opt;

	if( !opt.Parse(resource, cmdLine, videoOptions::INPUT) )
	{
		LogError(LOG_VIDEO "videoSource -- failed to parse command line options\n");
		return NULL;
	}

	return Create(opt);
}

// Create
videoSource* videoSource::Create( const int argc, char** argv, int positionArg )
{
	/*if( argc < 0 || !argv )
		return NULL;*/

	commandLine cmdLine(argc, argv);
	return Create(cmdLine, positionArg);
}

// Create
videoSource* videoSource::Create( const commandLine& cmdLine, int positionArg )
{
	videoOptions opt;

	if( !opt.Parse(cmdLine, videoOptions::INPUT, positionArg) )
	{
		LogError(LOG_VIDEO "videoSource -- failed to parse command line options\n");
		return NULL;
	}

	return Create(opt);
}

// Open
bool videoSource::Open()
{
	mStreaming = true;
	return true;
}

// Close
void videoSource::Close()
{
	mStreaming = false;
}

// TypeToStr
const char* videoSource::TypeToStr( uint32_t type )
{
	if( type == gstCamera::Type )
		return "gstCamera";
	else if( type == gstDecoder::Type )
		return "gstDecoder";
	else if( type == imageLoader::Type )
		return "imageLoader";

	return "(unknown)";
}


