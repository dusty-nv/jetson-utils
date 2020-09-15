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
 
#include "imageWriter.h"
#include "imageIO.h"

#include "filesystem.h"
#include "logging.h"

#include <strings.h>


// supported image file extensions
const char* imageWriter::SupportedExtensions[] = { "jpg", "jpeg", "png", 
										 "tga", "targa", "bmp", 
										 NULL };

bool imageWriter::IsSupportedExtension( const char* ext )
{
	if( !ext )
		return false;

	uint32_t extCount = 0;

	while(true)
	{
		if( !SupportedExtensions[extCount] )
			break;

		if( strcasecmp(SupportedExtensions[extCount], ext) == 0 )
			return true;

		extCount++;
	}

	return false;
}

// constructor
imageWriter::imageWriter( const videoOptions& options ) : videoOutput(options)
{
	mFileCount = 0;
	mStreaming = true;

	mOptions.deviceType = videoOptions::DEVICE_FILE;
}


// destructor
imageWriter::~imageWriter()
{

}


// Create
imageWriter* imageWriter::Create( const videoOptions& options )
{
	return new imageWriter(options);
}


// Create
imageWriter* imageWriter::Create( const char* resource, const videoOptions& options )
{
	videoOptions opt = options;
	opt.resource = resource;
	return Create(opt);
}


// Render
bool imageWriter::Render( void* image, uint32_t width, uint32_t height, imageFormat format )
{
	const bool substreams_success = videoOutput::Render(image, width, height, format);

	if( mOptions.resource.location.find("%") != std::string::npos )
	{
		// path has a format (should be '%u' or '%i')
		sprintf(mFileOut, mOptions.resource.location.c_str(), mFileCount);
	}
	else if( mOptions.resource.extension.size() == 0 )
	{
		// path is a dir, use default image numbering
		sprintf(mFileOut, "%u.jpg", mFileCount);
		const std::string path = pathJoin(mOptions.resource.location, mFileOut);
		strcpy(mFileOut, path.c_str());
	}
	else
	{
		// path is a single file, use it as-is
		strcpy(mFileOut, mOptions.resource.location.c_str());
	}

	CUDA(cudaDeviceSynchronize());
	
	// save the image
	if( !saveImage(mFileOut, image, width, height, format) )
	{
		LogError(LOG_IMAGE "imageWriter -- failed to save '%s'\n", mFileOut);
		return false;
	}

	mOptions.width  = width;
	mOptions.height = height;

	mFileCount++;

	return substreams_success;
}

