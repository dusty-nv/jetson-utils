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
 
#include "imageLoader.h"
#include "imageIO.h"

#include "filesystem.h"
#include "logging.h"

#include <strings.h>


// supported image file extensions
const char* imageLoader::SupportedExtensions[] = { "jpg", "jpeg", "png", 
										 "tga", "targa", "bmp", 
										 "gif", "psd", "hdr",
										 "pic", "pnm", "pbm",
										 "ppm", "pgm", NULL };

bool imageLoader::IsSupportedExtension( const char* ext )
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
imageLoader::imageLoader( const videoOptions& options ) : videoSource(options)
{
	mEOS = false;
	mNextFile = 0;

	mBuffers.reserve(options.numBuffers);
	mOptions.deviceType = videoOptions::DEVICE_FILE;

	// list files to use
	std::vector<std::string> files;

	if( !listDir(options.resource.location, files, FILE_REGULAR) )
	{
		LogError(LOG_IMAGE "imageLoader -- failed to find '%s'\n", options.resource.location.c_str());
		return;
	}

	// check extensions for image types
	const size_t numFiles = files.size();

	for( size_t n=0; n < numFiles; n++ )
	{
		if( fileHasExtension(files[n], SupportedExtensions) )
		{
			LogDebug(LOG_IMAGE "imageLoader -- found file %s\n", files[n].c_str());
			mFiles.push_back(files[n]);
		}
	} 

	if( mFiles.size() == 0 )
	{
		LogError(LOG_IMAGE "imageLoader -- failed to find any image files under '%s'\n", options.resource.location.c_str());
		return;
	}
}


// destructor
imageLoader::~imageLoader()
{
	const size_t numBuffers = mBuffers.size();

	for( size_t n=0; n < numBuffers; n++ )
		CUDA(cudaFreeHost(mBuffers[n]));

	mBuffers.clear();
}


// Create
imageLoader* imageLoader::Create( const videoOptions& options )
{
	imageLoader* loader = new imageLoader(options);

	if( loader->mFiles.size() == 0 )
	{
		delete loader;
		return NULL;
	}

	return loader;
}


// Create
imageLoader* imageLoader::Create( const char* resource, const videoOptions& options )
{
	videoOptions opt = options;
	opt.resource = resource;
	return Create(opt);
}


// Capture
bool imageLoader::Capture( void** output, imageFormat format, uint64_t timeout )
{
	// verify the output pointer exists
	if( !output )
		return false;

	// confirm the stream is open
	if( !mStreaming )
	{
		if( !Open() )
			return false;
	}

	// reclaim old buffers
	if( mBuffers.size() >= mOptions.numBuffers )
	{
		CUDA(cudaFreeHost(mBuffers[0]));
		mBuffers.erase(mBuffers.begin());
	}

	// get the next file to load
	const size_t currFile = mNextFile;
	mNextFile++;
	
	if( mNextFile >= mFiles.size() )
	{
		if( isLooping() )
		{
			mNextFile = 0;
			mLoopCount++;
		}
		else
		{
			mEOS = true;
			mStreaming = false;
		}
	}

	// load the next image
	void* imgPtr  = NULL;
	int imgWidth  = 0;
	int imgHeight = 0;

	if( !loadImage(mFiles[currFile].c_str(), &imgPtr, &imgWidth, &imgHeight, format) )
	{
		LogError(LOG_IMAGE "imageLoader -- failed to load '%s'\n", mFiles[currFile].c_str());
		return Capture(output, format, timeout);
	}

	// set outputs
	mOptions.width = imgWidth;
	mOptions.height = imgHeight;

	*output = imgPtr;
	mBuffers.push_back(imgPtr);

	return true;
}


// Open
bool imageLoader::Open()
{
	if( mEOS )
	{
		LogWarning(LOG_IMAGE "imageLoader -- End of Stream (EOS) has been reached, stream has been closed\n");
		return false;
	}

	mStreaming = true;
	return true;
}


// Close
void imageLoader::Close()
{
	mStreaming = false;
}


