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
// #include "gstDecoder.h"

#include "logging.h"

#include <cassert>
#include <cstdint>
#include <string>
#include <algorithm>

// constructor
videoSource::videoSource( const videoOptions& options ) : mOptions(options)
{
	mStreaming = false;
	mLastTimestamp = 0;
	mRawFormat = IMAGE_UNKNOWN;
}


// destructor
videoSource::~videoSource()
{

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
const char* videoSource::TypeToStr( uint32_t type ) const
{
	if( type == gstCamera::Type )
		return "gstCamera";
	// else if( type == gstDecoder::Type )
	// 	return "gstDecoder";
	// else if( type == imageLoader::Type )
	// 	return "imageLoader";
	assert(false);
	return "(unknown)";
}


