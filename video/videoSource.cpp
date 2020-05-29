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
	return NULL;
}


// Create
videoSource* videoSource::Create( const char* resource, const videoOptions& options )
{
	videoOptions opt = options;
	opt.resource = resource;
	return Create(opt);
}


// Create
videoSource* videoSource::Create( const int argc, char** argv )
{
	if( argc < 0 || !argv )
		return NULL;

	commandLine cmdLine(argc, argv);
	return Create(cmdLine);
}


// Create
videoSource* videoSource::Create( const commandLine& cmdLine )
{
	videoOptions opt;

	// TODO parse videoOptions

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


