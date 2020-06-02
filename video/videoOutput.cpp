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
 
#include "videoOutput.h"

#include "glDisplay.h"
//#include "gstEncoder.h"


// constructor
videoOutput::videoOutput( const videoOptions& options ) : mOptions(options)
{
	mStreaming = false;
}


// destructor
videoOutput::~videoOutput()
{
	const uint32_t numOutputs = mOutputs.size();

	for( uint32_t n=0; n < numOutputs; n++ )
		SAFE_DELETE(mOutputs[n]);
}


// Create
videoOutput* videoOutput::Create( const videoOptions& options )
{
	const URI& uri = options.resource;

	if( uri.protocol == "display" )
		return glDisplay::Create(options);
//	else if( uri.protocol == "file" || uri.protocol == "rtp" )
//		return gstEncoder::Create(options);

	printf("videoOutput -- unsupported protocol (%s)\n", uri.protocol.size() > 0 ? uri.protocol.c_str() : "null");
	return NULL;
}


// Create
videoOutput* videoOutput::Create( const char* resource, const videoOptions& options )
{
	videoOptions opt = options;
	opt.resource = resource;
	return Create(opt);
}


// Create
videoOutput* videoOutput::Create( const int argc, char** argv )
{
	if( argc < 0 || !argv )
		return NULL;

	commandLine cmdLine(argc, argv);
	return Create(cmdLine);
}


// Create
videoOutput* videoOutput::Create( const commandLine& cmdLine )
{
	videoOptions opt;

	if( !opt.Parse(cmdLine, videoOptions::OUTPUT) )
	{
		printf("videoOutput -- failed to parse command line options\n");
		return NULL;
	}

	return Create(opt);
}


// Open
bool videoOutput::Open()
{
	mStreaming = true;
	return true;
}


// Close
void videoOutput::Close()
{
	mStreaming = false;
}


// Render
bool videoOutput::Render( void* image, imageFormat format, uint32_t width, uint32_t height )
{	
	const uint32_t numOutputs = mOutputs.size();
	bool result = true;

	for( uint32_t n=0; n < numOutputs; n++ )
	{
		if( !mOutputs[n]->Render(image, format, width, height) )
			result = false;
	}

	return result;
}


// SetStatus
void videoOutput::SetStatus( const char* str )
{

}


