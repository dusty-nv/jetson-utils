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
#include "imageWriter.h"

#include "glDisplay.h"
#include "gstEncoder.h"


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
	videoOutput* output = NULL;
	const URI& uri = options.resource;
	
	if( uri.protocol == "file" )
	{
		if( gstEncoder::IsSupportedExtension(uri.extension.c_str()) )
			output = gstEncoder::Create(options);
		else
			output = imageWriter::Create(options);
	}
	else if( uri.protocol == "rtp" )
	{
		output = gstEncoder::Create(options);
	}
	else if( uri.protocol == "display" )
	{
		output = glDisplay::Create(options);
	}
	else
	{
		printf("videoOutput -- unsupported protocol (%s)\n", uri.protocol.size() > 0 ? uri.protocol.c_str() : "null");
	}

	if( !output )
		return NULL;

	output->GetOptions().Print(output->TypeToStr());
	return output;
}


// Create
videoOutput* videoOutput::Create( const char* resource, const videoOptions& options )
{
	videoOptions opt = options;
	opt.resource = resource;
	return Create(opt);
}


// Create
videoOutput* videoOutput::Create( const int argc, char** argv, int positionArg )
{
	if( argc < 0 || !argv )
		return NULL;

	commandLine cmdLine(argc, argv);
	return Create(cmdLine);
}


// Create
videoOutput* videoOutput::Create( const commandLine& cmdLine, int positionArg )
{
	videoOptions opt;

	if( !opt.Parse(cmdLine, videoOptions::OUTPUT, positionArg) )
	{
		printf("videoOutput -- failed to parse command line options\n");
		return NULL;
	}

	// create requested output interface
	videoOutput* output = Create(opt);

	// determine if display should also be created
	const bool headless = cmdLine.GetFlag("no-display") | cmdLine.GetFlag("headless");

	if( opt.resource.protocol != "display" && !headless )
	{
		opt.resource = "display://0";
		videoOutput* display = Create(opt);

		if( !display )
			return output;

		display->AddOutput(output);
		return display;
	}

	return output;
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
bool videoOutput::Render( void* image, uint32_t width, uint32_t height, imageFormat format )
{	
	const uint32_t numOutputs = mOutputs.size();
	bool result = true;

	for( uint32_t n=0; n < numOutputs; n++ )
	{
		if( !mOutputs[n]->Render(image, width, height, format) )
			result = false;
	}

	return result;
}


// SetStatus
void videoOutput::SetStatus( const char* str )
{

}


// TypeToStr
const char* videoOutput::TypeToStr( uint32_t type )
{
	if( type == glDisplay::Type )
		return "glDisplay";
	else if( type == gstEncoder::Type )
		return "gstEncoder";
	else if( type == imageWriter::Type )
		return "imageWriter";

	return "(unknown)";
}


