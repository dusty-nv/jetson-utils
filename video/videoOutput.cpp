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

#include "logging.h"


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


// create secondary display stream (if needed)
static videoOutput* createDisplaySubstream( videoOutput* output, videoOptions& options, const commandLine& cmdLine )
{
	const bool headless = cmdLine.GetFlag("no-display") | cmdLine.GetFlag("headless");

	if( options.resource.protocol != "display" && !headless )
	{
		options.resource = "display://0";
		videoOutput* display = videoOutput::Create(options);

		if( !display )
			return output;

		display->AddOutput(output);
		return display;
	}

	return output;
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
	else if( uri.protocol == "rtp" || uri.protocol == "rtmp" )
	{
		output = gstEncoder::Create(options);
	}
	else if( uri.protocol == "display" )
	{
		output = glDisplay::Create(options);
	}
	else
	{
		LogError(LOG_VIDEO "videoOutput -- unsupported protocol (%s)\n", uri.protocol.size() > 0 ? uri.protocol.c_str() : "null");
	}

	if( !output )
		return NULL;

	LogSuccess(LOG_VIDEO "created %s from %s\n", output->TypeToStr(), output->GetResource().string.c_str());
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
videoOutput* videoOutput::Create( const char* resource, const commandLine& cmdLine )
{
	videoOptions opt;

	if( !opt.Parse(resource, cmdLine, videoOptions::OUTPUT) )
	{
		LogError(LOG_VIDEO "videoOutput -- failed to parse command line options\n");
		return NULL;
	}

	return createDisplaySubstream(Create(opt), opt, cmdLine);
}

// Create
videoOutput* videoOutput::Create( const char* resource, const int argc, char** argv )
{
	commandLine cmdLine(argc, argv);
	return Create(resource, cmdLine);
}

// Create
videoOutput* videoOutput::Create( const commandLine& cmdLine, int positionArg )
{
	videoOptions opt;

	if( !opt.Parse(cmdLine, videoOptions::OUTPUT, positionArg) )
	{
		LogError(LOG_VIDEO "videoOutput -- failed to parse command line options\n");
		return NULL;
	}

	return createDisplaySubstream(Create(opt), opt, cmdLine);
}

// Create
videoOutput* videoOutput::Create( const int argc, char** argv, int positionArg )
{
	commandLine cmdLine(argc, argv);
	return Create(cmdLine);
}

// CreateNullOutput
videoOutput* videoOutput::CreateNullOutput()
{
	videoOptions opt;
	opt.ioType = videoOptions::OUTPUT;

	videoOutput* output = new videoOutput(opt);
	
	if( !output )
		return NULL;

	output->mStreaming = true;
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

	LogWarning(LOG_VIDEO "unknown videoOutput type - %u\n", type);
	return "(unknown)";
}


