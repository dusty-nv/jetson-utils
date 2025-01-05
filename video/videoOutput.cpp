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

// CreateNullOutput
videoOutput* videoOutput::CreateNullOutput()
{
	videoOptions opt;
	opt.ioType = videoOptions::OUTPUT;

	videoOutput* output = new videoOutput(opt);
	
	if( !output )
		return NULL;

	output->mStreaming = true;
	LogWarning(LOG_VIDEO "no valid output streams, creating fake null output\n");
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
bool videoOutput::Render( void* image, uint32_t width, uint32_t height, imageFormat format, cudaStream_t stream )
{	
	const uint32_t numOutputs = mOutputs.size();
	bool result = true;

	for( uint32_t n=0; n < numOutputs; n++ )
	{
		if( !mOutputs[n]->Render(image, width, height, format, stream) )
			result = false;
	}

	return result;
}


// SetStatus
void videoOutput::SetStatus( const char* str )
{

}
