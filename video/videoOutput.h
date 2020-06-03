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
 
#ifndef __VIDEO_OUTPUT_H_
#define __VIDEO_OUTPUT_H_


#include "videoOptions.h"
#include "imageFormat.h"		
#include "commandLine.h"

#include <vector>


/**
 * @ingroup image
 */
class videoOutput
{
public:
	/**
	 *
	 */
	static videoOutput* Create( const videoOptions& options );

	/**
	 *
	 */
	static videoOutput* Create( const char* resource, const videoOptions& options=videoOptions() );

	/**
	 *
	 */
	static videoOutput* Create( const int argc, char** argv );

	/**
	 *
	 */
	static videoOutput* Create( const commandLine& cmdLine );
	
	/**
	 *
	 */
	virtual ~videoOutput();

	/**
	 *
	 */
	virtual bool Render( void* image, imageFormat format, uint32_t width, uint32_t height );

	/**
	 *
	 */
	template<typename T> inline bool Render( T* image, uint32_t width, uint32_t height )		{ return Render((void**)image, imageFormatFromType<T>(), width, height); }
	
	/**
	 * 
	 */
	virtual bool Open();

	/**
	 * 
	 */
	virtual void Close();

	/**
	 * 
	 */
	inline bool IsStreaming() const	   					{ return mStreaming; }

	/**
	 *
	 */
	inline uint32_t GetWidth() const						{ return mOptions.width; }

	/**
	 *
	 */
	inline uint32_t GetHeight() const						{ return mOptions.height; }
	
	/**
	 *
	 */
	inline uint32_t GetFrameRate() const					{ return mOptions.frameRate; }
	
	/**
	 *
	 */
	inline const URI& GetResource() const					{ return mOptions.resource; }

	/**
	 *
	 */
	inline const videoOptions& GetOptions() const			{ return mOptions; }

	/**
	 *
	 */
	inline void AddOutput( videoOutput* output )				{ mOutputs.push_back(output); }

	/**
	 *
	 */
	inline uint32_t GetNumOutputs( videoOutput* output ) const	{ mOutputs.size(); }

	/**
	 *
	 */
	inline videoOutput* GetOutput( uint32_t index ) const		{ return mOutputs[index]; }

	/**
	 *
	 */
	virtual void SetStatus( const char* str );

protected:
	videoOutput( const videoOptions& options );

	bool         mStreaming;
	videoOptions mOptions;

	std::vector<videoOutput*> mOutputs;
};

#endif