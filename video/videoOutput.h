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
 * @ingroup video
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
	static videoOutput* Create( const char* URI, const videoOptions& options=videoOptions() );

	/**
	 *
	 */
	static videoOutput* Create( const char* URI, const commandLine& cmdLine );
	
	/**
	 *
	 */
	static videoOutput* Create( const char* URI, const int argc, char** argv );

	/**
	 *
	 */
	static videoOutput* Create( const int argc, char** argv, int positionArg=-1 );

	/**
	 *
	 */
	static videoOutput* Create( const commandLine& cmdLine, int positionArg=-1 );
	
	/**
	 *
	 */
	static videoOutput* CreateNullOutput();
	
	/**
	 *
	 */
	virtual ~videoOutput();

	/**
	 *
	 */
	template<typename T> bool Render( T* image, uint32_t width, uint32_t height )		{ return Render((void**)image, width, height, imageFormatFromType<T>()); }
	
	/**
	 *
	 */
	virtual bool Render( void* image, uint32_t width, uint32_t height, imageFormat format );

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
	inline float GetFrameRate() const						{ return mOptions.frameRate; }
	
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
	inline void AddOutput( videoOutput* output )				{ if(output != NULL) mOutputs.push_back(output); }

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

	/**
	 *
	 */
	virtual inline uint32_t GetType() const			{ return 0; }

	/**
	 *
	 */
	inline bool IsType( uint32_t type ) const		{ return (type == GetType()); }

	/**
	 *
	 */
	template<typename T> bool IsType() const		{ return IsType(T::Type); }

	/**
	 *
	 */
	inline const char* TypeToStr() const			{ return TypeToStr(GetType()); }

	/**
	 *
	 */
	static const char* TypeToStr( uint32_t type );

protected:
	videoOutput( const videoOptions& options );

	bool         mStreaming;
	videoOptions mOptions;

	std::vector<videoOutput*> mOutputs;
};

#endif
