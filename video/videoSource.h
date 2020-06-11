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
 
#ifndef __VIDEO_SOURCE_H_
#define __VIDEO_SOURCE_H_


#include "videoOptions.h"
#include "imageFormat.h"		
#include "commandLine.h"


/**
 * @ingroup video
 */
class videoSource
{
public:
	/**
	 *
	 */
	static videoSource* Create( const videoOptions& options );

	/**
	 *
	 */
	static videoSource* Create( const char* URI, const videoOptions& options=videoOptions() );

	/**
	 *
	 */
	static videoSource* Create( const char* URI, const commandLine& cmdLine );
	
	/**
	 *
	 */
	static videoSource* Create( const char* URI, const int argc, char** argv );

	/**
	 *
	 */
	static videoSource* Create( const int argc, char** argv, int positionArg=-1 );

	/**
	 *
	 */
	static videoSource* Create( const commandLine& cmdLine, int positionArg=-1 );
	
	/**
	 *
	 */
	virtual ~videoSource();

	/**
	 *
	 */
	template<typename T> bool Capture( T** image, uint64_t timeout=UINT64_MAX )		{ return Capture((void**)image, imageFormatFromType<T>(), timeout); }
	
	/**
	 *
	 */
	virtual bool Capture( void** image, imageFormat format, uint64_t timeout=UINT64_MAX ) = 0;

	/**
	 * Begin streaming the camera.
	 * After Open() is called, frames from the camera will begin to be captured.
	 *
	 * Open() is not stricly necessary to call, if you call one of the Capture()
	 * functions they will first check to make sure that the stream is opened,
	 * and if not they will open it automatically for you.
	 *
	 * @returns `true` on success, `false` if an error occurred opening the stream.
	 */
	virtual bool Open();

	/**
	 * Stop streaming the camera.
	 * @note Close() is automatically called by the camera's destructor when
	 * it gets deleted, so you do not explicitly need to call Close() before
	 * exiting the program if you delete your camera object.
	 */
	virtual void Close();

	/**
	 * Check if the camera is streaming or not.
	 * @returns `true` if the camera is streaming (open), or `false` if it's closed.
	 */
	inline bool IsStreaming() const	   			{ return mStreaming; }

	/**
	 *
	 */
	inline uint32_t GetWidth() const				{ return mOptions.width; }

	/**
	 *
	 */
	inline uint32_t GetHeight() const				{ return mOptions.height; }
	
	/**
	 *
	 */
	inline uint32_t GetFrameRate() const			{ return mOptions.frameRate; }

	/**
	 *
	 */
	inline const URI& GetResource() const			{ return mOptions.resource; }

	/**
	 *
	 */
	inline const videoOptions& GetOptions() const	{ return mOptions; }

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
	//videoSource();
	videoSource( const videoOptions& options );

	bool         mStreaming;
	videoOptions mOptions;
};

#endif
