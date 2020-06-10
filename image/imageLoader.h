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
 
#ifndef __IMAGE_LOADER_H_
#define __IMAGE_LOADER_H_


#include "videoSource.h"

#include <string>
#include <vector>


/**
 * @ingroup image
 */
class imageLoader : public videoSource
{
public:
	/**
	 * 
	 */
	static imageLoader* Create( const char* path, const videoOptions& options=videoOptions() );
	
	/**
	 * 
	 */
	static imageLoader* Create( const videoOptions& options );

	/**
	 *
	 */
	virtual ~imageLoader();

	/**
	 *
	 */
	template<typename T> bool Capture( T** image, uint64_t timeout=UINT64_MAX )		{ return Capture((void**)image, imageFormatFromType<T>(), timeout); }
	
	/**
	 *
	 */
	virtual bool Capture( void** image, imageFormat format, uint64_t timeout=UINT64_MAX );

	/**
	 * Open
	 */
	virtual bool Open();

	/**
	 * Close
	 */
	virtual void Close();

	/**
	 * IsEOS
	 */
	inline bool IsEOS() const				{ return mEOS; }

	/**
	 *
	 */
	virtual inline uint32_t GetType() const		{ return Type; }

	/**
	 *
	 */
	static const uint32_t Type = (1 << 4);

	/**
	 *
	 */
	static const char* SupportedExtensions[];

	/**
	 *
	 */
	static bool IsSupportedExtension( const char* ext );

protected:
	imageLoader( const videoOptions& options );

	inline bool isLooping() const { return (mOptions.loop < 0) || ((mOptions.loop > 0) && (mLoopCount < mOptions.loop)); }

	bool mEOS;
	size_t mLoopCount;
	size_t mNextFile;
	
	std::vector<std::string> mFiles;
	std::vector<void*> mBuffers;
};

#endif
