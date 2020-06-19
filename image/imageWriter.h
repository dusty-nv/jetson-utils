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
 
#ifndef __IMAGE_WRITER_H_
#define __IMAGE_WRITER_H_


#include "videoOutput.h"


/**
 * @ingroup image
 */
class imageWriter : public videoOutput
{
public:
	/**
	 *
	 */
	static imageWriter* Create( const char* path, const videoOptions& options=videoOptions() );

	/**
	 *
	 */
	static imageWriter* Create( const videoOptions& options );

	/**
	 *
	 */
	virtual ~imageWriter();

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
	virtual inline uint32_t GetType() const		{ return Type; }

	/**
	 *
	 */
	static const uint32_t Type = (1 << 5);

	/**
	 *
	 */
	static const char* SupportedExtensions[];

	/**
	 *
	 */
	static bool IsSupportedExtension( const char* ext );


protected:
	imageWriter( const videoOptions& options );

	uint32_t mFileCount;
	char     mFileOut[1024];
};

#endif
