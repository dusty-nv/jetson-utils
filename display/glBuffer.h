/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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
 
#ifndef __GL_BUFFER_H__
#define __GL_BUFFER_H__


#include "cudaUtility.h"
#include "cuda_gl_interop.h"


enum
{
	/**
	 * Alias for vertex buffers
	 * @ingroup OpenGL
	 */
	GL_VERTEX_BUFFER = GL_ARRAY_BUFFER,

	/**
	 * Alias for index buffers
	 * @ingroup OpenGL
	 */
	GL_INDEX_BUFFER = GL_ELEMENT_ARRAY_BUFFER
};


/**
 * OpenGL buffer with CUDA interoperability.
 * @ingroup OpenGL
 */
class glBuffer
{
public:
	/**
	 * Allocate an OpenGL buffer.
	 * @param type either GL_ARRAY_BUFFER for a vertex buffer, or GL_ELEMENT_ARRAY_BUFFER for an index buffer
	 * @param numElements the number of elements (i.e. vertices or indices) in the buffer
	 * @param elementSize the size in bytes of each element
	 * @param data pointer to the initial memory of this buffer
	 * @param usage GL_STATIC_DRAW (never updated), GL_STREAM_DRAW (once per frame), or GL_DYNAMIC_DRAW (multiple times per frame)
	 */
	static glBuffer* Create( uint32_t type, uint32_t numElements, uint32_t elementSize, void* data=NULL, uint32_t usage=GL_STATIC_DRAW );
	
	/**
	 * Free the buffer
	 */
	~glBuffer();
	
	/**
	 * Activate using the buffer
	 */
	bool Bind();

	/**
	 * Deactivate using the buffer
	 */
	void Unbind();

	/**
	 * Lock the buffer for accessing from CPU
	 * @param lockMode GL_WRITE_ONLY, GL_READ_ONLY, or GL_READ_WRITE
	 */
	void* Lock( uint32_t lockMode );

	/**
	 * Unlock the buffer for access
	 */
	void Unlock();
	
	/**
	 * Retrieve the OpenGL resource handle of the buffer.
	 */
	inline uint32_t GetID() const			{ return mID; }

	/**
	 * Retrieve the number of elements (i.e. vertices or indices)
	 */
	inline uint32_t GetNumElements() const	{ return mNumElements; }

	/**
	 * Retrieve the size in bytes of each element
	 */
	inline uint32_t GetElementSize() const	{ return mElementSize; }

	/**
	 * Retrieve the total size in bytes of the buffer
	 */
	inline uint32_t GetSize() const		{ return mSize; }

	/**
	 * Retrieve the buffer type (GL_VERTEX_BUFFER or GL_INDEX_BUFFER)
	 */
	inline uint32_t GetType() const		{ return mType; }
	
private:
	glBuffer();

	bool init( uint32_t type, uint32_t numElements, uint32_t elementSize, void* data, uint32_t usage);
	
	uint32_t mID;
	uint32_t mNumElements;
	uint32_t mElementSize;
	uint32_t mSize;
	uint32_t mType;
	uint32_t mUsage;
};


#endif
