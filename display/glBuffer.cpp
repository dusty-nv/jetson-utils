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

#include "glUtility.h"
#include "glBuffer.h"


// constructor
glBuffer::glBuffer()
{
	mID    = 0;
	mSize  = 0;
	mType  = 0;
	mUsage = 0;

	mNumElements = 0;
	mElementSize = 0;
}


// destructor
glBuffer::~glBuffer()
{
	if( mID != 0 )
	{
		GL(glDeleteBuffers(1, &mID));
		mID = 0;
	}
}


// Create
glBuffer* glBuffer::Create( uint32_t type, uint32_t numElements, uint32_t elementSize, void* data, uint32_t usage )
{
	if( !numElements || !elementSize )
		return NULL;

	// allocate new buffer
	glBuffer* buf = new glBuffer();

	if( !buf || !buf->init(type, numElements, elementSize, data, usage) )
	{
		printf(LOG_GL "failed to create buffer (%u bytes)\n", numElements * elementSize);
		return NULL;
	}

	return buf;
}


// init
bool glBuffer::init( uint32_t type, uint32_t numElements, uint32_t elementSize, void* data, uint32_t usage )
{
	const uint32_t size = numElements * elementSize;

	GL_VERIFY(glGenBuffers(1, &mID));
	GL_VERIFY(glBindBuffer(type, mID));
	GL_VERIFY(glBufferData(type, size, data, usage));
	GL_VERIFY(glBindBuffer(type, 0));

	mType  = type;
	mSize  = size;
	mUsage = usage;

	mNumElements = numElements;
	mElementSize = elementSize;

	return true;
}


// Bind
bool glBuffer::Bind()
{
	if( !mID )
		return false;

	GL_VERIFY(glBindBuffer(mType, mID));
	return true;
}


// Unbind
void glBuffer::Unbind()
{
	glBindBuffer(mType, 0);
}


// Lock
void* glBuffer::Lock( uint lockMode )
{
	if( !Bind() )
		return NULL;

	// if we are only writing, discard the old buffer so we can lock without stalling the CPU
	if( lockMode == GL_WRITE_ONLY )
		GL(glBufferData(mType, mSize, NULL, mUsage));

	// lock the buffer
	void* data = glMapBuffer(mType, lockMode);

	if( !data )
	{
		printf(LOG_GL "glMapBuffer() failed\n");
		return NULL;
	}

	return data;
}


// Unlock
void glBuffer::Unlock()
{
	if( !Bind() )
		return;

	GL(glUnmapBuffer(mType));
}


