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

	mMapDevice = 0;
	mMapFlags  = 0;
	
	mNumElements = 0;
	mElementSize = 0;
	mInteropCUDA = NULL;
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
glBuffer* glBuffer::Create( uint32_t type, uint32_t size, void* data, uint32_t usage )
{
	glBuffer* buf = new glBuffer();

	if( !buf )
	{
		LogError(LOG_GL "failed to construct new glBuffer object\n");
		return NULL;
	}

	if( !buf->init(type, size, data, usage) )
	{
		LogError(LOG_GL "failed to create buffer (%u bytes)\n", size);
		delete buf;		
		return NULL;
	}

	return buf;
}


// Create
glBuffer* glBuffer::Create( uint32_t type, uint32_t numElements, uint32_t elementSize, void* data, uint32_t usage )
{
	glBuffer* buf = new glBuffer();

	if( !buf )
	{
		LogError(LOG_GL "failed to construct new glBuffer object\n");
		return NULL;
	}

	if( !buf->init(type, numElements * elementSize, data, usage) )
	{
		LogError(LOG_GL "failed to create buffer (%u bytes)\n", numElements * elementSize);
		delete buf;		
		return NULL;
	}

	buf->mNumElements = numElements;
	buf->mElementSize = elementSize;

	return buf;
}


// init
bool glBuffer::init( uint32_t type, uint32_t size, void* data, uint32_t usage )
{
	GL_VERIFY(glGenBuffers(1, &mID));
	GL_VERIFY(glBindBuffer(type, mID));
	GL_VERIFY(glBufferData(type, size, data, usage));
	GL_VERIFY(glBindBuffer(type, 0));

	mType  = type;
	mSize  = size;
	mUsage = usage;

	mNumElements = size;
	mElementSize = 1;

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


// cudaGraphicsRegisterFlagsFromGL
cudaGraphicsRegisterFlags cudaGraphicsRegisterFlagsFromGL( uint32_t flags )
{
#if defined(__x86_64__) || defined(__amd64__)
	// there's a memory access issue on x86 when the flags below are used, so disable them
	return cudaGraphicsRegisterFlagsNone;
#else
	if( flags == GL_WRITE_DISCARD )
		return cudaGraphicsRegisterFlagsWriteDiscard;
	else if( flags == GL_READ_ONLY )
		return cudaGraphicsRegisterFlagsReadOnly;
	else
		return cudaGraphicsRegisterFlagsNone;
#endif
}


// Map
void* glBuffer::Map( uint32_t device, uint32_t flags, cudaStream_t stream )
{
	if( mMapDevice != 0 )
	{
		LogError(LOG_GL "error -- glBuffer is already mapped (call Unmap() first)\n");
		return NULL;
	}

	if( !Bind() )
		return NULL;

	if( device == GL_MAP_CPU )
	{
		// invalidate the old buffer so we can map without stalling
		if( flags == GL_WRITE_DISCARD )
		{
			GL(glBufferData(mType, mSize, NULL, mUsage));
			flags = GL_WRITE_ONLY; // GL expects GL_WRITE_ONLY
		}

		// lock the buffer
		void* ptr = glMapBuffer(mType, flags);

		if( !ptr )
		{
			LogError(LOG_GL "glMapBuffer() failed\n");
			GL_CHECK("glMapBuffer()\n");			
			return NULL;
		}

		mMapDevice = device;
		return ptr;
	}
	else if( device == GL_MAP_CUDA )
	{
		if( !mInteropCUDA )
		{
			if( CUDA_FAILED(cudaGraphicsGLRegisterBuffer(&mInteropCUDA, mID, cudaGraphicsRegisterFlagsFromGL(flags))) )
				return NULL;

			LogSuccess(LOG_CUDA "registered OpenGL buffer for interop access (%u bytes)\n", mSize);

			if( mUsage != GL_DYNAMIC_DRAW )
			{
				LogWarning(LOG_CUDA "warning: OpenGL interop buffer was not created with GL_DYNAMIC_DRAW\n");
				LogWarning(LOG_CUDA "it's recommended that GL interopability buffers use GL_DYNAMIC_DRAW\n");
			}		
		}

		if( mMapFlags != 0 && mMapFlags != flags )
			CUDA(cudaGraphicsResourceSetMapFlags(mInteropCUDA, cudaGraphicsRegisterFlagsFromGL(flags)));

		if( CUDA_FAILED(cudaGraphicsMapResources(1, &mInteropCUDA, stream)) )
			return NULL;

		// map CUDA device pointer
		void*  devPtr     = NULL;
		size_t mappedSize = 0;

		if( CUDA_FAILED(cudaGraphicsResourceGetMappedPointer(&devPtr, &mappedSize, mInteropCUDA)) )
		{
			CUDA(cudaGraphicsUnmapResources(1, &mInteropCUDA, stream));
			return NULL;
		}
		
		if( mSize != mappedSize )
			LogWarning(LOG_GL "glBuffer::Map() -- CUDA size mismatch %zu bytes  (expected=%u)\n", mappedSize, mSize);
		
		mMapDevice = device;
		mMapFlags = flags;	// these only need tracked for GPU	

		return devPtr;
	}

	LogError(LOG_GL "glBuffer::Map() -- invalid device (must be GL_MAP_CPU or GL_MAP_CUDA)\n");
	return NULL;
}


// Unmap
void glBuffer::Unmap( cudaStream_t stream )
{
	if( mMapDevice != GL_MAP_CPU && mMapDevice != GL_MAP_CUDA )
		return;

	if( !Bind() )
		return;

	if( mMapDevice == GL_MAP_CPU )
	{
		GL(glUnmapBuffer(mType));
	}
	else if( mMapDevice == GL_MAP_CUDA )
	{
		if( !mInteropCUDA )
			return;

		CUDA(cudaGraphicsUnmapResources(1, &mInteropCUDA, stream));
	}

	mMapDevice = 0;
	Unbind();
}


// Copy
bool glBuffer::Copy( void* ptr, uint32_t offset, uint32_t size, uint32_t flags, cudaStream_t stream )
{
	if( !ptr || size == 0 || size >= mSize || offset >= mSize || offset > (mSize - size) )
		return false;

	uint32_t mapFlags = GL_READ_ONLY;

	if( flags == GL_FROM_CPU || flags == GL_FROM_CUDA )
	{
		if( size == mSize )
			mapFlags = GL_WRITE_DISCARD;
		else
			mapFlags = GL_WRITE_ONLY;
	}
	
	if( flags == GL_FROM_CPU )
	{
		// TODO for faster CPU path, see http://hacksoflife.blogspot.com/2015/06/glmapbuffer-no-longer-cool.html
		void* dst = Map(GL_MAP_CPU, mapFlags, stream);

		if( !dst )
			return false;

		memcpy((uint8_t*)dst + offset, ptr, size);
	}
	else if( flags == GL_FROM_CUDA )
	{
		void* dst = Map(GL_MAP_CUDA, mapFlags, stream);

		if( !dst )
			return false;

		if( CUDA_FAILED(cudaMemcpyAsync((uint8_t*)dst + offset, ptr, size, cudaMemcpyDeviceToDevice, stream)) )
		{
			Unmap(stream);
			return false;
		}
	}
	else if( flags == GL_TO_CPU )
	{
		void* src = Map(GL_MAP_CPU, mapFlags, stream);

		if( !src )
			return false;
	
		memcpy(ptr, (uint8_t*)src + offset, size);
	}
	else if( flags == GL_TO_CUDA )
	{
		void* src = Map(GL_MAP_CUDA, mapFlags, stream);

		if( !src )
			return false;
	
		if( CUDA_FAILED(cudaMemcpyAsync(ptr, (uint8_t*)src + offset, size, cudaMemcpyDeviceToDevice, stream)) )
		{
			Unmap(stream);
			return false;
		}
	}

	Unmap(stream);
	return true;
}

// Copy
bool glBuffer::Copy( void* ptr, uint32_t size, uint32_t flags, cudaStream_t stream )
{
	return Copy(ptr, 0, size, flags, stream);
}

// Copy
bool glBuffer::Copy( void* ptr, uint32_t flags, cudaStream_t stream )
{
	return Copy(ptr, 0, mSize, flags, stream);
}



