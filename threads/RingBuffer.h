/*
 * Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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

#ifndef __MULTITHREAD_RINGBUFFER_H_
#define __MULTITHREAD_RINGBUFFER_H_

#include "Mutex.h"


/**
 * Ringbuffer queue
 * @ingroup threads
 */
class RingBuffer
{
public:
	/**
	 *
	 */
	enum Flags
	{
		Read           = (1 << 0),
		ReadOnce       = (1 << 1) | Read,
		ReadLatest     = (1 << 2) | Read,
		ReadLatestOnce = (1 << 3) | ReadLatest,
		Write          = (1 << 4),
		Threaded       = (1 << 5),      
		ZeroCopy       = (1 << 6),
	};
	
	/**
	 *
	 */
	inline RingBuffer( uint32_t flags=Threaded );

	/**
	 * Destructor
	 */
	inline ~RingBuffer();

	/**
	 *
	 */
	inline bool Alloc( uint32_t numBuffers, size_t size, uint32_t flags=0 );

	/**
	 * 
	 */
	inline void Free();

	/**
	 *
	 */
	inline void* Peek( uint32_t flags );

	/**
	 *
	 */
	inline void* Next( uint32_t flags );

	/**
	 *
	 */
	inline uint32_t GetFlags() const;
	
	/**
	 *
	 */
	inline void SetFlags( uint32_t flags );
	
	/**
	 *
	 */
	inline void SetThreaded( bool threaded );

protected:

	uint32_t mNumBuffers;
	uint32_t mLatestRead;
	uint32_t mLatestWrite;
	uint32_t mFlags;

	void** mBuffers;
	size_t mBufferSize;
	bool   mReadOnce;
	Mutex  mMutex;
};

// inline implementations
#include "RingBuffer.inl"

#endif

