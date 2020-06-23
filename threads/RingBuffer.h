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
 * Thread-safe circular ring buffer queue
 * @ingroup threads
 */
class RingBuffer
{
public:
	/**
	 * Ring buffer flags
	 */
	enum Flags
	{
		Read           = (1 << 0),				/**< Read the next buffer. */
		ReadOnce       = (1 << 1) | Read,			/**< Read the next buffer, but only if it hasn't been read before. */
		ReadLatest     = (1 << 2) | Read,			/**< Read the latest buffer in the queue, skipping other buffers that may not have been read. */
		ReadLatestOnce = (1 << 3) | ReadLatest,		/**< Combination of ReadOnce and ReadLatest flags. */
		Write          = (1 << 4),				/**< Write the next buffer. */
		Threaded       = (1 << 5),      			/**< Buffers should be thread-safe (enabled by default). */
		ZeroCopy       = (1 << 6),				/**< Buffers should be allocated in mapped CPU/GPU zeroCopy memory (otherwise GPU only) */
	};
	
	/**
	 * Construct a new ring buffer.
	 */
	inline RingBuffer( uint32_t flags=Threaded );

	/**
	 * Destructor
	 */
	inline ~RingBuffer();

	/**
	 * Allocate memory for a set of buffers, where each buffer has the specified size.
	 *
	 * If the requested allocation is compatible with what was already allocated,
	 * this will return `true` without performing additional allocations.
	 * Otherwise, the previous buffers are released and new ones are allocated.
	 *
	 * @returns `true` if the allocations succeeded or was previously done.
	 *          `false` if a memory allocation error occurred.
	 */
	inline bool Alloc( uint32_t numBuffers, size_t size, uint32_t flags=0 );

	/**
	 * Free the buffer allocations.
	 */
	inline void Free();

	/**
	 * Get the next read/write buffer without advancing the position in the queue.
	 */
	inline void* Peek( uint32_t flags );

	/**
	 * Get the next read/write buffer and advance the position in the queue.
	 */
	inline void* Next( uint32_t flags );

	/**
	 * Get the flags of the ring buffer.
	 */
	inline uint32_t GetFlags() const;
	
	/**
	 * Set the ring buffer's flags.
	 */
	inline void SetFlags( uint32_t flags );
	
	/**
	 * Enable or disable multi-threading.
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

