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

#ifndef __MULTITHREAD_MUTEX_H_
#define __MULTITHREAD_MUTEX_H_

#include <pthread.h>


/**
 * A lightweight mutual exclusion lock.  It is very fast to check if the mutex is available,
 * lock it, and release it.  However, if the mutex is unavailable when you attempt to
 * lock it, execution of the thread will stop until it becomes available.
 * @ingroup threads
 */
class Mutex
{
public:
	/**
	 * Constructor
	 */
	inline Mutex();

	/**
	 * Destructor
	 */
	inline ~Mutex();

	/**
	 * If the lock is free, aquire it.  Otherwise, return without waiting for it to become available.
	 * @result True if the lock was aquired, false if not.
	 */
	inline bool AttemptLock();
	
	/**
	 * Aquire the lock, whenever it becomes available.  This could mean just a few instructions
	 * if the lock is already free, or to block the thread if it isn't.
	 */
	inline void Lock();

	/**
	 * Release the lock
	 */
	inline void Unlock();	

	/**
	 * Wait for the lock, then release it immediately.  Use this in situations where you are waiting for
	 * an event to occur.
	 */
	inline void Sync();

	/**
	 * Get the mutex object
	 */
	inline pthread_mutex_t* GetID();

protected:
	pthread_mutex_t mID;
};

// inline implementations
#include "Mutex.inl"

#endif
