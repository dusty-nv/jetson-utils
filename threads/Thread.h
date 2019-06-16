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
 
#ifndef __MULTITHREAD_H_
#define __MULTITHREAD_H_

#include <pthread.h>


/**
 * Function pointer typedef representing a thread's main entry point.
 * A user-defined parameter is passed through such that the user can 
 * pass data or other value to their thread so that it can self-identify.
 * @ingroup threads
 */
typedef void* (*ThreadEntryFunction)( void* user_param );


/**
 * Thread class for launching an asynchronous operating-system dependent thread.
 * To make your own thread, provide a function pointer of the thread's entry point,
 * or inherit from this class and implement your own Run() function.
 * @ingroup threads
 */
class Thread
{
public:
	/**
	 * Default constructor
	 */
	Thread();

	/**
	 * Destructor.  Automatically stops the thread.
	 */
	virtual ~Thread();

	/**
	 * User-implemented thread main() function.
	 */
	virtual void Run();

	/**
	 * Start the thread.  This will asynchronously call the Run() function.
	 * @result False if an error occurred and the thread could not be started.
	 */
	bool StartThread();

	/**
	 * Start the thread, utilizing an entry function pointer provided by the user. 
	 * @result False if an error occurred and the thread could not be started.
	 */
	bool StartThread( ThreadEntryFunction entry, void* user_param=NULL );
	
	/**
	 * Halt execution of the thread.
	 */
	void StopThread();

	/**
	 * Prime the system for realtime use.  Mostly this is locking a large group of pages into memory.
	 */
	static void InitRealtime();
	
	/**
	 * Get the maximum priority level available
	 */
	static int GetMaxPriority();
	
	/**
	 * Get the minimum priority level avaiable
	 */
	static int GetMinPriority();

	/**
	 * Get the priority level of the thread.
	 * @param thread The thread, or if NULL, the currently running thread.
	 */
	static int GetPriority( pthread_t* thread=NULL );
 
	/**
	 * Set the priority level of the thread.
	 * @param thread The thread, or if NULL, the currently running thread.
	 */
	static int SetPriority( int priority, pthread_t* thread=NULL );

	/**
	 * Get this thread's priority level
	 */
	int GetPriorityLevel();

	/**
	 * Set this thread's priority level
	 */
	bool SetPriorityLevel( int priority );

	/**
	 * Whatever thread you are calling from, yield the processor for the specified number of milliseconds.
	 * Accuracy may vary wildly the lower you go, and depending on the platform.

	 */
	static void Yield( unsigned int ms );

	/**
	 * Get thread identififer
	 */
	inline pthread_t* GetThreadID() 						{ return &mThreadID; }

	/**
	 * Lock this thread to a CPU core.
	 */
	bool LockAffinity( unsigned int cpu );

	/**
	 * Lock the specified thread's affinity to a CPU core.
	 * @param cpu The CPU core to lock the thread to.
	 * @param thread The thread, or if NULL, the currently running thread.
	 */
	static bool SetAffinity( unsigned int cpu, pthread_t* thread=NULL );

	/**
	 * Look up which CPU core the thread is running on.
	 */
	static int GetCPU();

protected:



	static void* DefaultEntry( void* param );

	pthread_t mThreadID;
	bool mThreadStarted;
};

#endif
