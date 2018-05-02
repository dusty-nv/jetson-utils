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
 
#include "Thread.h"

#include <unistd.h>
#include <malloc.h>
#include <sys/mman.h>


// constructor
Thread::Thread()
{
	mThreadStarted = false;
}


// destructor
Thread::~Thread()
{
	StopThread();
}


// Run
void Thread::Run()
{
	printf("default Thread::Run() -- please implement your own Run() function\n");
}


// DefaultEntry
void* Thread::DefaultEntry( void* param )
{
	// the Thread object is contained in the param
	Thread* thread = (Thread*)param;
	
	// call the virtual Run() function
	thread->Run();

	// now that the thread has exited, make sure the object knows
	thread->mThreadStarted = false;
	return 0;
}


// StartThread
bool Thread::StartThread()
{
	return StartThread(&Thread::DefaultEntry, this);
}


// StartThread
bool Thread::StartThread( ThreadEntryFunction entry, void* user_param )
{
	// make sure this thread object hasn't already been started
	if( mThreadStarted )
	{
		printf( "Thread already initialized\n");
		return false;
	}

	// pthread_attr_setstacksize(&attr, PTHREAD_STACK_MIN + THREAD_STACK_SIZE);

	if( pthread_create(&mThreadID, NULL, entry, user_param) != 0 )
	{
		printf( "Thread failed to initialize\n");
		return false;
	}

	mThreadStarted = true;
	return true;
}


// StopThread
void Thread::StopThread()
{
	//pthread_join(mThreadID, NULL);
	mThreadStarted = false;
}

	
// GetMaxPriorityLevel
int Thread::GetMaxPriorityLevel()
{
	return sched_get_priority_min(SCHED_FIFO);
}


// GetMinPriorityLevel
int Thread::GetMinPriorityLevel()
{
	return sched_get_priority_max(SCHED_FIFO);
}


// GetPriorityLevel
int Thread::GetPriorityLevel()
{
	struct sched_param schedp;
	int policy = SCHED_FIFO;

	if( pthread_getschedparam(mThreadID, &policy, &schedp) != 0 )
	{
		printf("Thread::GetPriorityLevel() - Failed to retrieve thread's priority level\n");
		return 0;
	}

	return schedp.__sched_priority;
}


// SetPriorityLevel
bool Thread::SetPriorityLevel( int priority )
{
	struct sched_param schedp;
	schedp.__sched_priority = priority;

	if( pthread_setschedparam(mThreadID, SCHED_FIFO, &schedp) != 0 )
		return false;

	return true;
}


static int THREAD_STACK_SIZE = 200 * 1024;
static int PREALLOC_SIZE     = 200 * 1024 * 1024;

// InitRealtime
void Thread::InitRealtime()
{
	// disable paging for the current process
	mlockall(MCL_CURRENT | MCL_FUTURE);				// forgetting munlockall() when done!

	// turn off malloc trimming.
	mallopt(M_TRIM_THRESHOLD, -1);

	// turn off mmap usage.
	mallopt(M_MMAP_MAX, 0);

	unsigned int page_size = sysconf(_SC_PAGESIZE);
	unsigned char * buffer = (unsigned char *)malloc(PREALLOC_SIZE);

	// touch each page in this piece of memory to get it mapped into RAM
	for(int i = 0; i < PREALLOC_SIZE; i += page_size)
	{
		// each write to this buffer will generate a pagefault.
		// once the pagefault is handled a page will be locked in memory and never
		// given back to the system.
		buffer[i] = 0;
	}
		
	// release the buffer. As glibc is configured such that it never gives back memory to
	// the kernel, the memory allocated above is locked for this process. All malloc() and new()
	// calls come from the memory pool reserved and locked above. Issuing free() and delete()
	// does NOT make this locking undone. So, with this locking mechanism we can build applications
	// that will never run into a major/minor pagefault, even with swapping enabled.
	free(buffer);
}


// Yield
void Thread::Yield( unsigned int ms )
{
	sleep(ms);
}
