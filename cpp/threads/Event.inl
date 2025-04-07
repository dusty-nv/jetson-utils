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
 
#ifndef __MULTITHREAD_EVENT_INLINE_H
#define __MULTITHREAD_EVENT_INLINE_H

#include <errno.h>


// constructor
inline Event::Event( bool autoReset )
{
	mAutoReset = autoReset;
	mQuery     = false;
	
	pthread_cond_init(&mID, NULL);
}


// destructor
inline Event::~Event()
{
	pthread_cond_destroy(&mID);
}


// Query
inline bool Event::Query()
{
	bool r = false;
	mQueryMutex.Lock();
	r = mQuery;
	mQueryMutex.Unlock();
	return r;
}


// Wake
inline void Event::Wake()
{
	mQueryMutex.Lock();
	mQuery = true;
	pthread_cond_signal(&mID);
	mQueryMutex.Unlock();
}


// Reset
inline void Event::Reset()
{ 
	mQueryMutex.Lock(); 
	mQuery = false; 
	mQueryMutex.Unlock(); 
}


// Wait
inline bool Event::Wait()
{
	mQueryMutex.Lock();

	while(!mQuery)
		pthread_cond_wait(&mID, mQueryMutex.GetID());

	if( mAutoReset )
		mQuery = false;

	mQueryMutex.Unlock();
	return true;
}


// Wait
inline bool Event::Wait( const timespec& timeout )
{
	mQueryMutex.Lock();

	const timespec abs_time = timeAdd( timestamp(), timeout );

	while(!mQuery)
	{
		const int ret = pthread_cond_timedwait(&mID, mQueryMutex.GetID(), &abs_time);
		
		if( ret == ETIMEDOUT )
		{
			mQueryMutex.Unlock();
			return false;
		}
	}
	
	if( mAutoReset )
		mQuery = false;

	mQueryMutex.Unlock();
	return true;
}


// Wait
inline bool Event::Wait( uint64_t timeout )		
{ 
	return (timeout == UINT64_MAX) ? Wait() : Wait(timeNew(timeout*1000*1000));
}


// WaitNs
inline bool Event::WaitNs( uint64_t timeout )		
{ 
	return (timeout == UINT64_MAX) ? Wait() : Wait(timeNew(timeout)); 
}
	

// WaitUs
inline bool Event::WaitUs( uint64_t timeout )		
{ 
	return (timeout == UINT64_MAX) ? Wait() : Wait(timeNew(timeout*1000)); 
}


// GetID
inline pthread_cond_t* Event::GetID()	
{ 
	return &mID; 
}

	
#endif
