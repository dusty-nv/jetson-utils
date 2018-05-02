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
 
#ifndef __MULTITHREAD_EVENT_H_
#define __MULTITHREAD_EVENT_H_

#include "Mutex.h"


/**
 * Event
 */
class Event
{
public:
	/**
	 * Event constructor. By default, it will automatically be reset when it's raised.
	 * @param auto_reset Once this event has been raised, should it automatically be reset?
	 */
	Event( bool auto_reset=true );

	/**
	 * Destructor
	 */
	~Event();

	/**
	 * Wait until this event is raised.  It is likely this will block this thread (and will never timeout).
	 * @see Raise
	 */
	void Sync();

	/**
	 * Query the status of this event.
	 * @return True if the event is raised, false if not
	 */
	bool Query();

	/**
	 * Raise the event.  Any threads waiting to Sync() with this event will be awoken.
	 */
	void Raise();

	/**
	 * Reset the event status to un-raised.
	 */
	inline void Reset()									{ mQueryMutex.Lock(); mQuery = false; mQueryMutex.Unlock(); }

	/**
	 * Get the Event object
	 */
	inline pthread_cond_t* GetID()						{ return &mID; }

protected:

	pthread_cond_t mID;

	Mutex mQueryMutex;
	bool  mQuery;
	bool  mAutoReset;
};


#endif
