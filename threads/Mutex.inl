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

#ifndef __MULTITHREAD_MUTEX_INLINE_H
#define __MULTITHREAD_MUTEX_INLINE_H


// constructor
inline Mutex::Mutex()
{
	pthread_mutex_init(&mID, NULL);
}


// destructor
inline Mutex::~Mutex()
{
	pthread_mutex_destroy(&mID);
}


// AttemptLock
inline bool Mutex::AttemptLock()					
{ 
	return (pthread_mutex_trylock(&mID) == 0); 
}
	

// Lock
inline void Mutex::Lock()							
{ 
	pthread_mutex_lock(&mID); 
}


// Unlock
inline void Mutex::Unlock()
{ 
	pthread_mutex_unlock(&mID); 
}		


// Sync
inline void Mutex::Sync()							
{ 
	Lock(); 
	Unlock(); 
}


// GetID
inline pthread_mutex_t* Mutex::GetID()
{ 
	return &mID; 
}
	
#endif
