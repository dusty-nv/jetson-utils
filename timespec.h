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

#ifndef __TIMESPEC_UTIL_H__
#define __TIMESPEC_UTIL_H__

#include <time.h>
#include <stdint.h>
#include <stdio.h>

#include "logging.h"


/**
 * Retrieve a timestamp of the current system time.
 * @ingroup time
 */
inline void timestamp( timespec* timestampOut )					{ if(!timestampOut) return; timestampOut->tv_sec=0; timestampOut->tv_nsec=0; clock_gettime(CLOCK_REALTIME, timestampOut); } 

/**
 * Retrieve a timestamp of the current system time.
 * @ingroup time
 */
inline timespec timestamp()									{ timespec t; timestamp(&t); return t; }

/**
 * Return a blank timespec that's been zero'd.
 * @ingroup time
 */
inline timespec timeZero()									{ timespec t; t.tv_sec=0; t.tv_nsec=0; return t; }

/**
 * Return an initialized `timespec`
 * @ingroup time
 */
inline timespec timeNew( time_t seconds, long int nanoseconds )		{ timespec t; t.tv_sec=seconds; t.tv_nsec=nanoseconds; return t; }

/**
 * Return an initialized `timespec`
 * @ingroup time
 */
inline timespec timeNew( long int nanoseconds )					{ const time_t sec=nanoseconds/1e+9; return timeNew(sec, nanoseconds-sec*1e+9); }

/**
 * Add two times together.
 * @ingroup time
 */
inline timespec timeAdd( const timespec& a, const timespec& b )		{ timespec t; t.tv_sec=a.tv_sec+b.tv_sec; t.tv_nsec=a.tv_nsec+b.tv_nsec; const time_t sec=t.tv_nsec/1e+9; t.tv_sec+=sec; t.tv_nsec-=sec*1e+9; return t; }

/**
 * Find the difference between two timestamps.
 * @ingroup time
 */
inline void timeDiff( const timespec& start, const timespec& end, timespec* result )
{
	if ((end.tv_nsec-start.tv_nsec)<0) {
		result->tv_sec = end.tv_sec-start.tv_sec-1;
		result->tv_nsec = 1000000000+end.tv_nsec-start.tv_nsec;
	} else {
		result->tv_sec = end.tv_sec-start.tv_sec;
		result->tv_nsec = end.tv_nsec-start.tv_nsec;
	}
}

/**
 * Find the difference between two timestamps.
 * @ingroup time
 */
inline timespec timeDiff( const timespec& start, const timespec& end )
{
	timespec result;
	timeDiff(start, end, &result);
	return result;
}

/**
 * Compare two timestamps.
 *
 * @return 0, if timestamp A equals timestamp B
 *        >0, if timestamp A is greater than timestamp B
 *        <0, if timestamp A is less than timestamp B
 *
 * @ingroup time
 */
inline int timeCmp( const timespec& a, const timespec& b )
{
	if( a.tv_sec < b.tv_sec )
		return -1;
	else if( a.tv_sec > b.tv_sec )
		return 1;
	else
	{
		if( a.tv_nsec < b.tv_nsec )
			return -1;
		else if( a.tv_nsec > b.tv_nsec )
			return 1;
		else
			return 0;
	}
}

/**
 * @internal Reference timestamp of when the process started.
 * @ingroup time
 */
extern const timespec __apptime_begin__;

/**
 * Retrieve the elapsed time since the process started.
 * @ingroup time
 */
inline void apptime( timespec* a )										{ timespec t; timestamp(&t); timeDiff(__apptime_begin__, t, a); }

/**
 * Retrieve the elapsed time since the process started (in seconds).
 * @ingroup time
 */
inline float apptime()												{ timespec t; apptime(&t); return t.tv_sec + t.tv_nsec * 0.000000001f; }

/**
 * Convert to 32-bit float (in milliseconds).
 * @ingroup time
 */
inline float timeFloat( const timespec& a )								{ return a.tv_sec * 1000.0f + a.tv_nsec * 0.000001f; }

/**
 * Convert to 64-bit double (in milliseconds).
 * @ingroup time
 */
inline double timeDouble( const timespec& a )							{ return a.tv_sec * 1000.0 + a.tv_nsec * 0.000001; }

/**
 * Get current timestamp as 64-bit double (in milliseconds).
 * @ingroup time
 */
inline double timeDouble()											{ return timeDouble(timestamp()); }

/**
 * Produce a text representation of the timestamp.
 * @ingroup time
 */
inline char* timeStr( const timespec& timestamp, char* strOut )				{ sprintf(strOut, "%lus + %010luns", (uint64_t)timestamp.tv_sec, (uint64_t)timestamp.tv_nsec); return strOut; }

/**
 * Print the time to stdout.
 * @ingroup time
 */
inline void timePrint( const timespec& timestamp, const char* text=NULL )		{ LogInfo("%s   %lus + %010luns\n", text, (uint64_t)timestamp.tv_sec, (uint64_t)timestamp.tv_nsec); }

/**
 * Put the current thread to sleep for a specified time.
 * @ingroup time
 */
inline void sleepTime( const timespec& duration )							{ nanosleep(&duration, NULL); }

/**
 * Put the current thread to sleep for a specified time.
 * @ingroup time
 */
inline void sleepTime( time_t seconds, long int nanoseconds )				{ sleepTime(timeNew(seconds,nanoseconds)); } 

/**
 * Put the current thread to sleep for a specified number of milliseconds.
 * @ingroup time
 */
inline void sleepMs( uint64_t milliseconds )								{ sleepTime(timeNew(0, milliseconds * 1000 * 1000)); }

/**
 * Put the current thread to sleep for a specified number of microseconds.
 * @ingroup time
 */
inline void sleepUs( uint64_t microseconds )								{ sleepTime(timeNew(0, microseconds * 1000)); }

/**
 * Put the current thread to sleep for a specified number of nanoseconds.
 * @ingroup time
 */
inline void sleepNs( uint64_t nanoseconds )								{ sleepTime(timeNew(0, nanoseconds)); }


#endif

