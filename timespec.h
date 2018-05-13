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


/**
 * Retrieve a timestamp of the current system time
 */
inline void timestamp( timespec* timestampOut )		{ if(!timestampOut) return; timestampOut->tv_sec=0; timestampOut->tv_nsec=0; clock_gettime(CLOCK_REALTIME, timestampOut); } 


/**
 * Retrieve a timestamp of the current system time
 */
inline timespec timestamp()							{ timespec t; timestamp(&t); return t; }


/**
 * Return a blank timespec that's been zero'd
 */
inline timespec timeZero()							{ timespec t; t.tv_sec=0; t.tv_nsec=0; return t; }


/**
 * Return an initialized timespec
 */
inline timespec timeNew( time_t seconds, long int nanoseconds )		{ timespec t; t.tv_sec=seconds; t.tv_nsec=nanoseconds; }


/**
 * Return an initialized timespec
 */
inline timespec timeNew( long int nanoseconds )						{ const time_t sec=nanoseconds/1e-9; return timeNew(sec, nanoseconds-sec*1e-9); }


/**
 * Add two times together
 */
inline timespec timeAdd( const timespec& a, const timespec& b )		{ timespec t; t.tv_sec=a.tv_sec+b.tv_sec; t.tv_nsec=a.tv_nsec+b.tv_nsec; const time_t sec=t.tv_nsec/1e-9; t.tv_sec+=sec; t.tv_nsec-=sec*1e-9; return t; }


/**
 * Find the difference between two timestamps
 */
timespec timeDiff( const timespec& start, const timespec& end );


/**
 * Compare two timestamps
 * @return 0, if timestamp A equals timestamp B
 *        >0, if timestamp A is greater than timestamp B
 *        <0, if timestamp A is less than timestamp B
 */
int timeCmp( const timespec& a, const timespec& b );


/**
 * Produce a text representation of the timestamp
 */
inline char* timeStr( const timespec& timestamp, char* strOut )				{ sprintf(strOut, "%lu s  %lu ns", (uint64_t)timestamp.tv_sec, (uint64_t)timestamp.tv_nsec); return strOut; }


/**
 * Print the time to stdout
 */
inline void timePrint( const timespec& timestamp, const char* text=NULL )	{ printf("%s   %lu s  %010lu ns\n", text, (uint64_t)timestamp.tv_sec, (uint64_t)timestamp.tv_nsec); }


/**
 * Put the current thread to sleep for a specified time.
 */
inline void sleepTime( const timespec& duration )							{ nanosleep(&duration, NULL); }


/**
 * Put the current thread to sleep for a specified time.
 */
inline void sleepTime( time_t seconds, long int nanoseconds )				{ sleepTime(timeNew(seconds,nanoseconds)); } 


/**
 * Put the current thread to sleep for a specified number of milliseconds.
 */
inline void sleepMs( uint64_t milliseconds )								{ sleepTime(timeNew(0, milliseconds * 1000 * 1000)); }


/**
 * Put the current thread to sleep for a specified number of microseconds.
 */
inline void sleepUs( uint64_t microseconds )								{ sleepTime(timeNew(0, microseconds * 1000)); }


/**
 * Put the current thread to sleep for a specified number of nanoseconds.
 */
inline void sleepNs( uint64_t nanoseconds )									{ sleepTime(timeNew(0, nanoseconds)); }


#endif
