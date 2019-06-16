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
 
#ifndef __MULTITHREAD_PROCESS_H_
#define __MULTITHREAD_PROCESS_H_

#include <sys/types.h>
#include <unistd.h>
#include <string>

	   
/**
 * Static functions for retrieving information about the running process.
 * @ingroup threads
 */
class Process
{
public:
	/**
	 * Get process ID (PID)
	 */
	static pid_t GetID();
	
	/**
	 * Get the parent's process ID
	 */
	static pid_t GetParentID();
	
	/**
	 * Duplicate the calling process
	 */
	static void Fork();

	/**
	 * Retrieve the absolute path of the calling process executable,
	 * including the process' filename.
	 */
	static std::string ExecutablePath();

	/**
	 * Retrieve the directory that the process executable resides in.
	 */
	static std::string ExecutableDirectory();

	/**
	 * Retrieve the current working directory of the process.
	 */
	static std::string WorkingDirectory();
};


#endif

