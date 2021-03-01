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
 
#include "Process.h"

#include "filesystem.h"
#include <string.h>

 
// GetID
pid_t Process::GetID()
{
	return getpid();
}
	

// GetParentID
pid_t Process::GetParentID()
{
	return getppid();
}


// Fork
void Process::Fork()
{
	fork();
}


// ExecutablePath
std::string Process::ExecutablePath()
{
	char buf[512];
	memset(buf, 0, sizeof(buf));	// readlink() does not NULL-terminate

	const ssize_t size = readlink("/proc/self/exe", buf, sizeof(buf));

	if( size <= 0 )
		return "";
	
	return std::string(buf);
}


// ExecutableDirectory
std::string Process::ExecutableDirectory()
{
	const std::string path = ExecutablePath();

	if( path.length() == 0 )
		return "";

	return pathDir(path);
}


// WorkingDirectory
std::string Process::WorkingDirectory()
{
	char buf[1024];

	char* str = getcwd(buf, sizeof(buf));

	if( !str )
		return "";

	return buf;
}



