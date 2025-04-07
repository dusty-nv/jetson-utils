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
#include "logging.h"

#include <string.h>
#include <limits.h>


// readLink
static std::string readLink( const std::string& path )
{
	char buf[PATH_MAX];
	memset(buf, 0, sizeof(buf));	// readlink() does not NULL-terminate

	const ssize_t size = readlink(path.c_str(), buf, sizeof(buf));

	if( size <= 0 )
	{
		LogError("failed to read link %s\n", path.c_str());
		return "";
	}
	
	return std::string(buf);
}


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


// GetCommandLine
std::string Process::GetCommandLine( pid_t pid )
{
	const std::string path = "/proc/" + (pid < 0 ? "self" : std::to_string(pid)) + "/cmdline";
	char buf[PATH_MAX];
	
	FILE* file = fopen(path.c_str(), "rb");
	
	if( !file )
	{
		LogError("failed to open %s\n", path.c_str());
		return "";
	}
	
	const size_t bytes_read = fread(buf, 1, sizeof(buf), file);
	fclose(file);
	
	for( size_t n=0; n < bytes_read; n++ )
	{
		if( buf[n] == 0 )
			buf[n] = ' ';	// /proc/PID/cmdline strings are NULL-separated
	}

	return std::string(buf);
}


// GetExecutablePath
std::string Process::GetExecutablePath( pid_t pid )
{
	return readLink("/proc/" + (pid < 0 ? "self" : std::to_string(pid)) + "/exe");
}


// GetExecutableDir
std::string Process::GetExecutableDir( pid_t pid )
{
	const std::string path = GetExecutablePath(pid);

	if( path.length() == 0 )
		return "";

	return pathDir(path);
}


// GetWorkingDirectory
std::string Process::GetWorkingDir( pid_t pid )
{
	if( pid < 0 )
	{
		char buf[PATH_MAX];
		char* str = getcwd(buf, sizeof(buf));

		if( !str )
			return "";

		return std::string(buf);
	}
	else
	{
		return readLink("/proc/" + std::to_string(pid) + "/cwd");
	}
}


// Fork
void Process::Fork()
{
	fork();
}
