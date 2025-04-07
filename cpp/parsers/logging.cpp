/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
 
#include "logging.h"
#include <strings.h>


// set default logging options
Log::Level Log::mLevel = Log::DEFAULT;
FILE* Log::mFile = stdout;
std::string Log::mFilename = "stdout";


// ParseCmdLine
void Log::ParseCmdLine( const int argc, char** argv )
{
	ParseCmdLine(commandLine(argc, argv));
}


// ParseCmdLine
void Log::ParseCmdLine( const commandLine& cmdLine )
{
	const char* levelStr = cmdLine.GetString("log-level");

	if( levelStr != NULL )
	{
		SetLevel(LevelFromStr(levelStr));
	}
	else
	{
		if( cmdLine.GetFlag("verbose") )
			SetLevel(VERBOSE);

		if( cmdLine.GetFlag("debug") )
			SetLevel(DEBUG);		
	}

	SetFile(cmdLine.GetString("log-file"));
	
	// disable buffering so that the post-newline color resets are used (https://stackoverflow.com/a/1716621)
	if( mFile == stdout )
		setbuf(stdout, NULL);
}


// SetFile
void Log::SetFile( FILE* file )
{
	if( !file || mFile == file )
		return;

	mFile = file;

	if( mFile == stdout )
		mFilename = "stdout";
	else if( mFile == stderr )
		mFilename = "stderr";
}


// SetFilename
void Log::SetFile( const char* filename )
{
	if( !filename )
		return;

	if( strcasecmp(filename, "stdout") == 0 )
		SetFile(stdout);
	else if( strcasecmp(filename, "stderr") == 0 )
		SetFile(stderr);
	else
	{
		if( strcasecmp(filename, mFilename.c_str()) == 0 )
			return;

		FILE* file = fopen(filename, "w"); 

		if( file != NULL )
		{
			SetFile(file);
			mFilename = filename;
		}
		else
		{
			LogError("failed to open '%s' for logging\n", filename);
			return;
		}
	}	
}

// LevelToStr
const char* Log::LevelToStr( Log::Level level )
{
	switch(level)
	{
		case SILENT:	return "silent";
		case ERROR:    return "error";
		case WARNING:  return "warning";
		case SUCCESS:  return "success";
		case INFO:	return "info";
		case VERBOSE:	return "verbose";
		case DEBUG:	return "debug";
	}

	return "default";
}


// LevelFromStr
Log::Level Log::LevelFromStr( const char* str )
{
	if( !str )
		return DEFAULT;

	for( int n=0; n <= DEBUG; n++ )
	{
		const Level level = (Level)n;

		if( strcasecmp(str, LevelToStr(level)) == 0 )
			return level;
	}

	if( strcasecmp(str, "disable") == 0 || strcasecmp(str, "disabled") == 0 || strcasecmp(str, "none") == 0 )
		return SILENT;

	return DEFAULT;
}



	
